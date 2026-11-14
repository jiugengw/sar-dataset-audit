"""SARDet-100K leakage audit, stage 3: shifted copies (wrapped round, or overlapping tiles), which stage 1 misses.

The negative control in controls.py found val chips that match train chips rolled by half (np.roll). wrapcheck.py
later showed these are plain shifts (neighbouring crops), not wrapped images: the roll score is just a cheap way to
find them. Overlapping tiles are the same thing with less shared area. Both keep the strength of each spatial frequency (|FFT|), so the search runs on that.

1. Candidates: |FFT| of each zero-mean 64 px thumbnail (log, low-frequency half plane, no DC), under all 8 flips /
   turns, FAISS inner product against train; the top 5 train images per query are kept.
2. Check, at full resolution: the query is turned the matched way (and resized to the reference size); FFT cross-
   correlation of the 4x-averaged images gives the shift s with roll(query, s) ~ reference. Then
     wrap_full / wrap_low : Pearson of roll(query, s) and the reference (full resolution / 8x averaged);
     ovl_full / ovl_low   : the same over the overlap only, without wrapping (overlap share in ovl_frac).
   The best of the 5 candidates (by max(wrap_low, ovl_low)) is kept.
3. Labels (val only): the train boxes are moved by -s and turned back; box_recall = share of the val boxes (with
   their centre in the region the two images share) recovered at IoU >= 0.5 with the same class.

    ~/.venvs/sar-audit/bin/python sardet100k/shifts.py val    # -> results/shifts_val.csv
    ~/.venvs/sar-audit/bin/python sardet100k/shifts.py test
    ~/.venvs/sar-audit/bin/python sardet100k/shifts.py train  # train vs train (for resplit.py)
    ~/.venvs/sar-audit/bin/python sardet100k/shifts.py val test  # val vs test -> results/shifts_val_vs_test.csv
"""
import csv, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import candidates as C
import verify as V

K = 5; F = 16  # candidates per query; frequencies kept: |ky| <= F, 0 <= kx <= F


def fdesc(x):
    """(n, 64, 64) thumbnails -> shift-invariant unit vectors: log |FFT| of the zero-mean image, low frequencies."""
    x = x - x.mean((1, 2), keepdims=True)
    m = np.abs(np.fft.rfft2(x))                                  # (n, 64, 33)
    m = np.concatenate([m[:, :F + 1, :F + 1], m[:, -F:, :F + 1]], 1).reshape(len(x), -1)
    m = np.log1p(m); m[:, 0] = 0                                 # DC removed
    m = m - m.mean(1, keepdims=True)
    return np.ascontiguousarray(m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-6), dtype=np.float32)


def find_shift(a, b, k=4):
    """Shift s (in pixels of a) with roll(a, s) ~ b, from FFT cross-correlation of k x averaged zero-mean images."""
    sa, sb = V.shrink(a, k), V.shrink(b, k)
    sa, sb = sa - sa.mean(), sb - sb.mean()
    xc = np.fft.irfft2(np.conj(np.fft.rfft2(sa)) * np.fft.rfft2(sb), s=sa.shape)
    dy, dx = np.unravel_index(np.argmax(xc), xc.shape)
    H, W = sa.shape
    dy = dy - H if dy > H // 2 else dy; dx = dx - W if dx > W // 2 else dx
    # refine at full resolution within +-k px
    best = (-2, 0, 0)
    for ry in range(dy * k - k, dy * k + k + 1):
        for rx in range(dx * k - k, dx * k + k + 1):
            r = V.pearson(V.shrink(np.roll(a, (ry, rx), (0, 1)), 2), V.shrink(b, 2))
            if r > best[0]:
                best = (r, ry, rx)
    return best[1], best[2]


def overlap(a, b, sy, sx):
    """Views of a (moved by sy, sx without wrapping) and b over the region they share."""
    H, W = b.shape
    y0, y1 = max(0, sy), min(H, H + sy); x0, x1 = max(0, sx), min(W, W + sx)
    return a[y0 - sy:y1 - sy, x0 - sx:x1 - sx], b[y0:y1, x0:x1], (y1 - y0) * (x1 - x0) / (H * W)


def check(job):
    qsplit, q, cands, rsplit = (*job, 'train') if len(job) == 3 else job
    a0 = V.grey(qsplit, q); best = None
    for ref, t in cands:
        b = V.grey(rsplit, ref)
        a = np.ascontiguousarray(V.F[t](a0))
        if a.shape != b.shape:
            a = np.asarray(Image.fromarray(a).resize(b.shape[::-1], Image.BILINEAR), dtype=np.float32)
        sy, sx = find_shift(a, b)
        w = np.roll(a, (sy, sx), (0, 1))
        wf, wl = V.pearson(w, b), V.pearson(V.shrink(w), V.shrink(b))
        oa, ob, frac = overlap(a, b, sy, sx)
        of, ol = (V.pearson(oa, ob), V.pearson(V.shrink(oa), V.shrink(ob))) if min(oa.shape) >= 16 else (-1, -1)
        row = (max(wl, ol), ref, t, sy, sx, wf, wl, of, ol, frac)
        if best is None or row[0] > best[0]:
            best = row
    return q, best


def move(box, sy, sx, t, rw, rh, qw, qh):
    """Reference box -> query box: undo the shift (reference frame), then undo the turn and the resize."""
    x, y, w, h = box  # wrap round (a box pushed off one edge of a wrapped copy comes back on the other)
    return V.move_box(((x - sx) % rw, (y - sy) % rh, w, h), t, rw, rh, qw, qh)


def main(qsplit, rsplit='train'):
    import faiss
    tn, tx, _ = C.load(rsplit); qn, qx, _ = C.load(qsplit)
    _, ts = C.descr(tx); R = fdesc(tx); R[ts < C.MIN_STD] = 0
    index = faiss.IndexFlatIP(R.shape[1]); index.add(R)
    _, qs = C.descr(qx)
    cand = [dict() for _ in qn]; same = qsplit == rsplit  # same split: skip each image's match with itself
    for t, f in C.TRANSFORMS.items():
        s, i = index.search(fdesc(np.ascontiguousarray(f(qx))), K + same)
        for k in range(len(qn)):
            for sc, j in zip(s[k], i[k]):
                if same and j == k: continue
                if sc > cand[k].get(j, (-2,))[0]:
                    cand[k][j] = (sc, t)
        print('searched', t, flush=True)
    jobs = [(qsplit, qn[k], [(tn[j], t) for j, (sc, t) in sorted(cand[k].items(), key=lambda kv: -kv[1][0])[:K]], rsplit)
            for k in range(len(qn)) if qs[k] >= C.MIN_STD]
    with Pool() as p:
        res = dict(p.imap_unordered(check, jobs, chunksize=16))
    qims, qboxes = V.coco(qsplit); tims, tboxes = V.coco(rsplit)
    wh = {n: (im['width'], im['height']) for n, im in tims.items()}
    out = C.OUT / (f'shifts_{qsplit}.csv' if rsplit == 'train' else f'shifts_{qsplit}_vs_{rsplit}.csv')
    with out.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['query', 'ref', 'transform', 'shift_y', 'shift_x', 'wrap_full', 'wrap_low', 'ovl_full', 'ovl_low',
                    'ovl_frac', 'q_boxes', 'box_recall'])
        for q, (score, ref, t, sy, sx, wf, wl, of, ol, frac) in res.items():
            rec = ''
            if qims:
                qw, qh = qims[q]['width'], qims[q]['height']; rw, rh = wh[ref]
                moved = [(b[0], move(b[1:], sy, sx, t, rw, rh, qw, qh)) for b in tboxes.get(ref, [])]
                moved = [(c, m) for c, m in moved if m]
                qb = qboxes.get(q, [])
                if wf < 0.95 and wl < 0.95:  # not a wrapped copy: count only val boxes in the shared region
                    sc_x, sc_y = qw / rw, qh / rh  # approximate the shared region in query pixels (exact for 'id')
                    y0, y1 = max(0, sy) * sc_y, min(rh, rh + sy) * sc_y; x0, x1 = max(0, sx) * sc_x, min(rw, rw + sx) * sc_x
                    qb = [b for b in qb if t != 'id' or (x0 <= b[1] + b[3] / 2 <= x1 and y0 <= b[2] + b[4] / 2 <= y1)]
                hit = sum(any(c == qc and V.iou(m, qbb) >= 0.5 for c, m in moved) for qc, *qbb in qb)
                rec = f'{hit / len(qb):.3f}' if qb else ''
            w.writerow([q, ref, t, sy, sx, f'{wf:.4f}', f'{wl:.4f}', f'{of:.4f}', f'{ol:.4f}', f'{frac:.3f}',
                        len(qboxes.get(q, [])), rec])
    print('->', out)


if __name__ == '__main__':
    main(*sys.argv[1:3])
