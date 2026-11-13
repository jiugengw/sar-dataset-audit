"""Negative control for the shift rules (shifts.py + groups.shifted), with the look-elsewhere effect included.

shift_controls.py tested random pairs, but the real search keeps the best of 94k train images, chosen for a similar
|FFT|. For sparse images (one bright ship on dark sea) the best of 94k can line up its ship with the query's and
pass the speckle-reduced rules without being the same scene. This control repeats the whole shift search against a
fake train pool that cannot contain the query's scene: every train image (and its boxes) cut into 4 x 4 blocks and
shuffled with a fixed per-image order. Ships and texture survive inside the blocks; the layout of the scene does not.
Any val image passing a rule here is a false positive of that rule.

    ~/.venvs/sar-audit/bin/python sardet100k/shift_null.py val   # -> results/shifts_val_null.csv, shift_null.json
"""
import csv, json, sys, zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import candidates as C
import controls as K
import groups as G
import shifts as S
import verify as V

GRID = 4
_grey = V.grey


def seed(name):
    return zlib.crc32(name.encode())


def shuffled_grey(split, name):
    a = _grey(split, name)
    return K.shuffle_blocks(a, seed(name), GRID).astype(np.float32) if split == 'train' else a


V.grey = shuffled_grey  # module-level, so spawned workers get it too


def shuffle_boxes(boxes, name, w, h):
    """Move each box with the block holding its centre; boxes cut by a block edge are clipped to that block."""
    bh, bw = h // GRID, w // GRID
    perm = np.random.default_rng(seed(name)).permutation(GRID * GRID)
    where = {int(src): dst for dst, src in enumerate(perm)}  # piece src ends up at position dst
    out = []
    for c, x, y, bwid, bht in boxes:
        cx, cy = x + bwid / 2, y + bht / 2
        gx, gy = min(int(cx // bw), GRID - 1), min(int(cy // bh), GRID - 1)
        dst = where[gy * GRID + gx]; dy, dx = (dst // GRID - gy) * bh, (dst % GRID - gx) * bw
        x0, y0 = max(x, gx * bw), max(y, gy * bh); x1, y1 = min(x + bwid, (gx + 1) * bw), min(y + bht, (gy + 1) * bh)
        if x1 > x0 and y1 > y0:
            out.append((c, x0 + dx, y0 + dy, x1 - x0, y1 - y0))
    return out


def main(qsplit='val'):
    import faiss
    tn, tx, _ = C.load('train'); qn, qx, _ = C.load(qsplit)
    tx = np.stack([K.shuffle_blocks(a, seed(n), GRID) for a, n in zip(tx, tn)]).astype(np.float32)
    _, ts = C.descr(tx); R = S.fdesc(tx); R[ts < C.MIN_STD] = 0
    index = faiss.IndexFlatIP(R.shape[1]); index.add(R)
    _, qs = C.descr(qx)
    cand = [dict() for _ in qn]
    for t, f in C.TRANSFORMS.items():
        s, i = index.search(S.fdesc(np.ascontiguousarray(f(qx))), S.K)
        for k in range(len(qn)):
            for sc, j in zip(s[k], i[k]):
                if sc > cand[k].get(j, (-2,))[0]:
                    cand[k][j] = (sc, t)
        print('searched', t, flush=True)
    jobs = [(qsplit, qn[k], [(tn[j], t) for j, (sc, t) in sorted(cand[k].items(), key=lambda kv: -kv[1][0])[:S.K]])
            for k in range(len(qn)) if qs[k] >= C.MIN_STD]
    with Pool() as p:
        res = dict(p.imap_unordered(S.check, jobs, chunksize=16))
    qims, qboxes = V.coco(qsplit); tims, tboxes = V.coco('train')
    out = HERE / 'results' / f'shifts_{qsplit}_null.csv'
    rows = []
    with out.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['query', 'ref', 'transform', 'shift_y', 'shift_x', 'wrap_full', 'wrap_low', 'ovl_full', 'ovl_low',
                    'ovl_frac', 'q_boxes', 'box_recall'])
        for q, (score, ref, t, sy, sx, wf, wl, of, ol, frac) in res.items():
            rec = ''
            if qims:
                qw, qh = qims[q]['width'], qims[q]['height']; rw, rh = tims[ref]['width'], tims[ref]['height']
                tb = shuffle_boxes(tboxes.get(ref, []), ref, rw, rh)
                moved = [(b[0], S.move(b[1:], sy, sx, t, rw, rh, qw, qh)) for b in tb]
                moved = [(c, m) for c, m in moved if m]
                qb = qboxes.get(q, [])
                hit = sum(any(c == qc and V.iou(m, qbb) >= 0.5 for c, m in moved) for qc, *qbb in qb)
                rec = f'{hit / len(qb):.3f}' if qb else ''
            r = [q, ref, t, sy, sx, f'{wf:.4f}', f'{wl:.4f}', f'{of:.4f}', f'{ol:.4f}', f'{frac:.3f}',
                 len(qboxes.get(q, [])), rec]
            w.writerow(r); rows.append(dict(zip(['query', 'ref', 'transform', 'shift_y', 'shift_x', 'wrap_full',
                                                 'wrap_low', 'ovl_full', 'ovl_low', 'ovl_frac', 'q_boxes', 'box_recall'],
                                                map(str, r))))
    # compare with the real search, rule by rule and per source
    real = list(csv.DictReader((HERE / 'results' / f'shifts_{qsplit}.csv').open()))
    src = {r['query']: r['source'] for r in csv.DictReader((HERE / 'results' / f'verify_{qsplit}.csv').open())}
    summary = {}
    for name, rr in (('real', real), ('null', rows)):
        c = Counter(); per = Counter()
        for r in rr:
            g = G.shifted(r)
            if g: c[g] += 1; per[(src.get(r['query']), g)] += 1
        summary[name] = {'by_rule': dict(c), 'per_source': {f'{s} | {g}': n for (s, g), n in sorted(per.items())}}
    summary['queries'] = len(rows)
    (HERE / 'results' / 'shift_null.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k]['by_rule'] for k in ('real', 'null')}, indent=1))
    print('->', out)


if __name__ == '__main__':
    main(*sys.argv[1:2])
