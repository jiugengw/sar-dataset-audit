"""Every match, not only the best one, for the re-split's links.

candidates.py / verify.py keep one reference per image, and shifts.py checks 5 but keeps the best. An image whose
copies sit in two groups (a val tile overlapping two train tiles, a chip repeated three times) then links to only
one of them, and the other can end up on the other side of a split. This script keeps every candidate pair that
passes a rule, with the background check (background.bg_r) done in the same pass:

  whole image  FAISS top K_WHOLE per flip / turn on the 32 px thumbnails, thumb r >= MIN_THUMB_R; at full resolution
               full_r, low_r (8x averaged) and bg_r, as verify.py + background.py
  shifted      FAISS top K_SHIFT on |FFT| (shifts.fdesc), as shifts.py; every candidate's shift, wrap / overlap
               scores and bg_wrap / bg_ovl
Rows are written only if they pass groups.first_pass (exact / re-processed) or groups.shifted (wrapped / overlapping);
groups.confirmed_first / confirmed_shift then decide which are confirmed. Labels are not used.

    ~/.venvs/sar-audit/bin/python sardet100k/allpairs.py val train     # -> results/all_val_vs_train.csv
    (pairs: val train, test train, train train, val val, test test, val test)
"""
import csv, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import background as B
import candidates as C
import groups as G
import shifts as S
import verify as V

K_WHOLE, K_SHIFT, MIN_THUMB_R = 10, 5, 0.7
COLS = ['kind', 'query', 'ref', 'transform', 'score', 'full_r', 'low_r', 'bg_r', 'shift_y', 'shift_x', 'wrap_full',
        'wrap_low', 'ovl_full', 'ovl_low', 'ovl_frac', 'bg_wrap', 'bg_ovl']


def candidates(qsplit, rsplit):
    """{query index: {(kind, ref index): (score, transform)}}"""
    import faiss
    qn, qx, _ = C.load(qsplit); rn, rx, _ = C.load(rsplit)
    same = qsplit == rsplit
    out = [dict() for _ in qn]
    _, qs = C.descr(qx)
    for kind, desc, k in (('whole', lambda x: C.descr(x)[0], K_WHOLE), ('shift', S.fdesc, K_SHIFT)):
        R = desc(rx); R[C.descr(rx)[1] < C.MIN_STD] = 0
        index = faiss.IndexFlatIP(R.shape[1]); index.add(R)
        best = [dict() for _ in qn]
        for t, f in C.TRANSFORMS.items():
            s, i = index.search(desc(np.ascontiguousarray(f(qx))), k + same)
            for q in range(len(qn)):
                if qs[q] < C.MIN_STD: continue
                for sc, j in zip(s[q], i[q]):
                    if j < 0 or (same and j == q) or (kind == 'whole' and sc < MIN_THUMB_R): continue
                    if sc > best[q].get(j, (-2,))[0]: best[q][j] = (sc, t)
            print(qsplit, 'vs', rsplit, kind, t, flush=True)
        for q in range(len(qn)):
            top = sorted(best[q].items(), key=lambda kv: -kv[1][0])
            for j, v in (top if kind == 'whole' else top[:k]):
                out[q][(kind, j)] = v
    return qn, rn, out


def check(job):
    qsplit, q, rsplit, cands = job
    a0 = V.grey(qsplit, q); rows = []
    for kind, ref, t, sc in cands:
        try:
            b = V.grey(rsplit, ref); a = np.ascontiguousarray(V.F[t](a0))
            if a.shape != b.shape:
                a = np.asarray(Image.fromarray(a).resize(b.shape[::-1], Image.BILINEAR), dtype=np.float32)
            r = dict.fromkeys(COLS, ''); r.update(kind=kind, query=q, ref=ref, transform=t, score=f'{sc:.4f}')
            if kind == 'whole':
                fr, lr = V.pearson(a, b), V.pearson(V.shrink(a), V.shrink(b))
                r.update(full_r=f'{fr:.4f}', low_r=f'{lr:.4f}')
                if G.first_pass(r):
                    r['bg_r'] = f'{B.bg_r(a, b):.4f}'; rows.append(r)
            else:
                sy, sx = S.find_shift(a, b)
                w = np.roll(a, (sy, sx), (0, 1))
                wf, wl = V.pearson(w, b), V.pearson(V.shrink(w), V.shrink(b))
                oa, ob, frac = S.overlap(a, b, sy, sx)
                of, ol = (V.pearson(oa, ob), V.pearson(V.shrink(oa), V.shrink(ob))) if min(oa.shape) >= 16 else (-1, -1)
                r.update(shift_y=sy, shift_x=sx, wrap_full=f'{wf:.4f}', wrap_low=f'{wl:.4f}', ovl_full=f'{of:.4f}',
                         ovl_low=f'{ol:.4f}', ovl_frac=f'{frac:.3f}')
                if G.shifted(r):
                    r['bg_wrap'] = f'{B.bg_r(w, b):.4f}'
                    r['bg_ovl'] = f'{B.bg_r(oa, ob):.4f}' if min(oa.shape) >= 16 else 'nan'
                    rows.append(r)
        except Exception as e:  # noqa: BLE001  (one bad pair must not stop the run)
            print('error', q, ref, e, flush=True)
    return rows


def main(qsplit, rsplit):
    qn, rn, cand = candidates(qsplit, rsplit)
    jobs = [(qsplit, qn[q], rsplit, [(kind, rn[j], t, sc) for (kind, j), (sc, t) in c.items()])
            for q, c in enumerate(cand) if c]
    print(len(jobs), 'queries,', sum(len(j[3]) for j in jobs), 'pairs', flush=True)
    out = HERE / 'results' / f'all_{qsplit}_vs_{rsplit}.csv'
    n = 0
    with Pool() as p, out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, COLS); w.writeheader()
        for k, rows in enumerate(p.imap_unordered(check, jobs, chunksize=8)):
            w.writerows(rows); n += len(rows)
            if k % 5000 == 0: print(k, 'queries done', flush=True)
    print(n, 'rows ->', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
