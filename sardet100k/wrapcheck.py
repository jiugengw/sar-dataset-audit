"""Are the 'wrapped' matches really rolled images, or plain shifts?

The wrap rule rolls one image (np.roll) and correlates all of it. A plain shift by s also passes when the shared part
carries the bright targets. Rolling by s and by s - side give the same picture, so the test looks at both parts of the
rolled image separately: the part that wrapped round and the rest. A real roll matches in both parts; a plain shift
(neighbouring tile) in only one.

Result (val/test pairs, 29,449 confirmed wrap rows): no real rolls. ShipDataset neighbours at 128 px (half a tile)
mostly have file names that differ by one grid index.

    ~/.venvs/sar-audit/bin/python sardet100k/wrapcheck.py   # -> results/wrap_check.json
"""
import csv, json, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import groups as G
import verify as V

D = HERE / 'results'
PAIRS = [('val', 'train'), ('test', 'train'), ('val', 'test'), ('val', 'val'), ('test', 'test')]


def corr(x, y):
    if x.size < 200 or x.std() < 1e-3 or y.std() < 1e-3: return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def check(job):
    qs, rs, r = job
    try:
        a = np.ascontiguousarray(V.F[r['transform']](V.grey(qs, r['query']))); b = V.grey(rs, r['ref'])
        if a.shape != b.shape: return r['source'], 'size differs'
        sy, sx = int(r['shift_y']), int(r['shift_x']); h, w = a.shape
        x = np.roll(a, (sy, sx), (0, 1)); m = np.zeros(a.shape, bool)
        if sy > 0: m[:sy] = True
        elif sy < 0: m[h + sy:] = True
        if sx > 0: m[:, :sx] = True
        elif sx < 0: m[:, w + sx:] = True
        r1, r2 = corr(x[m], b[m]), corr(x[~m], b[~m])
        if np.isnan(r1) or np.isnan(r2): return r['source'], 'one part flat'
        # OGSOD copies are re-processed (matching part r ~0.6-0.9), so a part 'matches' at r >= 0.5;
        # the part that does not match sits near r = 0
        if min(r1, r2) >= 0.5: return r['source'], 'real roll (both parts match)'
        if max(r1, r2) >= 0.5 and min(r1, r2) < 0.3: return r['source'], 'plain shift (one part matches)'
        return r['source'], 'unclear'
    except Exception as e:  # noqa: BLE001
        return r.get('source', ''), f'error {e}'


def main():
    src = {r['file_name']: r['source'] for r in csv.DictReader((D / 'resplit_split.csv').open())}
    jobs = []
    for q, rr in PAIRS:
        for r in csv.DictReader((D / f'gp_{q}_vs_{rr}.csv').open()):
            if r['kind'] == 'shift' and r['query'] != r['ref'] and G.shifted(r) == G.WRAP and G.confirmed_shift(r):
                r['source'] = src.get(r['query'], ''); jobs.append((q, rr, r))
    with Pool() as p:
        res = list(p.imap_unordered(check, jobs, chunksize=32))
    out = {'confirmed wrap rows': len(jobs), 'verdict': dict(Counter(v for _, v in res)),
           'by source': {f'{s} | {v}': n for (s, v), n in sorted(Counter(res).items())}}
    (D / 'wrap_check.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
