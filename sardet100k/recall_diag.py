"""Where does the GPU search lose the planted copies it misses?

For every planted image (recall.py) that gpupairs.py did not link to its original (or the original's group), the
check step is run on the true pair, lined up the way it was made. Then:
  lost before the check   the true pair passes the rules: the search never picked the original as a candidate
  pixel check fails       no rule passes on the true pair (exact / re-processed / shifted / overlapping tile)
  background check fails  a rule passes, but the background (targets masked) does not repeat (bg r < 0.5)
For overlaps that fail, also why: shared part under 25%, or full r over the shared part under 0.9.

    ~/.venvs/sar-audit/bin/python sardet100k/recall_diag.py   # -> results/recall_diag.json, recall_diag.csv
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gpupairs as P
import groups as G
import verify as V

D = HERE / 'results'
f4 = lambda v: 'nan' if v != v else f'{v:.4f}'


def rows_for(q, ref, res, k):
    """The three rows gpupairs.full_check writes for one checked pair."""
    z = [v[k].item() for v in res['z']]; w = [v[k].item() for v in res['w']]; o = [v[k].item() for v in res['o']]
    base = {'query': q, 'ref': ref, 'transform': 'id'}
    return [dict(base, kind='whole', full_r=f4(z[0]), low_r=f4(z[1]), bg_r=f4(z[2])),
            dict(base, kind='shift', shift_y=int(w[0]), shift_x=int(w[1]), wrap_full=f4(w[2]), wrap_low=f4(w[3]),
                 ovl_full='-1', ovl_low='-1', ovl_frac='0', bg_wrap=f4(w[4]), bg_ovl=''),
            dict(base, kind='shift', shift_y=int(o[0]), shift_x=int(o[1]), wrap_full='-1', wrap_low='-1',
                 ovl_full=f4(o[3]), ovl_low=f4(o[4]), ovl_frac=f'{o[2]:.3f}', bg_wrap='', bg_ovl=f4(o[5]))]


def verdict(rows, kind):
    hit = lambda r: G.first_pass(r) if r['kind'] == 'whole' else G.shifted(r)
    conf = lambda r: G.confirmed_first(r) if r['kind'] == 'whole' else G.confirmed_shift(r)
    if any(conf(r) for r in rows): return 'lost before the check'
    if any(hit(r) in ('exact copy', 're-processed copy', G.WRAP, 'overlapping tile') for r in rows):
        return 'background check fails'
    if kind.startswith('overlap'):
        o = rows[2]; frac, of, ol = float(o['ovl_frac']), float(o['ovl_full']), float(o['ovl_low'])
        if frac < 0.25: return 'pixel check fails: shared part found under 25%'
        if max(of, ol) >= 0.95 and of < 0.9: return 'pixel check fails: full r under 0.9'
    return 'pixel check fails'


def main():
    truth = list(csv.DictReader((D / 'recall_truth.csv').open()))
    group = {r['file_name']: r['group'] for r in csv.DictReader((D / 'resplit_split.csv').open())
             if r['official_split'] == 'train'}
    linked = defaultdict(set)
    for r in csv.DictReader((D / 'gp_planted_vs_train.csv').open()):
        if G.confirmed_first(r) if r['kind'] == 'whole' else G.confirmed_shift(r):
            linked[r['query']].add(group.get(r['ref'], r['ref']))
    miss = [t for t in truth if group.get(t['original'], t['original']) not in linked[t['query']]]
    print(len(truth), 'planted,', len(miss), 'missed', flush=True)
    out, rows_out = Counter(), []
    for t in miss:
        a = V.grey('planted', t['query']); b = V.grey('train', t['original'])
        tr = t['edit'] if t['kind'] in ('flip + jpeg90', 'speckle + flip') else 'id'
        b = np.ascontiguousarray(V.F[tr](b))  # the planted image is F[tr](original)
        if a.shape != b.shape: v = 'size differs'
        else:
            res = P.check_batch(torch.from_numpy(a.copy())[None].float(), torch.from_numpy(b)[None].float())
            rows = rows_for(t['query'], t['original'], res, 0); v = verdict(rows, t['kind'])
            o = rows[2]; w = rows[1]; z = rows[0]
            rows_out.append({**t, 'verdict': v, 'full_r': z['full_r'], 'low_r': z['low_r'], 'bg_r': z['bg_r'],
                             'wrap_low': w['wrap_low'], 'bg_wrap': w['bg_wrap'], 'ovl_frac': o['ovl_frac'],
                             'ovl_full': o['ovl_full'], 'ovl_low': o['ovl_low'], 'bg_ovl': o['bg_ovl'],
                             'std': f'{a.std():.2f}'})
        out[(t['kind'], v)] += 1; out[('source ' + t['source'], v)] += 1; out[('all', v)] += 1
    with (D / 'recall_diag.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    res = defaultdict(dict)
    for (k, v), n in sorted(out.items()): res[k][v] = n
    res = {'planted': len(truth), 'missed': len(miss), **res}
    (D / 'recall_diag.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
