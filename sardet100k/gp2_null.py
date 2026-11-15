"""Why does the version-2 control (job23, references block-shuffled 4 x 4) still confirm ~350 val images?

For every confirmed row of the null run (thin overlaps off, no bg_low; gp2_eval.group): is the same query-ref pair
also confirmed in the real run (then the shuffle left the copy partly intact: a real link, not a false one), or does
the query have a real link at all? Also the shared-region size and r of the null rows next to the real ones, per
source, and the version-1 control (job19) under the same rules.

    python gp2_null.py RUN_DIR_V2 RUN_DIR_V1 SPLIT_CSV OUT.json
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gp2_eval as E


def confirmed(path, t=None, thin=None, rows=None):
    out = {}
    for r in rows if rows is not None else csv.DictReader(open(path)):
        if r['query'] == r['ref']: continue
        g = E.group(r, t, thin)
        if g: out[(r['query'], r['ref'])] = (g, r)
    return out


def per_rule(run):
    """Every rule mix of gp2_eval: null images whose pair the real run (same rule) does not also confirm."""
    nrows = list(csv.DictReader(open(run / 'gp_val_vs_train_null.csv')))
    rrows = list(csv.DictReader(open(run / 'gp_val_vs_train.csv')))
    out = {}
    for tn, thin in E.THIN.items():
        for t in E.BG_LOW:
            null = confirmed(None, t, thin, nrows); real = confirmed(None, t, thin, rrows)
            bad = {k for k in null if k not in real}
            out[f"{tn} | {'no bg_low' if t is None else f'bg_low >= {t}'}"] = {
                'null images': len({k[0] for k in null}), 'null images with a pair not in the real run': len({k[0] for k in bad}),
                'such pairs by rule': dict(Counter(null[k][0] for k in bad))}
    return out


def q(v):
    v = [x for x in v if x == x]
    return [round(float(x), 3) for x in np.quantile(v, [0.1, 0.5, 0.9])] if v else None


def main(v2, v1, split_csv, out):
    src = {r['file_name']: r['source'] for r in csv.DictReader(open(split_csv))}
    res = {}
    for name, run in (('v2', Path(v2)), ('v1', Path(v1))):
        null = confirmed(run / 'gp_val_vs_train_null.csv')
        real = confirmed(run / 'gp_val_vs_train.csv')
        realq = {k[0] for k in real}
        same_pair = sum(1 for k in null if k in real)
        qs = {k[0] for k in null}
        qs_new = {k[0] for k in null if k not in real}
        f = E.f
        rows = [r for _, r in null.values()]
        rr = [r for g, r in real.values() if g == 'overlapping tile']
        ov = [r for g, r in null.values() if g == 'overlapping tile']
        res[name] = {
            'null: confirmed rows': len(null), 'null: val images': len(qs),
            'null rows whose pair is also confirmed in the real run': same_pair,
            'null: val images with a pair NOT in the real run': len(qs_new),
            '... of which the image has some real link': len(qs_new & realq),
            'null images by source': dict(Counter(src.get(x, '?') for x in qs).most_common()),
            'null images with a pair not in the real run, by source': dict(Counter(src.get(x, '?') for x in qs_new).most_common()),
            'null rows by rule': dict(Counter(g for g, _ in null.values())),
            'null overlap rows: ovl_frac / ovl_full / bg_ovl (10/50/90%)':
                [q([f(r, 'ovl_frac') for r in ov]), q([f(r, 'ovl_full') for r in ov]), q([f(r, 'bg_ovl') for r in ov])],
            'real overlap rows: ovl_frac / ovl_full / bg_ovl (10/50/90%)':
                [q([f(r, 'ovl_frac') for r in rr]), q([f(r, 'ovl_full') for r in rr]), q([f(r, 'bg_ovl') for r in rr])],
            'real: val images with a link': len(realq),
        }
        if name == 'v2': res['v2 per rule'] = per_rule(run); print(json.dumps(res['v2 per rule'], indent=1), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
