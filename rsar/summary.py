"""Collect the RSAR rotated-box runs (cluster/job_ad): the official split (val, test) and the 5 cross-validation folds
(mean, sd, min, max), overall and per source.

    python summary.py RUNS_DIR OUT.json     # newest runs/rsar-obb-{official,fold0..4}-<jobid>/scores.json
"""
import json, statistics as st, sys
from pathlib import Path


def newest(runs, tag):
    c = sorted((p for p in Path(runs).glob(f'rsar-obb-{tag}-*') if (p / 'scores.json').exists()),
               key=lambda p: p.stat().st_mtime)
    return c[-1] if c else None


def main(runs, out):
    res = {'runs': {}, 'official': {}, 'cv folds': {}, 'cv': {}}
    off = newest(runs, 'official')
    if off:
        res['runs']['official'] = off.name; s = json.loads((off / 'scores.json').read_text())
        res['official'] = {k: {m: v[m] for m in ('mAP50-95', 'mAP50')} for k, v in s.items()}
    folds = {}
    for f in range(5):
        d = newest(runs, f'fold{f}')
        if d: res['runs'][f'fold{f}'] = d.name; folds[f] = json.loads((d / 'scores.json').read_text())
    for f, s in folds.items():
        res['cv folds'][f] = {k.split('__', 1)[1]: {m: v[m] for m in ('mAP50-95', 'mAP50')} for k, v in s.items()}
    keys = sorted({k for s in res['cv folds'].values() for k in s})
    for k in keys:
        for m in ('mAP50-95', 'mAP50'):
            v = [s[k][m] for s in res['cv folds'].values() if k in s]
            if v: res['cv'].setdefault(k, {})[m] = {'mean': round(st.mean(v), 2), 'sd': round(st.stdev(v), 2) if len(v) > 1 else None,
                                                     'min': min(v), 'max': max(v), 'folds': len(v)}
    Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
