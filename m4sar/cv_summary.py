"""Collect the M4-SAR rotated-box runs (cluster/job_ag): the official split (val, test) and the 5 leak-free
cross-validation folds (mean, sd, min, max), for all images and per version (vh, vv), overall and per class.

    python cv_summary.py RUNS_DIR OUT.json     # newest runs/m4-obb-{official,0..4}-<jobid>/scores.json
"""
import json, statistics as st, sys
from pathlib import Path

METRICS = ('mAP50-95', 'mAP50')


def flat(v):
    """One eval set -> {'mAP50-95': x, 'mAP50': y, 'mAP50-95 <class>': ..., 'mAP50 <class>': ...}."""
    out = {m: v[m] for m in METRICS}
    for m in METRICS:
        for c, x in v[f'per class {m}'].items(): out[f'{m} {c}'] = x
    return out


def newest(runs, tag):
    c = sorted((p for p in Path(runs).glob(f'm4-obb-{tag}-*') if (p / 'scores.json').exists()),
               key=lambda p: p.stat().st_mtime)
    return c[-1] if c else None


def main(runs, out):
    res = {'runs': {}, 'official': {}, 'cv folds': {}, 'cv': {}, 'drop (official test - cv mean)': {}}
    off = newest(runs, 'official')
    if off:
        res['runs']['official'] = off.name
        res['official'] = {k: flat(v) for k, v in json.loads((off / 'scores.json').read_text()).items()}
    for f in range(5):
        d = newest(runs, str(f))
        if d:
            res['runs'][f'fold{f}'] = d.name
            res['cv folds'][f] = {k.split('__', 1)[1]: flat(v) for k, v in json.loads((d / 'scores.json').read_text()).items()}
    for part in sorted({k for s in res['cv folds'].values() for k in s}):
        for m in sorted({m for s in res['cv folds'].values() for m in s.get(part, {})}):
            v = [s[part][m] for s in res['cv folds'].values() if m in s.get(part, {})]
            res['cv'].setdefault(part, {})[m] = {'mean': round(st.mean(v), 2), 'sd': round(st.stdev(v), 2) if len(v) > 1 else None,
                                                 'min': min(v), 'max': max(v), 'folds': len(v)}
            o = res['official'].get(f'test__{part}', {}).get(m)
            if o is not None:
                res['drop (official test - cv mean)'].setdefault(part, {})[m] = {
                    'points': round(o - st.mean(v), 2), '% of official': round(100 * (o - st.mean(v)) / o, 1) if o else None}
    Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
