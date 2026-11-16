"""The official-split YOLO11s (job_a; val and test predictions from job_b) on leaked vs clean images.

An official val / test image is 'leaked' when it has a direct confirmed link to an official train image: any link
file and rule resplit.py / leakpct.py use (groups.py rules), so the same set as the leak rate in leakpct.json.
Chains do not count. Everything else is 'clean (none found)'. mAP 0.5:0.95 and AP50, overall and per true source.
(The 14 Nov version used the GPU-search and name links only: 9,588 leaked val images instead of 9,659.)

    python score_leak.py [OUT.json]   (default results/scores_leak.json)
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import score_resplit as S

D = HERE / 'results'


def leaked():
    """{split: file names with a direct confirmed link to train}, for val and test."""
    import resplit as R
    out = {'val': set(), 'test': set()}
    for name, a, b, c in R.links():
        if not c: continue
        if a[0] == 'train' and b[0] in out: out[b[0]].add(b[1])
        if b[0] == 'train' and a[0] in out: out[a[0]].add(a[1])
    return out


def main(dest=D / 'scores_leak.json'):
    rows = list(csv.DictReader((D / 'resplit_split.csv').open()))
    src = {r['file_name']: r['source'] for r in rows}
    out = {}; leak = leaked()
    for split, pred in (('val', D / 'val_predictions.json'), ('test', D / 'test_predictions.json')):
        names = {r['file_name'] for r in rows if r['official_split'] == split}
        bad = leak[split] & names
        for label, part in (('all', names), ('leaked', bad), ('clean (none found)', names - bad)):
            out[f'{split}: {label}'] = S.score(S.gt_for(part), pred, src)
    Path(dest).write_text(json.dumps(out, indent=1))
    keys = sorted({k for v in out.values() for k in v}, key=lambda k: (k != 'all', k))
    for name, v in out.items():
        print(f'== {name}')
        for k in keys:
            if k in v: print(f"   {k:28s} {v[k]['images']:6d} img  mAP {v[k]['mAP']:6.2f}  AP50 {v[k]['AP50']:6.2f}")
    print('->', dest)


if __name__ == '__main__':
    main(*sys.argv[1:2])
