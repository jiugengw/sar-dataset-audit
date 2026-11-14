"""The official-split YOLO11s (job_a; val and test predictions from job_b) on leaked vs clean images.

An official val / test image is 'leaked' when a proven link reaches official train: a confirmed GPU-search link
(gpupairs.py, background rule), a name link with matching pixels (namelinks.py), or the same ShipDataset tile in
another polarization with every box matching (polcheck.py). Everything else is 'clean (none found)'. mAP 0.5:0.95
and AP50, overall and per true source -> results/scores_leak.json.

    ~/.venvs/sar-yolo/bin/python sardet100k/score_leak.py
"""
import csv, gzip, json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import groups as G
import score_resplit as S
import verify as V

D = HERE / 'results'


def leaked(split):
    conf = lambda r: G.confirmed_first(r) if r['kind'] == 'whole' else G.confirmed_shift(r)
    out = {r['query'] for r in csv.DictReader((D / f'gp_{split}_vs_train.csv').open()) if conf(r)}
    boxes = {}
    for s in ('train', 'val', 'test'): boxes.update(V.coco(s)[1])
    import polcheck
    for r in csv.DictReader(gzip.open(D / 'name_links.csv.gz', 'rt')):
        ok = r['verdict'] == 'confirmed' or (r['kind'] == 'other polarization' and
                                             (polcheck.matched(boxes.get(r['a'], []), boxes.get(r['b'], [])) or 0) >= 0.999)
        if not ok: continue
        if r['split_a'] == split and r['split_b'] == 'train': out.add(r['a'])
        if r['split_b'] == split and r['split_a'] == 'train': out.add(r['b'])
    return out


def main():
    rows = list(csv.DictReader((D / 'resplit_split.csv').open()))
    src = {r['file_name']: r['source'] for r in rows}
    out = {}
    for split, pred in (('val', D / 'val_predictions.json'), ('test', D / 'test_predictions.json')):
        names = {r['file_name'] for r in rows if r['official_split'] == split}
        bad = leaked(split) & names
        for label, part in (('all', names), ('leaked', bad), ('clean (none found)', names - bad)):
            out[f'{split}: {label}'] = S.score(S.gt_for(part), pred, src)
    (D / 'scores_leak.json').write_text(json.dumps(out, indent=1))
    keys = sorted({k for v in out.values() for k in v}, key=lambda k: (k != 'all', k))
    for name, v in out.items():
        print(f'== {name}')
        for k in keys:
            if k in v: print(f"   {k:28s} {v[k]['images']:6d} img  mAP {v[k]['mAP']:6.2f}  AP50 {v[k]['AP50']:6.2f}")
    print('->', D / 'scores_leak.json')


if __name__ == '__main__':
    main()
