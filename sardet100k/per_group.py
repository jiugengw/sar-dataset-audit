"""Score each SAR-AIRcraft group (one airport, as far as the links can tell) in the new val / test on its own, for
every re-split draw, to see whether the draws differ because some airports are easier than others.

    python per_group.py OUT.json NAME:SPLIT:VAL_PRED:TEST_PRED ...
SPLIT is 'saved' (results/resplit_split.csv) or a redraw.py split.txt. Per group: images, boxes, boxes per image,
median box side (px), mAP and AP50; also the whole split, and which groups the draws share.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import score_resplit as S

SOURCE = 'SAR-AIRcraft'


def main(out, *draws):
    rows = list(csv.DictReader((HERE / 'results' / 'resplit_split.csv').open()))
    src = {r['file_name']: r['source'] for r in rows}
    group = {r['file_name']: r['group'] for r in rows}
    res = {}
    for d in draws:
        name, split, vp, tp = d.split(':')
        new = ({r['file_name']: r['new_split'] for r in rows} if split == 'saved'
               else dict(l.split()[1:] for l in open(split)))
        res[name] = {}
        for part, pred in (('val', vp), ('test', tp)):
            members = defaultdict(set)
            for f, s in new.items():
                if s == part and src[f] == SOURCE: members[group[f]].add(f)
            gt = S.gt_for(set().union(*members.values()))
            box = defaultdict(list); fid = {im['id']: im['file_name'] for im in gt.dataset['images']}
            for a in gt.dataset['annotations']: box[fid[a['image_id']]].append(a['bbox'])
            def stats(names):
                b = [x for f in names for x in box[f]]
                return {'images': len(names), 'boxes': len(b), 'boxes per image': round(len(b) / len(names), 1),
                        'median box side': round(float(np.median([np.sqrt(w * h) for _, _, w, h in b])), 1) if b else None}
            rows_out = {'all': {**stats(set().union(*members.values())), **S.score(gt, pred, src)['all']}}
            for g, names in sorted(members.items(), key=lambda kv: -len(kv[1])):
                sc = S.score(S.gt_for(names), pred, src)['all']
                rows_out[g] = {**stats(names), 'mAP': sc['mAP'], 'AP50': sc['AP50']}
            res[name][part] = rows_out
            print(f'== {name} {part}')
            for g, v in rows_out.items():
                print(f"  {g:28s} {v['images']:5d} img {v['boxes']:6d} boxes {v['boxes per image']:5.1f}/img "
                      f"side {v['median box side']}  mAP {v['mAP']:6.2f}  AP50 {v['AP50']:6.2f}", flush=True)
    names = list(res)
    shared = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for pa in ('val', 'test'):
                for pb in ('val', 'test'):
                    both = sorted((set(res[a][pa]) & set(res[b][pb])) - {'all'})
                    if both: shared[f'{a} {pa} & {b} {pb}'] = both
    res['groups in both draws'] = shared
    print(json.dumps(shared, indent=1))
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
