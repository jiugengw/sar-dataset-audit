"""Score one cross-validation fold (folds.py): the held-out fold's images, overall, per source, and per group for
groups of 100+ images (airports and large scenes). With 'pooled', score every image with the prediction of the
model that did not train on it (all folds' predictions together).

    python score_fold.py folds.csv FOLD PRED.json OUT.json
    python score_fold.py folds.csv pooled PRED_0.json,PRED_1.json,... OUT.json
"""
import csv, json, sys, tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import score_resplit as S


def main(folds_csv, which, preds, out):
    rows = list(csv.DictReader(open(folds_csv)))
    src = {r['file_name']: r['source'] for r in rows}
    if which == 'pooled':
        fold_of = {r['file_name']: int(r['fold']) for r in rows}
        allp = []
        for i, p in enumerate(preds.split(',')):
            allp += [x for x in json.load(open(p)) if fold_of.get(x['file_name']) == i]
        tmp = Path(tempfile.mkdtemp()) / 'pooled.json'; tmp.write_text(json.dumps(allp)); pred = tmp
        part = rows
    else:
        pred = preds; part = [r for r in rows if r['fold'] == which]
    names = {r['file_name'] for r in part}
    res = {'fold': which, 'all + sources': S.score(S.gt_for(names), pred, src), 'groups (100+ images)': {}}
    members = defaultdict(set)
    for r in part: members[r['group']].add(r['file_name'])
    for g, m in sorted(members.items(), key=lambda kv: -len(kv[1])):
        if len(m) < 100: break
        v = S.score(S.gt_for(m), pred, src)
        res['groups (100+ images)'][g] = {'source': src[next(iter(m))], **v['all']}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res['all + sources'], indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
