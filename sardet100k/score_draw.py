"""Score a YOLO run trained on a redrawn re-split (redraw.py) on its new val and test, like score_resplit.py.

    python score_draw.py split.txt VAL_PRED TEST_PRED OUT.json     # needs results/resplit_split.csv for the sources
"""
import csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import score_resplit as S


def main(split_txt, val_pred, test_pred, out):
    src = {r['file_name']: r['source'] for r in csv.DictReader((HERE / 'results' / 'resplit_split.csv').open())}
    new = dict(l.split()[1:] for l in open(split_txt))
    res = {}
    for split, pred in (('val', val_pred), ('test', test_pred)):
        names = {f for f, s in new.items() if s == split}
        res[f'redrawn re-split, new {split}'] = S.score(S.gt_for(names), pred, src)
    Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
