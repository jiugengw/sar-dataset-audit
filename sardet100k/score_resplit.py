"""Score a YOLO run on the leak-free re-split, next to the same recipe on the official split.

    ~/.venvs/sar-yolo/bin/python sardet100k/score_resplit.py RESPLIT_VAL_PRED [RESPLIT_TEST_PRED]
Ground truth for the new val / test is gathered from the official train / val / test annotation files
(resplit_split.csv says which image went where). The official-split model's predictions (results/val_predictions.json and
test_predictions.json, job_a / job_b) are scored on the official val and test the same way. mAP (0.5:0.95) and AP50 overall and per source (true source,
mapping.json) -> results/scores_resplit.json.
"""
import contextlib, csv, io, json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import resplit as R
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


_ANN = {}


def _ann(split):
    if split not in _ANN:
        p = R.ROOT / 'Annotations' / f'{split}.json'; p = p if p.exists() else R.ROOT / f'{split}.json'
        _ANN[split] = json.loads(p.read_text())
    return _ANN[split]


def gt_for(names):
    """One COCO object holding the images in `names` (any official split), ids renumbered."""
    ims, anns, cats = [], [], None
    for split in ('train', 'val', 'test'):
        d = _ann(split); cats = cats or d['categories']
        keep = {im['id']: im for im in d['images'] if im['file_name'] in names}
        new_id = {}
        for i, im in keep.items():
            new_id[i] = len(ims) + 1; ims.append({**im, 'id': new_id[i]})
        anns += [{**a, 'id': len(anns) + k + 1, 'image_id': new_id[a['image_id']]}
                 for k, a in enumerate(x for x in d['annotations'] if x['image_id'] in keep)]
    gt = COCO()
    gt.dataset = {'images': ims, 'annotations': anns, 'categories': cats}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def score(gt, pred_path, src):
    ids = {im['file_name']: im['id'] for im in gt.dataset['images']}
    cat = {c['name']: c['id'] for c in gt.dataset['categories']}
    res = [{'image_id': ids[p['file_name']], 'category_id': cat[p['category']], 'bbox': p['bbox'], 'score': p['score']}
           for p in json.loads(Path(pred_path).read_text()) if p['file_name'] in ids]
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res)
    sets = defaultdict(set)
    for f, i in ids.items():
        sets['all'].add(i); sets['source: ' + src[f]].add(i)
    out = {}
    for k, v in sorted(sets.items()):
        with contextlib.redirect_stdout(io.StringIO()):
            e = COCOeval(gt, dt, 'bbox'); e.params.imgIds = sorted(v); e.evaluate(); e.accumulate(); e.summarize()
        out[k] = {'images': len(v), 'mAP': round(100 * e.stats[0], 2), 'AP50': round(100 * e.stats[1], 2)}
    return out


def main(val_pred, test_pred=None):
    rows = list(csv.DictReader((R.D / 'resplit_split.csv').open()))
    src = {r['file_name']: r['source'] for r in rows}
    out = {}
    for split in ('val', 'test'):
        names = {r['file_name'] for r in rows if r['official_split'] == split}
        out[f'official split, official {split}'] = score(gt_for(names), R.D / f'{split}_predictions.json', src)
    for split, pred in (('val', val_pred), ('test', test_pred)):
        if pred:
            names = {r['file_name'] for r in rows if r['new_split'] == split}
            out[f're-split, new {split}'] = score(gt_for(names), pred, src)
    (R.D / 'scores_resplit.json').write_text(json.dumps(out, indent=1))
    keys = sorted({k for v in out.values() for k in v}, key=lambda k: (k != 'all', k))
    print(f"{'':32s}" + ''.join(f'{n[:34]:>36s}' for n in out))
    for k in keys:
        print(f'{k:32s}' + ''.join(f"{v[k]['images']:8d} img mAP {v[k]['mAP']:6.2f} AP50 {v[k]['AP50']:6.2f}" if k in v
                                   else f"{'':36s}" for v in out.values()))
    print('->', R.D / 'scores_resplit.json')


if __name__ == '__main__':
    main(*sys.argv[1:3])
