"""COCO mAP on val or test, split by leak group and source (groups.py).

    ~/.venvs/sar-yolo/bin/python sardet100k/score_groups.py copy [val|test]
        no-learning baseline: every image gets the boxes of its best train match, moved the matched way
        (flip / turn / shift / resize), scored by how well the images match. No model involved.
    ~/.venvs/sar-yolo/bin/python sardet100k/score_groups.py model results/val_predictions.json [val|test]
        a trained detector's predictions (cluster/job_a.py output).
Writes results/scores_<name>.json (val) or scores_<name>_test.json: mAP (0.5:0.95) and AP50 for all images, clean,
leaked, each group, and each source split into clean / leaked.
"""
import contextlib, io, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import groups as G
import shifts as S
import verify as V
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def copy_predictions(gt, grp):
    """The nearest-neighbour 'detector': train boxes of the best match, moved onto the val image."""
    _, tboxes = V.coco('train'); tims, _ = V.coco('train')
    by_name = {im['file_name']: im for im in gt.dataset['images']}; cat = {c['name']: c['id'] for c in gt.dataset['categories']}
    res = []
    for q, g in grp.items():
        im = by_name[q]; qw, qh = im['width'], im['height']
        f, s = g['first'], g['shift']
        opts = []
        if f['ref'] and f['low_r']:
            opts.append((float(f['low_r']), f['ref'], f['transform'], 0, 0))
        if s:
            opts.append((max(float(s['wrap_low']), float(s['ovl_low'])), s['ref'], s['transform'], int(s['shift_y']), int(s['shift_x'])))
        if not opts:
            continue
        score, ref, t, sy, sx = max(opts)
        rw, rh = tims[ref]['width'], tims[ref]['height']
        for c, *box in tboxes.get(ref, []):
            m = S.move(box, sy, sx, t, rw, rh, qw, qh)
            if m:
                res.append({'image_id': im['id'], 'category_id': cat[c], 'bbox': [float(v) for v in m], 'score': max(score, 0.0) + 1e-3})
    return res


def model_predictions(gt, path):
    ids = {im['file_name']: im['id'] for im in gt.dataset['images']}; cat = {c['name']: c['id'] for c in gt.dataset['categories']}
    return [{'image_id': ids[p['file_name']], 'category_id': cat[p['category']], 'bbox': p['bbox'], 'score': p['score']}
            for p in json.loads(Path(path).read_text())]


def evaluate(gt, dt, img_ids):
    if not img_ids: return None
    with contextlib.redirect_stdout(io.StringIO()):
        e = COCOeval(gt, dt, 'bbox'); e.params.imgIds = sorted(img_ids); e.evaluate(); e.accumulate(); e.summarize()
    return {'images': len(img_ids), 'mAP': round(100 * e.stats[0], 2), 'AP50': round(100 * e.stats[1], 2)}


def main(name, pred=None, split='val'):
    p = V.ROOT / 'Annotations' / f'{split}.json'
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(p if p.exists() else V.ROOT / f'{split}.json'))
    grp = G.load(split); ids = {im['file_name']: im['id'] for im in gt.dataset['images']}
    res = copy_predictions(gt, grp) if name == 'copy' else model_predictions(gt, pred)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res)
    sets = defaultdict(set)
    for q, g in grp.items():
        i = ids[q]; sets['all'].add(i); leak = g['group'] != 'clean'
        sets['leaked' if leak else 'clean'].add(i); sets['group: ' + g['group']].add(i)
        sets[f"source: {g['source']} | {'leaked' if leak else 'clean'}"].add(i)
        sets['evidence: ' + g['evidence']].add(i)
        sets[f"source x evidence: {g['source']} | {g['evidence']}"].add(i)
    out = {k: evaluate(gt, dt, v) for k, v in sorted(sets.items())}
    (G.D / f"scores_{name}{'' if split == 'val' else '_' + split}.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:70s} {v['images']:6d} img  mAP {v['mAP']:6.2f}  AP50 {v['AP50']:6.2f}")


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'copy':
        main('copy', None, *a[1:2])
    else:
        main(*a[:3])
