"""SARDet-100K leakage audit, stage 2: check stage-1 candidates at full resolution and by label transfer.

For every query image whose best train match (candidates.py) has thumbnail r >= 0.7:
  full_r : Pearson of the full-resolution query (turned the matched way, resized to the reference size) and the reference.
           A re-saved copy gives ~0.95+; a new pass over the same place (new speckle) gives much less.
  low_r  : the same on 8x area-averaged images (speckle mostly gone): high low_r with modest full_r = same scene, new pass.
  box_*  : (val only, which has labels) training boxes turned back onto the query; share of query boxes matched by a
           transferred box of the same class at IoU >= 0.5 ('label transfer'), and share of transferred boxes matched.
Source group comes from size / format / classes (the annotation files carry no source field), see source().

    ~/.venvs/sar-audit/bin/python sardet100k/verify.py val    # -> results/verify_val.csv
    ~/.venvs/sar-audit/bin/python sardet100k/verify.py test   # (no labels: no box_* columns)
    ~/.venvs/sar-audit/bin/python sardet100k/verify.py val val    # duplicates inside val
                                                                               # -> results/verify_val_vs_val.csv
"""
import csv, json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'data' / 'SARDet_100K'; OUT = HERE / 'results'
MIN_THUMB_R = 0.7
F = {'id': lambda a: a, 'rot90': lambda a: np.rot90(a, 1), 'rot180': lambda a: np.rot90(a, 2), 'rot270': lambda a: np.rot90(a, 3),
     'lr': lambda a: a[:, ::-1], 'ud': lambda a: a[::-1, :], 'transpose': lambda a: a.T, 'antitranspose': lambda a: np.rot90(a, 2).T}
INV = {'id': 'id', 'rot90': 'rot270', 'rot270': 'rot90', 'rot180': 'rot180', 'lr': 'lr', 'ud': 'ud',
       'transpose': 'transpose', 'antitranspose': 'antitranspose'}


def coco(split):
    p = ROOT / 'Annotations' / f'{split}.json'
    if not p.exists():
        return {}, {}
    d = json.loads(p.read_text()); cat = {c['id']: c['name'] for c in d['categories']}
    ims = {im['id']: im for im in d['images']}; boxes = defaultdict(list)
    for a in d['annotations']:
        boxes[ims[a['image_id']]['file_name']].append((cat[a['category_id']], *a['bbox']))
    return {im['file_name']: im for im in d['images']}, boxes


def source(w, h, ext, classes):
    """Source dataset from size / format / labelled classes (counts match the per-source table in the SARDet-100K repo)."""
    if ext == 'bmp': return 'SADD'
    if ext == 'png': return 'OGSOD'
    if (w, h) == (800, 800): return 'HRSID'
    if (w, h) == (1000, 1000): return 'AIR_SARShip-2'
    if w != h or w not in (256, 512): return 'SSDD'
    if w == 512:
        if 'aircraft' in classes: return 'SAR-AIRcraft'
        if 'car' in classes: return 'SIVED'
        return '512 other (AIR_SARShip-1/MSAR/SIVED/SAR-AIRcraft)' if not classes else '512 other (AIR_SARShip-1/MSAR)'
    if classes and classes - {'ship'}: return 'MSAR'
    return '256 ship (ShipDataset/MSAR)'


def grey(split, name):
    return np.asarray(Image.open(ROOT / 'JPEGImages' / split / name).convert('L'), dtype=np.float32)


def pearson(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def shrink(a, k=8):
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean((1, 3))


def iou(a, b):
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0)); iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy; return inter / (aw * ah + bw * bh - inter + 1e-9)


def move_box(box, t, rw, rh, qw, qh):
    """A reference box (x, y, w, h) on the reference image -> the query image (undo transform t, rescale)."""
    m = np.zeros((rh, rw), bool); x, y, w, h = (int(round(v)) for v in box)
    m[max(0, y):max(0, y) + max(1, h), max(0, x):max(0, x) + max(1, w)] = True
    m = F[INV[t]](m)  # now in query orientation, reference scale
    ys, xs = np.nonzero(m)
    if len(xs) == 0: return None
    sx, sy = qw / m.shape[1], qh / m.shape[0]
    return (xs.min() * sx, ys.min() * sy, (xs.max() + 1 - xs.min()) * sx, (ys.max() + 1 - ys.min()) * sy)


def check(row):
    qsplit, q, rsplit, ref, t, thumb_r = row
    a = grey(qsplit, q); b = grey(rsplit, ref); qh, qw = a.shape; rh, rw = b.shape
    at = np.ascontiguousarray(F[t](a))
    if at.shape != b.shape:
        at = np.asarray(Image.fromarray(at).resize((rw, rh), Image.BILINEAR), dtype=np.float32)
    return q, pearson(at, b), pearson(shrink(at), shrink(b)), (qw, qh, rw, rh)


def main(qsplit, rsplit='train'):
    rows = list(csv.DictReader((OUT / f'nn_{qsplit}_vs_{rsplit}.csv').open()))
    qims, qboxes = coco(qsplit); _, tboxes = coco(rsplit)
    todo = [(qsplit, r['query'], rsplit, r['ref'], r['transform'], float(r['r'])) for r in rows if float(r['r']) >= MIN_THUMB_R]
    with Pool() as p:
        res = {q: rest for q, *rest in p.imap_unordered(check, todo, chunksize=32)}
    out = OUT / (f'verify_{qsplit}.csv' if rsplit == 'train' else f'verify_{qsplit}_vs_{rsplit}.csv')
    with out.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['query', 'ref', 'transform', 'thumb_r', 'full_r', 'low_r', 'source', 'ref_source',
                    'q_boxes', 'ref_boxes', 'box_recall', 'box_precision'])
        for r in rows:
            q, ref, t = r['query'], r['ref'], r['transform']
            qw, qh, rw, rh = (int(r[k]) for k in ('qw', 'qh', 'rw', 'rh'))
            ext = q.rsplit('.', 1)[1].lower(); qc = {b[0] for b in qboxes.get(q, [])}; rc = {b[0] for b in tboxes.get(ref, [])}
            src = source(qw, qh, ext, qc if qims else set()); rsrc = source(rw, rh, ref.rsplit('.', 1)[1].lower(), rc) if ref else ''
            full_r, low_r = (f'{res[q][0]:.4f}', f'{res[q][1]:.4f}') if q in res else ('', '')
            rec = prec = ''
            if qims and q in res:
                moved = [(b[0], move_box(b[1:], t, rw, rh, qw, qh)) for b in tboxes.get(ref, [])]
                moved = [(c, m) for c, m in moved if m]
                qb = qboxes.get(q, [])
                hit_q = sum(any(c == qc_ and iou(m, qbb) >= 0.5 for c, m in moved) for qc_, *qbb in qb)
                hit_r = sum(any(c == qc_ and iou(m, qbb) >= 0.5 for qc_, *qbb in qb) for c, m in moved)
                rec = f'{hit_q / len(qb):.3f}' if qb else ''; prec = f'{hit_r / len(moved):.3f}' if moved else ''
            w.writerow([q, ref, t, r['r'], full_r, low_r, src, rsrc, len(qboxes.get(q, [])), len(tboxes.get(ref, [])), rec, prec])
    print('->', out)


if __name__ == '__main__':
    main(*sys.argv[1:3])
