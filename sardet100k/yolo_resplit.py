"""Pack the leak-free re-split (results/resplit_split.csv) for a YOLO run on the cluster.

The cluster already holds the official train / val images (sardet100k-yolo.tar). This adds what it lacks, in one tar:
  test_images/<file>             the official test images (labels released with the dataset's test.json)
  labels/{train,val,test}/<stem>.txt   YOLO labels for the NEW split (same 6 classes, same order as yolo_prepare.py)
  split.txt                      '<official split> <file> <new split>' per image; the job moves images by it
File names are unique across the official splits (0000001 ... 0116598), so images can be moved freely.

    ~/.venvs/sar-audit/bin/python sardet100k/yolo_resplit.py     # -> data/sardet100k-resplit.tar
"""
import csv, io, json, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'data' / 'SARDet_100K'; OUT = HERE / 'data' / 'sardet100k-resplit.tar'


def coco(split):
    p = SRC / 'Annotations' / f'{split}.json'; p = p if p.exists() else SRC / f'{split}.json'
    return json.loads(p.read_text())


def add_text(tar, name, text):
    b = text.encode(); ti = tarfile.TarInfo(name); ti.size = len(b); tar.addfile(ti, io.BytesIO(b))


rows = list(csv.DictReader((HERE / 'results' / 'resplit_split.csv').open()))
new = {r['file_name']: r['new_split'] for r in rows}
cls = None; labels = {}
for split in ('train', 'val', 'test'):
    d = coco(split)
    cats = sorted(d['categories'], key=lambda c: c['id'])
    names = [c['name'] for c in cats]; assert cls is None or [c for c in cls] == names, (cls, names); cls = names
    idx = {c['id']: k for k, c in enumerate(cats)}
    ims = {im['id']: im for im in d['images']}; lines = {im['id']: [] for im in d['images']}
    for a in d['annotations']:
        im = ims[a['image_id']]; W, H = im['width'], im['height']
        x, y, w, h = a['bbox']; x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        lines[im['id']].append(f"{idx[a['category_id']]} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                               f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    for i, im in ims.items():
        labels[im['file_name']] = '\n'.join(lines[i]) + ('\n' if lines[i] else '')
print('classes', cls)
with tarfile.open(OUT, 'w') as tar:
    add_text(tar, 'split.txt', ''.join(f"{r['official_split']} {r['file_name']} {r['new_split']}\n" for r in rows))
    add_text(tar, 'names.txt', '\n'.join(cls) + '\n')
    for f, text in labels.items():
        add_text(tar, f'labels/{new[f]}/{Path(f).stem}.txt', text)
    for p in sorted((SRC / 'JPEGImages' / 'test').iterdir()):
        tar.add(p, f'test_images/{p.name}')
print(len(labels), 'labels;', OUT, f'{OUT.stat().st_size / 1e9:.2f} GB')
