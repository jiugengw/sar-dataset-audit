"""Build a YOLO view of SARDet-100K (official train / val split) for the leaked-vs-clean scoring.

Images are hard-linked, not copied (same disk, no extra space): data/yolo/images/{train,val}/<file> are links to
data/SARDet_100K/JPEGImages/{train,val}/<file> (a folder symlink does not work: Ultralytics resolves it and then
looks for labels beside the originals);
labels are written to data/yolo/labels/{train,val}/<stem>.txt; data/yolo/data.yaml lists the 6 classes.

    ~/.venvs/sar-yolo/bin/python sardet100k/yolo_prepare.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'data' / 'SARDet_100K'; Y = HERE / 'data' / 'yolo'

for split in ('train', 'val'):
    d = json.loads((SRC / 'Annotations' / f'{split}.json').read_text())
    cats = sorted(d['categories'], key=lambda c: c['id']); cls = {c['id']: k for k, c in enumerate(cats)}
    ims = {im['id']: im for im in d['images']}; lines = {im['id']: [] for im in d['images']}
    for a in d['annotations']:
        im = ims[a['image_id']]; W, H = im['width'], im['height']
        x, y, w, h = a['bbox']; x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        lines[im['id']].append(f"{cls[a['category_id']]} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    (Y / 'images').mkdir(parents=True, exist_ok=True); lab = Y / 'labels' / split; lab.mkdir(parents=True, exist_ok=True)
    img = Y / 'images' / split; img.mkdir(exist_ok=True)
    for im in ims.values():
        dst = img / im['file_name']
        if not dst.exists():
            dst.hardlink_to(SRC / 'JPEGImages' / split / im['file_name'])
    for i, im in ims.items():
        (lab / (Path(im['file_name']).stem + '.txt')).write_text('\n'.join(lines[i]) + ('\n' if lines[i] else ''))
    print(split, len(ims), 'images', sum(map(len, lines.values())), 'boxes')
names = '\n'.join(f'  {k}: {c["name"]}' for k, c in enumerate(cats))
(Y / 'data.yaml').write_text(f'path: {Y}\ntrain: images/train\nval: images/val\nnames:\n{names}\n')
print((Y / 'data.yaml').read_text())
