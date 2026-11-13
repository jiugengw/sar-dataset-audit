"""job_a helper: predict every image in a folder with a YOLO model and save COCO-style results by file name.

    python job_a.py weights.pt images/val val_predictions.json
Output: [{"file_name", "category", "bbox": [x, y, w, h] in pixels, "score"}, ...] at conf >= 0.001, up to 300 boxes
per image (the usual COCO mAP setting). Scored locally against Annotations/val.json by score_groups.py.
"""
import json, sys
from pathlib import Path
from ultralytics import YOLO

weights, folder, out = sys.argv[1:4]
model = YOLO(weights); names = model.names
files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in ('.jpg', '.png', '.bmp', '.tif'))
res = []
for k in range(0, len(files), 64):
    for f, r in zip(files[k:k + 64], model.predict([str(f) for f in files[k:k + 64]], imgsz=512, conf=0.001,
                                                    max_det=300, verbose=False, device=0)):
        for (x0, y0, x1, y1), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            res.append({'file_name': f.name, 'category': names[int(c)], 'bbox': [round(x0, 2), round(y0, 2),
                        round(x1 - x0, 2), round(y1 - y0, 2)], 'score': round(s, 5)})
    if k % 2048 == 0:
        print(f'{k + 64} / {len(files)} images', flush=True)
Path(out).write_text(json.dumps(res))
print(len(res), 'boxes ->', out)
