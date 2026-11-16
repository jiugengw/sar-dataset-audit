"""Score a rotated-box YOLO model on every eval set prepare.py wrote (each held-out part, overall and per source), with
ultralytics' OBB validator (rotated IoU; conf 0.001, up to 300 boxes per image): mAP50-95, mAP50, per class.

    python score_obb.py weights.pt EVAL_DIR OUT.json
"""
import json, os, sys
from pathlib import Path
from ultralytics import YOLO

weights, eval_dir, out = sys.argv[1:4]
model = YOLO(weights); res = {}
for y in sorted(Path(eval_dir).glob('*.yaml')):
    m = model.val(data=str(y), imgsz=512, batch=32, conf=0.001, max_det=300, plots=False, verbose=False, device=os.environ.get('DEVICE', '0'),
                  workers=4, project='/tmp', name=f'v{Path(weights).stat().st_ino}', exist_ok=True)
    b = m.box; names = m.names
    res[y.stem] = {'mAP50-95': round(100 * b.map, 2), 'mAP50': round(100 * b.map50, 2),
                   'per class mAP50-95': {names[int(c)]: round(100 * a, 2) for c, a in zip(b.ap_class_index, b.ap)},
                   'per class mAP50': {names[int(c)]: round(100 * a, 2) for c, a in zip(b.ap_class_index, b.ap50)}}
    print(y.stem, res[y.stem]['mAP50-95'], res[y.stem]['mAP50'], flush=True)
Path(out).write_text(json.dumps(res, indent=1))
