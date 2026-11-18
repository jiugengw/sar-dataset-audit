"""Lay out M4-SAR's SAR images for a rotated-box YOLO run (ultralytics OBB format), on its official split or on one
fold of a leak-free group cross-validation. Reads straight from the zip; the labels are already in that format
(class x1 y1 ... x4 y4, normalised; classes from M4-SAR/default.txt). Both versions of every tile (k: VH, k + 56,087:
VV) are used, as the dataset ships them; they always fall in the same part.

    python prepare.py M4-SAR.zip FOLD OUT_DIR [FOLDS_CSV]      # FOLD = official | 0..4; FOLDS_CSV columns id, fold
Output: OUT_DIR/{images,labels}/{train,val,test}/ (official) or {train,test}/ (fold k held out as test),
OUT_DIR/data.yaml, OUT_DIR/eval/<part>__{all,vh,vv}.yaml, OUT_DIR/prepare.json.
"""
import csv, json, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path

NAMES = ['bridge', 'harbor', 'oil_tank', 'playground', 'airport', 'wind_turbine']
HALF = 56087


def main(zpath, fold, out, folds_csv=None):
    out = Path(out); z = zipfile.ZipFile(zpath)
    fold_of = {int(r['id']): r['fold'] for r in csv.DictReader(open(folds_csv))} if folds_csv else None
    imgs, labels = {}, {}
    for x in z.namelist():
        p = x.split('/')
        if len(p) != 5 or p[1] != 'sar' or '.' not in p[4]: continue
        i = int(p[4].split('.')[0])
        if p[2] == 'images': imgs[i] = (p[3], x)
        elif p[2] == 'labels': labels[i] = x
    stats, per, lists = Counter(), defaultdict(Counter), defaultdict(list)
    for i, (split, member) in sorted(imgs.items()):
        if fold == 'official': part = split
        else: part = 'test' if fold_of[i] == fold else 'train'      # KeyError = image missing from the folds: stop
        version = 'vh' if i <= HALF else 'vv'
        dst = out / 'images' / part / Path(member).name; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(member))
        lines = []
        if i in labels:
            for ln in z.read(labels[i]).decode().split('\n'):
                t = ln.split()
                if not t: continue
                if len(t) != 9 or not (0 <= int(t[0]) < len(NAMES)): raise ValueError(f'bad label line in {labels[i]}: {ln}')
                v = [min(max(float(a), 0.0), 1.0) for a in t[1:]]
                stats['coords clipped'] += sum(a != float(b) for a, b in zip(v, t[1:]))
                lines.append(t[0] + ' ' + ' '.join(f'{a:.6f}' for a in v)); stats['boxes ' + part] += 1
        else: stats['no label file'] += 1
        lab = out / 'labels' / part / (dst.stem + '.txt'); lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text('\n'.join(lines) + ('\n' if lines else ''))
        per[part][version] += 1; lists[(part, version)].append(str(dst))
    names = 'names:\n' + ''.join(f'  {k}: {n}\n' for k, n in enumerate(NAMES))
    (out / 'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/test\n' + names)
    (out / 'eval').mkdir(exist_ok=True); (out / 'lists').mkdir(exist_ok=True)
    for part in sorted(per):
        if part == 'train': continue
        (out / 'eval' / f'{part}__all.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/{part}\n' + names)
        for version in ('vh', 'vv'):
            lst = out / 'lists' / f'{part}__{version}.txt'; lst.write_text('\n'.join(lists[(part, version)]) + '\n')
            (out / 'eval' / f'{part}__{version}.yaml').write_text(f'path: {out}\ntrain: images/train\nval: {lst}\n' + names)
    res = {'fold': fold, 'images per part': {p: sum(c.values()) for p, c in per.items()},
           'per part and version': {p: dict(c) for p, c in per.items()}, 'notes': dict(stats)}
    (out / 'prepare.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
