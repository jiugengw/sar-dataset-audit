"""Lay out RSAR for a rotated-box YOLO run (ultralytics OBB format), either on its official split or on one fold of the
SARDet-100K 5-fold group cross-validation (sardet100k/folds.py). RSAR's images are SARDet-100K images under the same
name and split (match.py), so every RSAR image takes the fold of its SARDet image: the folds are leak-free for RSAR
too, and a fold's per-source share stays ~20%.

    python prepare.py RSAR_ROOT CV_FOLDS_CSV FOLD OUT_DIR      # FOLD = official | 0..4
RSAR_ROOT: {train,val,test}/{images,annfiles} (DOTA lines: x1 y1 ... x4 y4 class difficult). Images are MOVED.
Output: OUT_DIR/{images,labels}/{train,test[,val]}/ (labels: class x1 y1 ... x4 y4, normalised and clipped),
OUT_DIR/data.yaml (training), OUT_DIR/eval/<part>__<source>.yaml (scoring each part overall and per source).
"""
import csv, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

NAMES = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']


def main(root, folds_csv, fold, out):
    root, out = Path(root), Path(out)
    info = {(r['official_split'], r['file_name']): r for r in csv.DictReader(open(folds_csv))}
    rows = []                                # (rsar split, name, source, part)
    for split in ('train', 'val', 'test'):
        for p in sorted((root / split / 'images').iterdir()):
            r = info[(split, p.name)]        # KeyError = not a SARDet image of that split: stop
            part = split if fold == 'official' else ('test' if r['fold'] == fold else 'train')
            rows.append((split, p.name, r['source'], part))
    assert len({n for _, n, _, _ in rows}) == len(rows), 'image names repeat across splits'
    parts = sorted({x[3] for x in rows})
    for d in ('images', 'labels'):
        for part in parts: (out / d / part).mkdir(parents=True, exist_ok=True)
    stats = Counter(); per = defaultdict(Counter); lists = defaultdict(list)
    for split, name, source, part in rows:
        src = root / split / 'images' / name
        with Image.open(src) as im: w, h = im.size
        lines = []
        ann = root / split / 'annfiles' / (Path(name).stem + '.txt')
        for ln in (ann.read_text().split('\n') if ann.exists() else []):
            t = ln.split()
            if len(t) < 9: continue
            xy = [float(v) for v in t[:8]]; c = t[8]
            if c not in NAMES: raise ValueError(f'unknown class {c} in {ann}')
            if len(t) > 9 and t[9] != '0': stats['difficult flag'] += 1
            nx = [min(max(v / (w if i % 2 == 0 else h), 0.0), 1.0) for i, v in enumerate(xy)]
            stats['coords clipped'] += sum(a != b / (w if i % 2 == 0 else h) for i, (a, b) in enumerate(zip(nx, xy)))
            lines.append(f'{NAMES.index(c)} ' + ' '.join(f'{v:.6f}' for v in nx)); stats['boxes ' + part] += 1
        if not ann.exists(): stats['no annotation file'] += 1
        dst = out / 'images' / part / name
        os.rename(src, dst)
        (out / 'labels' / part / (Path(name).stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
        per[part][source] += 1; lists[(part, source)].append(str(dst))
    names = 'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES))
    (out / 'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/test\n' + names)
    (out / 'eval').mkdir(exist_ok=True); (out / 'lists').mkdir(exist_ok=True)
    for part in parts:
        if part == 'train': continue
        (out / 'eval' / f'{part}__all.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/{part}\n' + names)
        for (pp, source), files in lists.items():
            if pp != part: continue
            lst = out / 'lists' / f'{part}__{source}.txt'; lst.write_text('\n'.join(files) + '\n')
            (out / 'eval' / f'{part}__{source}.yaml').write_text(f'path: {out}\ntrain: images/train\nval: {lst}\n' + names)
    res = {'fold': fold, 'images per part': {p: sum(c.values()) for p, c in per.items()},
           'per part and source': {p: dict(sorted(c.items())) for p, c in per.items()}, 'notes': dict(stats)}
    (out / 'prepare.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
