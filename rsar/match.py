"""Which SARDet-100K image is each RSAR image? RSAR (95,842 images, rotated boxes) is built on SARDet-100K, with its
own train / val / test split. Every image of both datasets is decoded and hashed by its pixels (array bytes, shape
and mode), so a re-saved file with the same pixels still matches; RSAR images whose pixels match no SARDet image
are listed for a looser check.

    python match.py RSAR_ROOT SARDET_ROOT OUT_DIR
RSAR_ROOT: {train,val,test}/images/*. SARDET_ROOT: images/{train,val,test}/* (official split, as job_a unpacks it).
Output: OUT_DIR/rsar_to_sardet.csv (rsar_split, rsar_name, h, w, sardet = 'split/name;...' or ''),
OUT_DIR/match.json (counts).
"""
import csv, hashlib, json, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def key(path):
    try:
        with Image.open(path) as im:
            a = np.asarray(im)
        h = hashlib.md5(a.tobytes()); h.update(str((a.shape, a.dtype.str)).encode())
        return str(path), h.hexdigest(), a.shape[0], a.shape[1]
    except Exception as e:  # unreadable file: keep a note, never stop the run
        return str(path), 'error: ' + type(e).__name__, 0, 0


def scan(files):
    with Pool(4) as p:
        return p.map(key, files, chunksize=256)


def main(rsar_root, sardet_root, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    sd = sorted(p for s in ('train', 'val', 'test') for p in (Path(sardet_root) / 'images' / s).iterdir()
                if p.suffix.lower() in EXT)
    rs = sorted(p for s in ('train', 'val', 'test') for p in (Path(rsar_root) / s / 'images').iterdir()
                if p.suffix.lower() in EXT)
    print(len(sd), 'SARDet images,', len(rs), 'RSAR images', flush=True)
    by_hash = defaultdict(list)
    for f, h, _, _ in scan(sd):
        p = Path(f); by_hash[h].append(f'{p.parent.name}/{p.name}')
    print('SARDet hashed', flush=True)
    rows, cnt = [], Counter()
    same_name = Counter()
    for f, h, H, W in scan(rs):
        p = Path(f); split = p.parent.parent.name
        hits = [] if h.startswith('error') else by_hash.get(h, [])
        rows.append((split, p.name, H, W, ';'.join(hits)))
        cnt[(split, 'error' if h.startswith('error') else 'matched' if hits else 'unmatched')] += 1
        if hits:
            cnt[(split, 'matched, SARDet ' + '+'.join(sorted({x.split('/')[0] for x in hits})))] += 1
            same_name[(split, any(x.split('/')[1].rsplit('.', 1)[0] == p.stem for x in hits))] += 1
    with (out / 'rsar_to_sardet.csv').open('w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n'); w.writerow(['rsar_split', 'rsar_name', 'h', 'w', 'sardet']); w.writerows(rows)
    # an RSAR image whose pixels appear in two RSAR splits is a cross-split duplicate in RSAR itself
    rs_split_of = defaultdict(set)
    for split, name, _, _, hits in rows:
        if hits: rs_split_of[hits].add(split)
    res = {'sardet images': len(sd), 'rsar images': len(rs),
           'counts': {f'{a}: {b}': n for (a, b), n in sorted(cnt.items())},
           'rsar name equals a matched SARDet name': {f'{a}: {b}': n for (a, b), n in sorted(same_name.items())},
           'pixel groups in more than one RSAR split': sum(len(v) > 1 for v in rs_split_of.values())}
    (out / 'match.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
