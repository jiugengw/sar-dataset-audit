"""M4-SAR: images whose label files are identical (same rotated boxes, 3 decimals), read straight from M4-SAR.zip.
Checks the two-versions pattern (image k and k+56087: 10 m optical + VH and 60 m optical + VV of the same tile) and
which val / test images share their box list with a train image. Then checks those links on the pixels: the optical
and SAR correlation (grey, 128 px) of each val / test image with its train twin, against a control pair (the same
image with a random train image that has the same number of boxes).

    python labeltwins.py M4-SAR.zip OUT_DIR
Output: OUT_DIR/label_groups.csv (group, n_boxes, ids with splits), OUT_DIR/labeltwins.json, OUT_DIR/twin_pixels.csv.
"""
import csv, hashlib, io, json, random, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

HALF = 56087


def main(zip_path, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(zip_path); names = z.namelist()
    img = {}                                             # (modality, id) -> zip member
    for x in names:
        p = x.split('/')
        if len(p) == 5 and p[2] == 'images' and '.' in p[4]: img[(p[1], int(p[4].split('.')[0]))] = x
    split, nbox, by = {}, {}, defaultdict(list)
    for x in names:
        if not (x.startswith('M4-SAR/sar/labels/') and x.endswith('.txt')): continue
        s, f = x.split('/')[-2:]; i = int(f[:-4]); split[i] = s
        rows = sorted(' '.join(f'{float(v):.3f}' if j else v for j, v in enumerate(l.split()))
                      for l in z.read(x).decode().split('\n') if l.strip())
        nbox[i] = len(rows); by[hashlib.md5('\n'.join(rows).encode()).hexdigest()].append(i)
    # the optical labels should be the same files
    diff_opt = sum(1 for x in names if x.startswith('M4-SAR/optical/labels/') and x.endswith('.txt')
                   and z.read(x) != z.read(x.replace('/optical/', '/sar/')))
    inv = {i: h for h, v in by.items() for i in v}
    gid = {h: k for k, h in enumerate(sorted(by, key=lambda h: min(by[h])))}
    with (out / 'label_groups.csv').open('w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n'); w.writerow(['group', 'n_boxes', 'members'])
        for h, v in sorted(by.items(), key=lambda kv: gid[kv[0]]):
            w.writerow([gid[h], nbox[v[0]], ';'.join(f'{split[i]}/{i}' for i in sorted(v))])
    res = {'images': len(split), 'per split': dict(Counter(split.values())),
           'optical label files differing from SAR label files': diff_opt,
           'k and k+56087 identical labels': sum(1 for i in range(1, HALF + 1) if inv.get(i) and inv.get(i) == inv.get(i + HALF)),
           'k and k+56087 in the same split': sum(1 for i in range(1, HALF + 1) if split.get(i) == split.get(i + HALF)),
           'group sizes': dict(sorted(Counter(len(v) for v in by.values()).items())), 'splits': {}}
    train_by_n = defaultdict(list)
    for i, s in split.items():
        if s == 'train': train_by_n[nbox[i]].append(i)
    rng = random.Random(0); pix = []

    def grey(mod, i):
        with Image.open(io.BytesIO(z.read(img[(mod, i)]))) as im:
            return np.asarray(im.convert('L').resize((128, 128), Image.BILINEAR), np.float32)

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0

    for s in ('val', 'test'):
        ids = sorted(i for i in split if split[i] == s)
        tw = {i: sorted(k for k in by[inv[i]] if split[k] == 'train') for i in ids}
        hit = [i for i in ids if tw[i]]
        res['splits'][s] = {'images': len(ids), 'identical labels with a train image': len(hit),
                            '%': round(100 * len(hit) / len(ids), 1),
                            'of those, boxes per image': dict(sorted(Counter(min(nbox[i], 10) for i in hit).items()))}
        for i in rng.sample(hit, min(1500, len(hit))):
            # the twin of the same version (k <= HALF with k <= HALF) when there is one
            same = [k for k in tw[i] if (k <= HALF) == (i <= HALF)] or tw[i]; t = same[0]
            c = rng.choice(train_by_n[nbox[i]])
            row = {'split': s, 'id': i, 'twin': t, 'control': c, 'n_boxes': nbox[i]}
            for mod in ('optical', 'sar'):
                a = grey(mod, i); row[f'{mod} twin'] = round(corr(a, grey(mod, t)), 3); row[f'{mod} control'] = round(corr(a, grey(mod, c)), 3)
            pix.append(row)
    with (out / 'twin_pixels.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(pix[0]), lineterminator='\n'); w.writeheader(); w.writerows(pix)
    q = lambda v: {p: round(float(np.percentile(v, p)), 3) for p in (5, 25, 50, 75, 95)}
    res['pixel check (sample)'] = {f'{mod} {kind} corr percentiles': q([r[f'{mod} {kind}'] for r in pix])
                                   for mod in ('optical', 'sar') for kind in ('twin', 'control')}
    res['pixel check (sample)']['pairs'] = len(pix)
    for th in (0.5, 0.8, 0.95):
        res['pixel check (sample)'][f'optical corr > {th}: twin / control'] = (
            f"{sum(r['optical twin'] > th for r in pix)} / {sum(r['optical control'] > th for r in pix)}")
    (out / 'labeltwins.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
