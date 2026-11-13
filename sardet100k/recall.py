"""How many real copies does the full search still miss? Plant edited copies of known train images and search for them.

120 train images per source (fixed seed) are each edited KINDS ways and saved as a fake split 'planted'
(data/SARDet_100K/JPEGImages/planted/, git-ignored). The same search as for the re-split then runs against all of
train: allpairs.py (every whole-image and shifted match, background check) and ship.py (target pixels). A planted
image counts as found when a confirmed link reaches its original, or any image in the original's group (a copy of
the original is just as good for keeping the two apart).

Kinds:
  flip + jpeg90        a random flip / turn (not identity), re-saved as JPEG 90
  speckle + flip       new multiplicative speckle (gamma, 4 looks) and a random flip / turn
  gain + jpeg60        brightness x 0.7, JPEG 60
  wrap                 rolled by a random (dy, dx), each 16 px .. side - 16
  wrap + speckle       the same plus 4-look speckle
  overlap 75 / 50 / 30 the image shifted by 25 / 50 / 70% of its side (random direction), the uncovered part filled
                       from another train image of the same source and size: a neighbouring tile sharing that share
Chips cut from larger images are measured by contained.py's own planted test on the GPU.

    ~/.venvs/sar-audit/bin/python sardet100k/recall.py make     # images + results/recall_truth.csv
    ~/.venvs/sar-audit/bin/python sardet100k/candidates.py thumbs  (adds the planted split's thumbnails)
    ~/.venvs/sar-audit/bin/python sardet100k/allpairs.py planted train
    ~/.venvs/sar-audit/bin/python sardet100k/ship.py all all_planted_vs_train.csv
    ~/.venvs/sar-audit/bin/python sardet100k/recall.py score    # -> results/recall.json
"""
import csv, io, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import groups as G
import verify as V

D = HERE / 'results'; OUT = V.ROOT / 'JPEGImages' / 'planted'
PER_SOURCE = 120
KINDS = ['flip + jpeg90', 'speckle + flip', 'gain + jpeg60', 'wrap', 'wrap + speckle', 'overlap 75', 'overlap 50',
         'overlap 30']
FLIPS = [t for t in V.F if t != 'id']


def jpeg(a, q):
    b = io.BytesIO(); Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(b, 'JPEG', quality=q)
    return np.asarray(Image.open(b), dtype=np.float32)


def edit(a, kind, rng, other):
    if kind == 'flip + jpeg90':
        t = FLIPS[rng.integers(len(FLIPS))]; return jpeg(np.ascontiguousarray(V.F[t](a)), 90), t
    if kind == 'speckle + flip':
        t = FLIPS[rng.integers(len(FLIPS))]; return np.ascontiguousarray(V.F[t](a * rng.gamma(4, 1 / 4, a.shape))), t
    if kind == 'gain + jpeg60':
        return jpeg(a * 0.7, 60), 'id'
    h, w = a.shape
    if kind.startswith('wrap'):
        dy, dx = (int(rng.integers(16, s - 16)) for s in (h, w))
        b = np.roll(a, (dy, dx), (0, 1))
        return (b * rng.gamma(4, 1 / 4, a.shape) if 'speckle' in kind else b), f'roll {dy} {dx}'
    share = int(kind.split()[1]) / 100  # overlapping tile: shift along one axis, fill from another image
    axis = int(rng.integers(2)); sign = 1 if rng.integers(2) else -1
    s = int(round((1 - share) * a.shape[axis])) * sign
    b = np.roll(a, s, axis); o = np.roll(other, s, axis)
    idx = [slice(None), slice(None)]
    idx[axis] = slice(0, s) if s > 0 else slice(a.shape[axis] + s, None)
    b[tuple(idx)] = o[tuple(idx)]
    return b, f'shift axis {axis} by {s}'


def make():
    rng = np.random.default_rng(0)
    rows = list(csv.DictReader((D / 'resplit_split.csv').open()))
    train = [r for r in rows if r['official_split'] == 'train']
    by = defaultdict(list)
    for r in train: by[r['source']].append(r['file_name'])
    OUT.mkdir(parents=True, exist_ok=True)
    truth = []
    for src in sorted(by):
        names = sorted(by[src]); pick = rng.permutation(len(names))[:PER_SOURCE]
        for i in pick:
            n = names[i]; a = V.grey('train', n)
            for k, kind in enumerate(KINDS):
                other = None
                if kind.startswith('overlap'):
                    for _ in range(50):  # a different image of the same source and size
                        o = names[rng.integers(len(names))]
                        if o != n:
                            ob = V.grey('train', o)
                            if ob.shape == a.shape: other = ob; break
                    if other is None: continue
                b, how = edit(a, kind, rng, other)
                q = f'p{len(truth):05d}.png'
                Image.fromarray(b.clip(0, 255).astype(np.uint8)).save(OUT / q)
                truth.append({'query': q, 'original': n, 'source': src, 'kind': kind, 'edit': how})
    with (D / 'recall_truth.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, list(truth[0])); w.writeheader(); w.writerows(truth)
    print(len(truth), 'planted images ->', OUT)


def score():
    truth = list(csv.DictReader((D / 'recall_truth.csv').open()))
    group = {r['file_name']: r['group'] for r in csv.DictReader((D / 'resplit_split.csv').open())
             if r['official_split'] == 'train'}
    ship = {(r['kind'], r['query'], r['ref']): r['fg_r'] for r in csv.DictReader((D / 'all_planted_vs_train_ship.csv').open())}
    hit_direct, hit_group, hit_any = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in csv.DictReader((D / 'all_planted_vs_train.csv').open()):
        c = G.confirmed_first(r) if r['kind'] == 'whole' else G.confirmed_shift(r)
        how = 'background' if c else None
        f = ship.get((r['kind'], r['query'], r['ref']), '')
        if not how and f not in ('', 'nan') and not f.startswith('error') and float(f) >= G.SHIP: how = 'ship pixels'
        if how:
            hit_direct[r['query']].add((r['ref'], how)); hit_any[r['query']].add(how)
            hit_group[r['query']].add(group.get(r['ref']))
    out = {'by_kind': {}, 'by_source': {}, 'by_kind_and_source': {}}
    agg = defaultdict(Counter)
    for t in truth:
        q = t['query']
        direct = any(ref == t['original'] for ref, _ in hit_direct[q])
        found = direct or group.get(t['original']) in hit_group[q]
        for key in (('by_kind', t['kind']), ('by_source', t['source']), ('by_kind_and_source', f"{t['kind']} | {t['source']}"),
                    ('all', 'all')):
            agg[key]['n'] += 1; agg[key]['found'] += found; agg[key]['direct'] += direct
            agg[key]['any confirmed link'] += bool(hit_any[q])
    for (sec, k), c in sorted(agg.items()):
        v = {'planted': c['n'], 'found': round(c['found'] / c['n'], 3), 'found (the original itself)': round(c['direct'] / c['n'], 3)}
        if sec == 'all': out['all'] = v
        else: out[sec][k] = v
    (D / 'recall.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('all', 'by_kind', 'by_source')}, indent=1))


if __name__ == '__main__':
    make() if sys.argv[1] == 'make' else score()
