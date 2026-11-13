"""SARDet-100K leakage audit, stage 1: candidate pairs across splits (and within a split).

Every image is read as grey, shrunk with area averaging (which also averages out most speckle) to a 64 px thumbnail
(cached), and turned into a 32 px zero-mean unit-norm descriptor, so an inner product is a Pearson correlation.
Each query image is tried under all 8 flips / 90-degree turns against the reference split; the best match is kept.

    ~/.venvs/sar-audit/bin/python sardet100k/candidates.py thumbs            # cache thumbnails (~20 s)
    ~/.venvs/sar-audit/bin/python sardet100k/candidates.py search val train  # val (query) vs train
    ~/.venvs/sar-audit/bin/python sardet100k/candidates.py search test train
    ~/.venvs/sar-audit/bin/python sardet100k/candidates.py search val val    # duplicates inside val

Paths are relative to this folder: reads data/SARDet_100K/, writes data/cache/thumbs_<split>.npz (git-ignored) and
results/nn_<query>_vs_<ref>.csv
(query file, best reference file, r, transform, query/ref sizes, query/ref std).
"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'data' / 'SARDet_100K'; CACHE = HERE / 'data' / 'cache'; OUT = HERE / 'results'
SPLITS = ('train', 'val', 'test'); T, D, K = 64, 32, 5
MIN_STD = 4.0  # nearly blank images match everything; leave them out (as in near_copies.py)


def files(split):
    ann = ROOT / 'Annotations' / f'{split}.json'
    if ann.exists():
        return sorted(im['file_name'] for im in json.loads(ann.read_text())['images'])
    return sorted(p.name for p in (ROOT / 'JPEGImages' / split).iterdir() if p.suffix.lower() in ('.jpg', '.png', '.tif', '.bmp'))


def thumb(path):
    try:
        im = Image.open(path)
        w, h = im.size
        x = np.asarray(im.convert('L').resize((T, T), Image.BOX), dtype=np.uint8)
        return x, w, h
    except Exception as e:  # report and keep going; a broken file is itself worth noting
        print('unreadable', path, e, file=sys.stderr)
        return np.zeros((T, T), np.uint8), 0, 0


def cache_thumbs(split):
    out = CACHE / f'thumbs_{split}.npz'
    if out.exists():
        return
    names = files(split)
    with Pool() as p:
        res = p.map(thumb, [ROOT / 'JPEGImages' / split / n for n in names], chunksize=64)
    np.savez_compressed(out, names=np.array(names), x=np.stack([r[0] for r in res]),
                        wh=np.array([(r[1], r[2]) for r in res], np.int32))
    print(split, len(names), 'thumbnails ->', out)


def load(split):
    z = np.load(CACHE / f'thumbs_{split}.npz')
    return z['names'], z['x'].astype(np.float32), z['wh']


def descr(x):
    """64 px thumbnails -> 32 px zero-mean unit-norm vectors (float32) and their pixel std."""
    x = x.reshape(len(x), D, T // D, D, T // D).mean((2, 4)).reshape(len(x), -1)
    s = x.std(1); x = x - x.mean(1, keepdims=True)
    return np.ascontiguousarray(x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), dtype=np.float32), s


TRANSFORMS = {  # name -> function on (n, h, w) arrays
    'id': lambda a: a, 'rot90': lambda a: np.rot90(a, 1, (1, 2)), 'rot180': lambda a: np.rot90(a, 2, (1, 2)),
    'rot270': lambda a: np.rot90(a, 3, (1, 2)), 'lr': lambda a: a[:, :, ::-1], 'ud': lambda a: a[:, ::-1, :],
    'transpose': lambda a: a.transpose(0, 2, 1), 'antitranspose': lambda a: np.rot90(a, 2, (1, 2)).transpose(0, 2, 1)}


def search(qsplit, rsplit):
    import faiss
    qn, qx, qwh = load(qsplit); rn, rx, rwh = load(rsplit)
    R, rs = descr(rx); R[rs < MIN_STD] = 0  # blank references can never win
    index = faiss.IndexFlatIP(R.shape[1]); index.add(R)
    best_r = np.full(len(qn), -2.0, np.float32); best_i = np.full(len(qn), -1); best_t = np.full(len(qn), '', object)
    same = qsplit == rsplit
    for t, f in TRANSFORMS.items():
        Q, qs = descr(np.ascontiguousarray(f(qx)))
        sims, idx = index.search(Q, K)
        if same:  # drop self-matches (only possible for the identity transform, but a symmetric image can match itself)
            sims[idx == np.arange(len(qn))[:, None]] = -2
        j = sims.argmax(1); s = sims[np.arange(len(qn)), j]; i = idx[np.arange(len(qn)), j]
        s[qs < MIN_STD] = -2
        better = s > best_r
        best_r[better], best_i[better], best_t[better] = s[better], i[better], t
        print(f'{qsplit} vs {rsplit} {t:14s} r>=0.9: {(s >= 0.9).sum():6d}  r>=0.97: {(s >= 0.97).sum():6d}')
    _, qs = descr(qx)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'nn_{qsplit}_vs_{rsplit}.csv'
    with path.open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['query', 'ref', 'r', 'transform', 'qw', 'qh', 'rw', 'rh', 'qstd', 'rstd'])
        for k in np.argsort(-best_r):
            i = best_i[k]
            w.writerow([qn[k], rn[i] if i >= 0 else '', f'{best_r[k]:.4f}', best_t[k], *qwh[k], *(rwh[i] if i >= 0 else (0, 0)),
                        f'{qs[k]:.1f}', f'{rs[i]:.1f}' if i >= 0 else ''])
    for th in (0.99, 0.97, 0.95, 0.9, 0.8, 0.7):
        print(f'  best r >= {th}: {(best_r >= th).sum():6d} of {len(qn)} ({100 * (best_r >= th).mean():.1f}%)')
    print('->', path)


if __name__ == '__main__':
    if sys.argv[1] == 'thumbs':
        CACHE.mkdir(parents=True, exist_ok=True)
        for s in SPLITS:
            cache_thumbs(s)
    elif sys.argv[1] == 'search':
        search(sys.argv[2], sys.argv[3])
