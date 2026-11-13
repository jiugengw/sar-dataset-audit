"""Controls for the similarity search (the analogue of adding noise to weights: does the finding survive, or is it luck?).

Negative control (chance matches): every train image is cut into a 4 x 4 grid and its 16 pieces are shuffled
(a fixed random order per image), which keeps its texture and brightness but breaks any real layout. Val is
searched against this fake pool exactly as against the real one (8 flips / turns, best of 94k), then checked at
full resolution. Whatever reaches the leak bands here is chance, so the counts give a false-positive rate per band.
(A first version rolled each train image by half its size instead; 860 val images then matched pixel for pixel,
because SARDet-100K holds wrapped-round copies of train chips; see shifts.py. That result is kept under
'negative_roll_half' as a finding, not as a control.)

Positive control (noise injection): 1,000 random train images are corrupted and searched for against the real
train pool. Corruptions: new multiplicative speckle (gamma noise, 4 looks and 1 look), gain x0.6, JPEG quality 40,
shifts of 4 and 16 px (edge padded), and speckle + flip. Reports how often the original is found as the best
match, and which band (A / B / C) the pair lands in.

    ~/.venvs/sar-audit/bin/python sardet100k/controls.py   # -> results/controls.json
"""
import io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import candidates as C
import verify as V

rng = np.random.default_rng(0)


def band(full_r, low_r):
    return 'A' if full_r >= 0.95 else 'B' if low_r >= 0.95 else 'C (low_r 0.8-0.95)' if low_r >= 0.8 else 'below'


def index_of(desc):
    import faiss
    idx = faiss.IndexFlatIP(desc.shape[1]); idx.add(desc); return idx


def best_over_transforms(idx, x):
    best_r = np.full(len(x), -2.0, np.float32); best_i = np.zeros(len(x), int); best_t = np.full(len(x), 'id', object)
    for t, f in C.TRANSFORMS.items():
        Q, qs = C.descr(np.ascontiguousarray(f(x))); s, i = idx.search(Q, 1); s = s[:, 0]; s[qs < C.MIN_STD] = -2
        m = s > best_r; best_r[m], best_i[m], best_t[m] = s[m], i[m, 0], t
    return best_r, best_i, best_t


def full_check(q_arr, ref_arr, t):
    at = np.ascontiguousarray(V.F[t](q_arr))
    if at.shape != ref_arr.shape:
        at = np.asarray(Image.fromarray(at).resize(ref_arr.shape[::-1], Image.BILINEAR), dtype=np.float32)
    return V.pearson(at, ref_arr), V.pearson(V.shrink(at), V.shrink(ref_arr))


def roll_half(a):
    return np.roll(a, (a.shape[-2] // 2, a.shape[-1] // 2), axis=(-2, -1))


def shuffle_blocks(a, seed, g=4):
    """Cut a (h, w) image into a g x g grid (edges cropped to a multiple of g) and shuffle the pieces."""
    h, w = (a.shape[0] // g) * g, (a.shape[1] // g) * g
    p = a[:h, :w].reshape(g, h // g, g, w // g).transpose(0, 2, 1, 3).reshape(g * g, h // g, w // g)
    p = p[np.random.default_rng(seed).permutation(g * g)]
    return p.reshape(g, g, h // g, w // g).transpose(0, 2, 1, 3).reshape(h, w)


def negative_control():
    tn, tx, _ = C.load('train'); vn, vx, _ = C.load('val')
    R, rs = C.descr(np.stack([shuffle_blocks(x, j) for j, x in enumerate(tx)])); R[rs < C.MIN_STD] = 0
    r, i, t = best_over_transforms(index_of(R), vx)
    real = np.array([float(x) for x in np.genfromtxt(C.OUT / 'nn_val_vs_train.csv', delimiter=',', skip_header=1, usecols=2)])
    out = {'thumb_r_at_least': {str(th): {'real': int((real >= th).sum()), 'null': int((r >= th).sum())}
                                for th in (0.99, 0.97, 0.95, 0.9, 0.8, 0.7)}}
    bands = {}
    for k in np.nonzero(r >= 0.7)[0]:
        q = V.grey('val', vn[k]); ref = shuffle_blocks(V.grey('train', tn[i[k]]), int(i[k]))
        q = q[:ref.shape[0], :ref.shape[1]] if q.shape == V.grey('train', tn[i[k]]).shape else q
        bd = band(*full_check(q, ref, t[k])); bands[bd] = bands.get(bd, 0) + 1
    out['null_pairs_checked'] = int((r >= 0.7).sum()); out['null_bands'] = bands
    return out


def corrupt(a, kind):
    if kind == 'speckle 4 looks': return a * rng.gamma(4, 1 / 4, a.shape)
    if kind == 'speckle 1 look': return a * rng.gamma(1, 1, a.shape)
    if kind == 'gain x0.6': return a * 0.6
    if kind == 'jpeg q40':
        b = io.BytesIO(); Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(b, 'JPEG', quality=40)
        return np.asarray(Image.open(b), dtype=np.float32)
    if kind.startswith('shift'):
        s = int(kind.split()[1]); return np.pad(a, ((s, 0), (s, 0)), mode='edge')[:a.shape[0], :a.shape[1]]
    if kind == 'speckle 4 looks + flip': return (a * rng.gamma(4, 1 / 4, a.shape))[:, ::-1]
    raise ValueError(kind)


def positive_control(n=1000):
    tn, tx, _ = C.load('train'); R, rs = C.descr(tx); R[rs < C.MIN_STD] = 0; idx = index_of(R)
    pick = rng.choice(np.nonzero(rs >= C.MIN_STD)[0], n, replace=False)
    kinds = ('speckle 4 looks', 'speckle 1 look', 'gain x0.6', 'jpeg q40', 'shift 4', 'shift 16', 'speckle 4 looks + flip')
    out = {}
    for kind in kinds:
        thumbs, origs, noisy = [], [], []
        for k in pick:
            a = V.grey('train', tn[k]); b = corrupt(a, kind); origs.append(a); noisy.append(b)
            im = Image.fromarray(b.clip(0, 255).astype(np.uint8)).resize((C.T, C.T), Image.BOX)
            thumbs.append(np.asarray(im, dtype=np.float32))
        r, i, t = best_over_transforms(idx, np.stack(thumbs))
        found = i == pick
        bands = {}
        for j in np.nonzero(found)[0]:
            bd = band(*full_check(noisy[j], origs[j], t[j])); bands[bd] = bands.get(bd, 0) + 1
        out[kind] = {'found_as_best': int(found.sum()), 'of': n, 'median_thumb_r': float(np.median(r[found])) if found.any() else None,
                     'bands_of_found': bands}
        print(kind, out[kind], flush=True)
    return out


if __name__ == '__main__':
    path = C.OUT / 'controls.json'; res = json.loads(path.read_text()) if path.exists() else {}
    if 'negative' in res and 'negative_roll_half' not in res:
        res['negative_roll_half'] = res.pop('negative')  # the first (rolled) version, kept as a finding
    res['negative'] = negative_control(); print('negative', res['negative'], flush=True)
    if 'positive' not in res:
        res['positive'] = positive_control()
    path.write_text(json.dumps(res, indent=1)); print('->', path)
