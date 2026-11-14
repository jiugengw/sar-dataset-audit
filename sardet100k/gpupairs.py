"""Every-match pair search on the GPU, with the three holes the planted-copy test (recall.py) found fixed.

recall.py found 54% of planted copies with allpairs.py. The misses came from:
  1. low-contrast images were never searched (32 px thumbnail std < 4: 5-8% of the dataset, mostly dark-sea HRSID);
  2. shifted copies: only the top 5 |FFT| candidates were checked, and sea clutter spectra all look alike;
  3. overlapping tiles: the shift came from a whole-image (wrap-round) cross-correlation, which locks onto the wrong
     offset when much of the image differs.
Here:
  1. MIN_STD = 1 (a flat image cannot pass the rules + background check by chance; the null run below measures it);
  2. the top K_FFT = 50 |FFT| candidates per image are reranked by a cheap 64 px alignment (best wrap-round or
     overlap-normalised correlation over all shifts) and the best K_SHIFT = 10 go to the full check (finds wrapped
     copies: 100% of planted ones in the top 5); and patch voting finds partly overlapping tiles, which |FFT| ranks
     poorly: 16 px patches of the 64 px thumbnails (query every 8 px, reference every 2 px, no flips; flat patches
     left out), each query patch's top K_PATCH reference patches with r >= PATCH_R vote for their image, and the
     K_VOTE images with most votes (>= 2) go to the full check;
     candidates are kept per (reference, flip), so a chance whole-image match under one flip cannot hide the true
     shifted match under another;
  3. at full resolution the overlap shift is the argmax of the overlap-normalised cross-correlation (Pearson over
     the shared region only, for every shift with >= 25% shared area), computed with FFTs; the wrap-round shift is the
     argmax of the circular cross-correlation. Both are exact (no coarse-to-fine step).
The whole-image candidates are as before (32 px thumbnails, top K_WHOLE per flip / turn, r >= 0.7).

For every candidate pair (query turned the matched way, resized to the reference's size) the full check gives three
rows in allpairs.py's format: 'whole' (full_r, low_r, bg_r at zero shift), and two 'shift' rows (wrap-round and
overlap: shift, r, 8x averaged r, shared share, background r). Every row also has fg_r, the target-pixel r of ship.py.
A row is written when it passes groups.first_pass or groups.shifted (labels are not used).

Version 2 (after recall_diag.py: 56% of planted misses were never candidates, mostly 30% overlaps; most of the rest
were copies with new speckle, which fail the full-resolution background check by design):
  - patch votes: K_PATCH 64 (was 32); the top K_VOTE_POOL vote candidates (>= 1 vote) are reranked by the overlap-
    normalised alignment of the 64 px thumbnails down to THIN_SHARE shared area, and the best K_VOTE go to the check;
  - a fourth row per pair, 'thin': the best overlap with THIN_SHARE <= shared area < MIN_SHARE (thin overlaps);
  - bg_low_*: the background r on 4x averaged images (targets masked), which survives new speckle; with low_r it can
    confirm a copy with new speckle (threshold set from the shuffled control afterwards);
  - rows are written when any speckle-reduced r >= WRITE_LOW (0.85), not only when a rule passes, so the rules can be
    tuned afterwards on the same files;
  - --pairs FILE: check given pairs (csv: query_split, query, ref_split, ref) instead of searching, e.g. MSAR files
    with neighbouring numbers. Output as above.

    python gpupairs.py IMAGES_DIR QUERY_SPLIT REF_SPLIT OUT.csv [--null]
    python gpupairs.py IMAGES_DIR - - OUT.csv --pairs PAIRS.csv
IMAGES_DIR/<split>/<file>. --null block-shuffles every reference image (4 x 4, fixed per-image order), as
shift_null.py: no query can truly match, so confirmed rows count false positives (some references are still real
copies, e.g. a wrap by whole blocks survives the shuffle; recheck those on the originals).
"""
import csv, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

T, DD = 64, 32
K_WHOLE, K_FFT, K_SHIFT, MIN_THUMB_R, MIN_STD = 10, 50, 10, 0.7, 1.0
PS, Q_STRIDE, R_STRIDE, PATCH_STD, K_PATCH, PATCH_R, K_VOTE = 16, 8, 2, 2.0, 64, 0.8, 20
F = 16  # |FFT| frequencies kept, as shifts.fdesc
MIN_SHARE = 0.25
THIN_SHARE = 0.1
WRITE_LOW = 0.85
K_VOTE_POOL = 60
TOP, GROW, MIN_FG = 0.98, 9, 30
TRANSFORMS = ['id', 'rot90', 'rot180', 'rot270', 'lr', 'ud', 'transpose', 'antitranspose']
TWIN = {'id': 'rot180', 'rot180': 'id', 'rot90': 'rot270', 'rot270': 'rot90', 'lr': 'ud', 'ud': 'lr',
        'transpose': 'antitranspose', 'antitranspose': 'transpose'}  # same |FFT|: the rerank tries both
COLS = ['kind', 'query', 'ref', 'transform', 'score', 'full_r', 'low_r', 'bg_r', 'shift_y', 'shift_x', 'wrap_full',
        'wrap_low', 'ovl_full', 'ovl_low', 'ovl_frac', 'bg_wrap', 'bg_ovl', 'fg_r', 'bg_low_r', 'bg_low_wrap', 'bg_low_ovl']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FOLDER = None


def tf(a, t):
    if t == 'id': return a
    if t == 'rot90': return torch.rot90(a, 1, (-2, -1))
    if t == 'rot180': return torch.rot90(a, 2, (-2, -1))
    if t == 'rot270': return torch.rot90(a, 3, (-2, -1))
    if t == 'lr': return a.flip(-1)
    if t == 'ud': return a.flip(-2)
    if t == 'transpose': return a.transpose(-2, -1)
    return torch.rot90(a, 2, (-2, -1)).transpose(-2, -1)


def shuffle_blocks(a, name, grid=4):
    h, w = a.shape; bh, bw = h // grid, w // grid
    perm = np.random.default_rng(zlib.crc32(name.encode())).permutation(grid * grid)
    out = a.copy()
    for dst, src in enumerate(perm):
        out[dst // grid * bh:(dst // grid + 1) * bh, dst % grid * bw:(dst % grid + 1) * bw] = \
            a[src // grid * bh:(src // grid + 1) * bh, src % grid * bw:(src % grid + 1) * bw]
    return out


def _load(args):
    split, name, null = args
    im = Image.open(Path(FOLDER) / split / name).convert('L')
    a = np.asarray(im)
    if null: a = shuffle_blocks(a, name); im = Image.fromarray(a)
    th = np.asarray(im.resize((T, T), Image.BOX), dtype=np.uint8)  # as candidates.thumb
    return name, a, th


def _init(folder):
    global FOLDER; FOLDER = folder


def load(split, null=False):
    names = sorted(p.name for p in (Path(FOLDER) / split).iterdir()
                   if p.suffix.lower() in ('.jpg', '.png', '.bmp', '.tif') and not p.name.startswith('._'))  # macOS tar metadata
    with Pool(None, _init, (FOLDER,)) as p:
        res = p.map(_load, [(split, n, null) for n in names], chunksize=128)
    return [r[0] for r in res], [r[1] for r in res], torch.from_numpy(np.stack([r[2] for r in res])).float()


# ---------------------------------------------------------------- candidates
def unit(x):
    x = x - x.mean(1, keepdim=True)
    return x / (x.norm(dim=1, keepdim=True) + 1e-6)


def descr32(th):
    x = Fn.avg_pool2d(th[:, None], 2)[:, 0].reshape(len(th), -1)
    return unit(x), x.std(1)


def fdesc(th):
    x = th - th.mean((1, 2), keepdim=True)
    m = torch.fft.rfft2(x).abs()
    m = torch.cat([m[:, :F + 1, :F + 1], m[:, -F:, :F + 1]], 1).reshape(len(x), -1)
    m = torch.log1p(m); m[:, 0] = 0
    return unit(m)


def topk_search(Q, R, k, same, chunk=4096):
    """Inner products Q @ R.T, top k per row (self-matches removed when same)."""
    vals, idx = [], []
    for c0 in range(0, len(Q), chunk):
        s = Q[c0:c0 + chunk] @ R.T
        if same: s[torch.arange(len(s)), torch.arange(c0, c0 + len(s))] = -2
        v, i = s.topk(k, 1); vals.append(v); idx.append(i)
    return torch.cat(vals), torch.cat(idx)


def ncc_thumbs(a, b):
    """Best wrap-round and overlap-normalised correlation over all shifts, for batches of 64 px thumbnails."""
    n = a.shape[-1]
    a = a - a.mean((1, 2), keepdim=True); b = b - b.mean((1, 2), keepdim=True)
    c = torch.fft.irfft2(torch.conj(torch.fft.rfft2(a)) * torch.fft.rfft2(b), s=(n, n))
    circ = (c / (n * n * a.std((1, 2), unbiased=False)[:, None, None] * b.std((1, 2), unbiased=False)[:, None, None] + 1e-6))
    return torch.maximum(circ.flatten(1).max(1).values, masked_ncc(a, b, MIN_SHARE)[0])


def masked_ncc(a, b, min_share, exclude=0, max_share=None):
    """Pearson over the shared region for every shift s (roll a by s, no wrap, onto b). a, b: (B, H, W).
    Returns best r, its shift (sy, sx) and shared share; shifts with share < min_share (or |sy|+|sx| <= exclude) are
    left out."""
    B, H, W = a.shape; P = (2 * H, 2 * W)
    ma = torch.ones((1, H, W), device=a.device)
    fa = lambda x: torch.fft.rfft2(x, s=P)
    Fa, Fa2, Fb, Fb2, Fm = fa(a), fa(a * a), fa(b), fa(b * b), fa(ma)
    cc = lambda X, Y: torch.fft.irfft2(torch.conj(X) * Y, s=P)  # cc[s] = sum_x X[x] Y[x + s]
    N = cc(Fm, Fm).round().clamp_min(1)
    Sa, Sb = cc(Fa, Fm), cc(Fm, Fb)
    Saa, Sbb, Sab = cc(Fa2, Fm), cc(Fm, Fb2), cc(Fa, Fb)
    num = Sab - Sa * Sb / N
    den = ((Saa - Sa * Sa / N).clamp_min(1e-6) * (Sbb - Sb * Sb / N).clamp_min(1e-6)).sqrt()
    r = num / den
    sy = torch.arange(P[0], device=a.device); sy = torch.where(sy >= H, sy - P[0], sy)
    sx = torch.arange(P[1], device=a.device); sx = torch.where(sx >= W, sx - P[1], sx)
    bad = (N < min_share * H * W) | ((sy[:, None].abs() + sx[None].abs()) <= exclude)
    if max_share is not None: bad = bad | (N >= max_share * H * W)
    r = r.masked_fill(bad, -2)
    v, j = r.flatten(1).max(1)
    jy, jx = j // P[1], j % P[1]
    return v, sy[jy], sx[jx], N.flatten(1)[0][j] / (H * W)  # N is the same for every pair


def patches(th, stride):
    """(n, 64, 64) thumbnails -> chunks of (unit patch vectors (m, 256), owner image index (m,)), patches with
    std >= PATCH_STD only. Kept in chunks (no concatenation) so the reference side never needs twice its memory."""
    out = []
    for c0 in range(0, len(th), 2048):
        p = th[c0:c0 + 2048].to(dev).unfold(1, PS, stride).unfold(2, PS, stride)
        n, a, b = p.shape[:3]; p = p.reshape(n * a * b, PS * PS)
        o = torch.arange(c0, c0 + n, device=dev).repeat_interleave(a * b)
        keep = p.std(1) >= PATCH_STD
        v = unit(p[keep]); out.append((v.half() if dev.type == 'cuda' else v, o[keep]))
    return out


def patch_candidates(qth, rth, same):
    """{query index: {ref index: votes}}: images sharing many patches (partly overlapping tiles)."""
    Rch = patches(rth, R_STRIDE); Qch = patches(qth, Q_STRIDE)
    Q = torch.cat([v for v, _ in Qch]); qown = torch.cat([o for _, o in Qch]); del Qch
    print(f'  patches: {len(Q)} query, {sum(len(v) for v, _ in Rch)} reference', time.strftime('%H:%M:%S'), flush=True)
    votes = [dict() for _ in range(len(qth))]
    for c0 in range(0, len(Q), 1024):
        q = Q[c0:c0 + 1024]; qo = qown[c0:c0 + 1024]
        bv = torch.full((len(q), K_PATCH), -2.0, device=dev); bo = torch.zeros((len(q), K_PATCH), dtype=torch.long, device=dev)
        for R, rown in Rch:
            s = (q @ R.T).float()
            if same: s = s.masked_fill(qo[:, None] == rown[None], -2)
            v, i = s.topk(min(K_PATCH, s.shape[1]), 1)
            v = torch.cat([bv, v], 1); o_ = torch.cat([bo, rown[i]], 1)
            o = v.topk(K_PATCH, 1).indices; bv = v.gather(1, o); bo = o_.gather(1, o)
        owner = bo
        hit = bv >= PATCH_R
        for qq, row_hit, row_own in zip(qo.tolist(), hit.tolist(), owner.tolist()):
            for j in {j for h, j in zip(row_hit, row_own) if h}:  # one vote per query patch and image
                votes[qq][j] = votes[qq].get(j, 0) + 1
        if c0 % (1024 * 500) == 0: print(f'  patch search {c0} / {len(Q)}', time.strftime('%H:%M:%S'), flush=True)
    return votes


def candidates(qth, rth, same):
    """{query index: {(ref index, transform): score}} from whole-image, reranked |FFT| and patch-vote candidates."""
    R32, rs = descr32(rth.to(dev)); R32[rs < MIN_STD] = 0
    Rf = fdesc(rth.to(dev)); Rf[rs < MIN_STD] = 0
    q32std = descr32(qth.to(dev))[1]
    whole = [dict() for _ in range(len(qth))]; fft = [dict() for _ in range(len(qth))]
    for t in TRANSFORMS:
        qt = tf(qth.to(dev), t)
        v, i = topk_search(descr32(qt)[0], R32, K_WHOLE, same and t == 'id')
        for q, (vv, ii) in enumerate(zip(v.tolist(), i.tolist())):
            if q32std[q] < MIN_STD: continue
            for s, j in zip(vv, ii):
                if s >= MIN_THUMB_R and s > whole[q].get(j, (-2,))[0]: whole[q][j] = (s, t)
        v, i = topk_search(fdesc(qt), Rf, K_FFT, same and t == 'id')
        for q, (vv, ii) in enumerate(zip(v.tolist(), i.tolist())):
            if q32std[q] < MIN_STD: continue
            for s, j in zip(vv, ii):
                if same and j == q: continue
                if s > fft[q].get(j, (-2,))[0]: fft[q][j] = (s, t)
        print('  searched', t, time.strftime('%H:%M:%S'), flush=True)
    # rerank the |FFT| candidates by a 64 px alignment, keep K_SHIFT
    pairs = [(q, j, tt) for q in range(len(qth)) for j, (s, t) in sorted(fft[q].items(), key=lambda kv: -kv[1][0])[:K_FFT]
             for tt in (t, TWIN[t])]
    score = torch.empty(len(pairs))
    for c0 in range(0, len(pairs), 4096):
        chunk = pairs[c0:c0 + 4096]
        a = torch.stack([tf(qth[q], t) for q, _, t in chunk]).to(dev); b = rth[[j for _, j, _ in chunk]].to(dev)
        score[c0:c0 + len(chunk)] = ncc_thumbs(a, b).cpu()
    best = [dict() for _ in range(len(qth))]
    for (q, j, t), s in zip(pairs, score.tolist()):
        if s > best[q].get(j, (-2,))[0]: best[q][j] = (s, t)
    pv = patch_candidates(qth, rth, same)
    vpairs = [(q, j) for q in range(len(qth)) for j, n in sorted(pv[q].items(), key=lambda kv: -kv[1])[:K_VOTE_POOL]]
    vscore = torch.empty(len(vpairs))
    for c0 in range(0, len(vpairs), 4096):
        chunk = vpairs[c0:c0 + 4096]
        a = qth[[q for q, _ in chunk]].to(dev); b = rth[[j for _, j in chunk]].to(dev)
        a = a - a.mean((1, 2), keepdim=True); b = b - b.mean((1, 2), keepdim=True)
        vscore[c0:c0 + len(chunk)] = masked_ncc(a, b, THIN_SHARE, exclude=0)[0].cpu()
    vbest = [dict() for _ in range(len(qth))]
    for (q, j), s in zip(vpairs, vscore.tolist()): vbest[q][j] = s
    out = []
    for q in range(len(qth)):
        c = {(j, t): s for j, (s, t) in whole[q].items()}
        for j, (s, t) in sorted(best[q].items(), key=lambda kv: -kv[1][0])[:K_SHIFT]:
            c.setdefault((j, t), s)
        for j, s in sorted(vbest[q].items(), key=lambda kv: -kv[1])[:K_VOTE]:
            c.setdefault((j, 'id'), s)
        out.append(c)
    print(f'  candidates: {sum(map(len, out))} pairs for {len(out)} queries', time.strftime('%H:%M:%S'), flush=True)
    return out


# ---------------------------------------------------------------- full-resolution check
def mpearson(x, y, m):
    n = m.sum((1, 2)).clamp_min(1)
    mx = (x * m).sum((1, 2)) / n; my = (y * m).sum((1, 2)) / n
    dx = (x - mx[:, None, None]) * m; dy = (y - my[:, None, None]) * m
    return (dx * dy).sum((1, 2)) / ((dx * dx).sum((1, 2)).sqrt() * (dy * dy).sum((1, 2)).sqrt() + 1e-6)


def mquantile(x, m, q):
    """Per-row q-quantile of x over the mask m (nearest rank)."""
    v = x.masked_fill(m == 0, float('-inf')).flatten(1).sort(1).values
    n = m.flatten(1).sum(1).long(); start = v.shape[1] - n
    k = (start + (q * (n - 1).clamp_min(0)).floor().long()).clamp(0, v.shape[1] - 1)
    return v.gather(1, k[:, None])[:, 0]


def stats(x, y, m):
    """full r, 8x averaged r, background r, target r over the mask m (batched)."""
    full = mpearson(x, y, m)
    px, py, pm = (Fn.avg_pool2d(v[:, None], 8)[:, 0] for v in (x * m, y * m, m))
    pm1 = (pm > 0.999).float()
    low = mpearson(px / pm.clamp_min(1e-6), py / pm.clamp_min(1e-6), pm1)
    hot = ((x > mquantile(x, m, TOP)[:, None, None]) | (y > mquantile(y, m, TOP)[:, None, None])).float() * m
    grown = Fn.max_pool2d(hot[:, None], GROW, 1, GROW // 2)[:, 0] * m
    keep = m * (1 - grown)
    bg = mpearson(x, y, keep); fg = mpearson(x, y, grown)
    bg = torch.where(keep.sum((1, 2)) >= 0.2 * m.sum((1, 2)), bg, torch.full_like(bg, float('nan')))
    fg = torch.where(grown.sum((1, 2)) >= MIN_FG, fg, torch.full_like(fg, float('nan')))
    qx, qy, qk = (Fn.avg_pool2d(v[:, None], 4)[:, 0] for v in (x * keep, y * keep, keep))
    qk1 = (qk > 0.999).float()
    bgl = mpearson(qx / qk.clamp_min(1e-6), qy / qk.clamp_min(1e-6), qk1)
    bgl = torch.where(qk1.sum((1, 2)) * 16 >= 0.2 * m.sum((1, 2)), bgl, torch.full_like(bgl, float('nan')))
    return full, low, bg, fg, bgl


def roll_rows(a, sy, sx, wrap):
    """Per-row roll of a (B, H, W) by (sy, sx); without wrap the region rolled in is masked out."""
    B, H, W = a.shape
    iy = (torch.arange(H, device=a.device)[None] - sy[:, None]) % H
    ix = (torch.arange(W, device=a.device)[None] - sx[:, None]) % W
    out = a.gather(1, iy[:, :, None].expand(B, H, W)).gather(2, ix[:, None, :].expand(B, H, W))
    if wrap: return out, torch.ones_like(a)
    yy = torch.arange(H, device=a.device)[None]; xx = torch.arange(W, device=a.device)[None]
    my = ((yy - sy[:, None]) >= 0) & ((yy - sy[:, None]) < H)
    mx = ((xx - sx[:, None]) >= 0) & ((xx - sx[:, None]) < W)
    return out, (my[:, :, None] & mx[:, None, :]).float()


def check_batch(a, b):
    """a, b: (B, H, W) float on dev, a already turned and resized. -> dict of per-pair values."""
    B, H, W = a.shape
    one = torch.ones_like(a)
    z = stats(a, b, one)
    # wrap-round shift: circular cross-correlation (excluding |s| <= 2: the zero-shift case is the 'whole' row)
    az = a - a.mean((1, 2), keepdim=True); bz = b - b.mean((1, 2), keepdim=True)
    c = torch.fft.irfft2(torch.conj(torch.fft.rfft2(az)) * torch.fft.rfft2(bz), s=(H, W))
    sy = torch.arange(H, device=dev); sy = torch.where(sy > H // 2, sy - H, sy)
    sx = torch.arange(W, device=dev); sx = torch.where(sx > W // 2, sx - W, sx)
    c = c.masked_fill(((sy[:, None].abs() + sx[None].abs()) <= 2)[None], float('-inf'))
    j = c.flatten(1).argmax(1); wy, wx = sy[j // W], sx[j % W]
    aw, mw = roll_rows(a, wy, wx, True); w = stats(aw, b, mw)
    # overlap shift: overlap-normalised cross-correlation
    _, oy, ox, frac = masked_ncc(a, b, MIN_SHARE, exclude=2)
    ao, mo = roll_rows(a, oy, ox, False); o = stats(ao, b, mo)
    # thin overlap: the best shift whose shared area is at least THIN_SHARE but under MIN_SHARE
    _, ty, tx, tfrac = masked_ncc(a, b, THIN_SHARE, exclude=2, max_share=MIN_SHARE)
    at, mt = roll_rows(a, ty, tx, False); th = stats(at, b, mt)
    return {'z': z, 'w': (wy, wx) + w, 'o': (oy, ox, frac) + o, 't': (ty, tx, tfrac) + th}


def full_check(pairs, qimgs, rimgs, qnames, rnames, writer, rules):
    """pairs: [(q, j, score, transform)]; grouped by reference shape and run in batches."""
    by = {}
    for p in pairs: by.setdefault(rimgs[p[1]].shape, []).append(p)
    done = 0
    for shape, ps in sorted(by.items(), key=lambda kv: -len(kv[1])):
        H, W = shape; bs = max(1, int(2.5e7 // (H * W)))
        for c0 in range(0, len(ps), bs):
            chunk = ps[c0:c0 + bs]
            a = []
            for q, j, s, t in chunk:
                x = tf(torch.from_numpy(qimgs[q]).to(dev).float(), t)
                if x.shape != shape:
                    x = Fn.interpolate(x[None, None], size=shape, mode='bilinear', align_corners=False, antialias=True)[0, 0]
                a.append(x)
            a = torch.stack(a); b = torch.stack([torch.from_numpy(rimgs[j]).to(dev).float() for _, j, _, _ in chunk])
            res = check_batch(a, b)
            z = [v.tolist() for v in res['z']]; w = [v.tolist() for v in res['w']]; o = [v.tolist() for v in res['o']]
            tt = [v.tolist() for v in res['t']]
            for k, (q, j, s, t) in enumerate(chunk):
                base = {'query': qnames[q], 'ref': rnames[j], 'transform': t, 'score': f'{s:.4f}'}
                f4 = lambda v: 'nan' if v != v else f'{v:.4f}'
                rows = [dict(base, kind='whole', full_r=f4(z[0][k]), low_r=f4(z[1][k]), bg_r=f4(z[2][k]), fg_r=f4(z[3][k]),
                             bg_low_r=f4(z[4][k])),
                        dict(base, kind='shift', shift_y=w[0][k], shift_x=w[1][k], wrap_full=f4(w[2][k]),
                             wrap_low=f4(w[3][k]), ovl_full='-1', ovl_low='-1', ovl_frac='0', bg_wrap=f4(w[4][k]),
                             bg_ovl='', fg_r=f4(w[5][k]), bg_low_wrap=f4(w[6][k])),
                        dict(base, kind='shift', shift_y=o[0][k], shift_x=o[1][k], wrap_full='-1', wrap_low='-1',
                             ovl_full=f4(o[3][k]), ovl_low=f4(o[4][k]), ovl_frac=f'{o[2][k]:.3f}', bg_wrap='',
                             bg_ovl=f4(o[5][k]), fg_r=f4(o[6][k]), bg_low_ovl=f4(o[7][k])),
                        dict(base, kind='thin', shift_y=tt[0][k], shift_x=tt[1][k], wrap_full='-1', wrap_low='-1',
                             ovl_full=f4(tt[3][k]), ovl_low=f4(tt[4][k]), ovl_frac=f'{tt[2][k]:.3f}', bg_wrap='',
                             bg_ovl=f4(tt[5][k]), fg_r=f4(tt[6][k]), bg_low_ovl=f4(tt[7][k]))]
                for r in rows:
                    r = {c: r.get(c, '') for c in COLS}
                    if rules(r): writer.writerow(r)
            done += len(chunk)
        print(f'  checked {done} / {len(pairs)} pairs (shape {shape})', time.strftime('%H:%M:%S'), flush=True)


def check_given(pairs_file, out, rules):
    """Check given pairs (no search), under the identity transform."""
    want = list(csv.DictReader(open(pairs_file)))
    names = sorted({(r['query_split'], r['query']) for r in want} | {(r['ref_split'], r['ref']) for r in want})
    with Pool(None, _init, (FOLDER,)) as p:
        res = p.map(_load, [(s, n, False) for s, n in names], chunksize=128)
    idx = {sn: k for k, sn in enumerate(names)}
    imgs = [r[1] for r in res]; nm = [r[0] for r in res]
    pairs = [(idx[(r['query_split'], r['query'])], idx[(r['ref_split'], r['ref'])], 0.0, 'id') for r in want]
    print(len(pairs), 'given pairs,', len(names), 'images', flush=True)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, COLS); w.writeheader()
        full_check(pairs, imgs, imgs, nm, nm, w, rules)
    print('->', out, flush=True)


def main(folder, qsplit, rsplit, out, null=False, pairs_file=None):
    _init(folder)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import groups as G
    def lowest(r):
        v = [float(r[k]) for k in ('low_r', 'wrap_low', 'ovl_low') if r.get(k) not in (None, '', 'nan', '-1')]
        return max(v) if v else -1
    rule = lambda r: (G.first_pass(r) if r['kind'] == 'whole' else G.shifted({**r, 'box_recall': ''})) is not None
    rules = lambda r: rule(r) or lowest(r) >= WRITE_LOW
    t0 = time.time()
    if pairs_file:
        return check_given(pairs_file, out, rules)
    qn, qi, qth = load(qsplit)
    rn, ri, rth = (qn, qi, qth) if (qsplit == rsplit and not null) else load(rsplit, null)
    print(f'{qsplit} {len(qn)} vs {rsplit} {len(rn)}{" (null)" if null else ""}: loaded {time.time() - t0:.0f}s', flush=True)
    cand = candidates(qth, rth, qsplit == rsplit and not null)
    pairs = [(q, j, s, t) for q, c in enumerate(cand) for (j, t), s in c.items()]
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, COLS); w.writeheader()
        full_check(pairs, qi, ri, qn, rn, w, rules)
    print('->', out, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], '--null' in a, a[a.index('--pairs') + 1] if '--pairs' in a else None)
