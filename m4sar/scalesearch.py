"""M4-SAR scale check: copies at another zoom between val / test and train, which pixsearch.py (same zoom only) misses.

A FINE tile is a magnified view of part of a COARSE tile (zoom z = 1.1-2: the fine tile covers 1/z of the coarse tile's
width). Both directions leak: a val / test tile can be the fine one (direction 1, coarse tile in train) or the coarse
one (direction 2, fine tile in train). Version-1 images only (k and k + 56,087 are the same tile); optical image for the
search, SAR checked at the found alignment, as in pixsearch.py.

1. Candidates: the fine tile's 64 px thumbnail is shrunk by 1/z for z in ZOOMS, under the 8 flips / turns; its 16 px
   patches (every QSTRIDE px, slightly blurred) vote for coarse images among the coarse images' reference patches
   (every 2 px, pixsearch.py). Images with >= MIN_VOTES votes (top K_VOTE per fine tile) are candidates, with the
   zoom and turn that got most votes.
2. Stage A, 128 px: the fine tile (voted turn) shrunk to 128 / z for every zoom in ZOOMS x A_STEPS (the voted zoom
   is often a neighbour of the right one), slightly blurred; Pearson over the shared region only, for every shift where
   >= MIN_SHARE of the shrunk fine tile lies on the coarse tile (FFT).
3. Stage B, full resolution, pairs with stage-A r >= STAGE_B_R: the same at 512 px for z = stage-A zoom x B_STEPS,
   within +-6 px of 4x the stage-A shift, both images slightly blurred (shrinking filters differ) (r_opt); then the
   SAR images, the fine one shrunk the same way, at that alignment (r_sar).

Controls, in the same job:
   null    N_NULL val / test fine tiles against block-shuffled train tiles (pixsearch.py), and N_NULL block-shuffled
           train tiles as fine tiles against val / test: no copies possible.
   planted N_PLANT val / test images copied into the pool at a random zoom 1.1-2 (re-saved as JPEG): half as a coarse
           copy (the image shrunk and pasted into another train image, fully inside or half outside), half as a fine
           copy (a crop of the image enlarged to 512 px). Recall = found with the right zoom (+-2%), turn and shift (+-3 px).

    python scalesearch.py M4-SAR.zip OUT_DIR       (LIMIT=2000 for a trial; RESUME=<earlier OUT_DIR> reuses its cands_*.npz;
                                                    ALL_PAIRS=1: every image against every other, main1 only)
Output: OUT_DIR/pairs_<set>.csv (every pair that reached stage B; sets main1, main2, null1, null2, planted1, planted2),
planted_truth.json, search.json.
"""
import csv, io, json, os, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import pixsearch as px
from pixsearch import log, small, turn, tturn, blur, patches, shuffle, perm_of, build_ref

ZOOMS = [1.1, 1.2, 1.32, 1.45, 1.6, 1.78, 2.0]       # ~10% apart
QSTRIDE = 8
K_PATCH, PATCH_R, MIN_VOTES, K_VOTE = 48, 0.8, 2, 20
A_STEPS = [0.95, 0.975, 1.0, 1.025, 1.05]
B_STEPS = [round(0.97 + 0.005 * k, 3) for k in range(13)]   # +-3%
MIN_SHARE = 0.3
STAGE_B_R = 0.8
N_NULL = int(os.environ.get('N_NULL', 6000))
N_PLANT = 400
PLANT_BASE = 10 ** 6
BQ = 8
DEV = 'cuda'
WORKERS = int(os.environ.get('WORKERS', 3))
REF_CHUNK = int(os.environ.get('REF_CHUNK', 250_000))
CHUNK = 2048
PLANTED = {}

# ---------------------------------------------------------------- pool images


def img(i, null):        # (optical, sar) uint8 512 px: real, planted, or block-shuffled (null)
    if i >= PLANT_BASE: return PLANTED[i]
    o, s = px._read(px.MEMBERS[i]['optical']), px._read(px.MEMBERS[i]['sar'])
    if null: p = perm_of(i); o, s = shuffle(o, p), shuffle(s, p)
    return o, s


def _load(job):
    f, fnull, c, cnull, t = job
    of, sf = img(f, fnull); oc, sc = img(c, cnull)
    return turn(of, t), turn(sf, t), oc, sc

# ---------------------------------------------------------------- candidates


def qpatches(x128):
    """fine thumbnails (b, 128, 128) -> patches of the 64 px thumbnail shrunk by every zoom, under 8 turns:
    Q (n, 256) f16 and labels (n, 3) = (image in batch, zoom index, turn)."""
    x64 = F.avg_pool2d(x128[:, None].float(), 2)
    out, lab = [], []
    for zi, z in enumerate(ZOOMS):
        n = int(round(64 / z))
        xs = F.interpolate(x64, size=(n, n), mode='bilinear', antialias=True, align_corners=False)[:, 0]
        for t in range(8):
            p = patches(tturn(xs, t), QSTRIDE)
            P = p.shape[1]
            out.append(p.reshape(-1, 256))
            im = torch.arange(len(x128), device=x128.device).repeat_interleave(P)
            lab.append(torch.stack([im, torch.full_like(im, zi), torch.full_like(im, t)], 1))
    return torch.cat(out).half(), torch.cat(lab)


def candidates(FT, R, nref, name, qid, rid):
    """FT (n, 128, 128) cpu f16 fine thumbnails; R reference patches of the coarse images (first nref used); qid, rid:
    pool id of each fine / coarse image (an image never pairs with itself). Returns {(fine index, coarse index): (zoom
    index, turn, votes)}."""
    g2 = ((64 - 16) // px.RSTRIDE + 1) ** 2
    Rv = R[:nref * g2]; out = {}; t1 = time.time()
    qid_t, rid_t = torch.as_tensor(np.asarray(qid), device=DEV), torch.as_tensor(np.asarray(rid)[:nref], device=DEV)
    for s in range(0, len(FT), BQ):
        if s and s % 4000 < BQ:
            el = time.time() - t1; log(f'{name} candidates: {s} / {len(FT)}, ~{el / s * (len(FT) - s) / 60:.0f} min left')
        Q, lab = qpatches(FT[s:s + BQ].to(DEV))
        bv = torch.full((len(Q), K_PATCH), -2.0, device=DEV); bi = torch.zeros((len(Q), K_PATCH), dtype=torch.long, device=DEV)
        for c in range(0, len(Rv), REF_CHUNK):
            v, ix = (Q @ Rv[c:c + REF_CHUNK].T).topk(K_PATCH, dim=1)
            v = torch.cat([bv, v.float()], 1); ix = torch.cat([bi, ix + c], 1)
            bv, o = v.topk(K_PATCH, dim=1); bi = ix.gather(1, o)
        ref = bi // g2
        a, p = torch.nonzero((bv >= PATCH_R) & (rid_t[ref] != qid_t[s + lab[:, 0]][:, None]), as_tuple=True)
        if len(a) == 0: continue
        pr = torch.stack([a, ref[a, p]], 1).unique(dim=0)                  # each query patch votes once per image
        k, cnt = torch.cat([lab[pr[:, 0]], pr[:, 1:2]], 1).unique(dim=0, return_counts=True)
        keep = cnt >= MIN_VOTES
        per = defaultdict(dict)
        for (i, zi, t, r), c_ in zip(k[keep].tolist(), cnt[keep].tolist()):
            if r not in per[i] or c_ > per[i][r][2]: per[i][r] = (zi, t, c_)
        for i, d in per.items():
            for r, v_ in sorted(d.items(), key=lambda kv: -kv[1][2])[:K_VOTE]: out[(s + i, r)] = v_
    log(f'{name} candidates: {len(out)} pairs, {time.time() - t1:.0f} s')
    return out


def cached(name, fn, out):
    p = Path(os.environ.get('RESUME', out)) / f'cands_{name}.npz'
    if p.exists():
        z = np.load(p); d = {(int(a), int(b)): (int(c), int(e), int(f)) for a, b, c, e, f in z['k']}
        log(f'{name}: {len(d)} candidates loaded from {p}'); return d
    d = fn(); k = np.array([(a, b, *v) for (a, b), v in d.items()], dtype=np.int64).reshape(-1, 5)
    np.savez(Path(out) / f'cands_{name}.npz', k=k); return d

# ---------------------------------------------------------------- alignment


def shrink(a, n):        # (B, H, W) f32 -> (B, n, n), area-averaged
    return F.interpolate(a[:, None], size=(n, n), mode='bilinear', antialias=True, align_corners=False)[:, 0]


def place(a, H):         # (B, n, n) -> top-left of an H x H canvas, and its mask
    n = a.shape[-1]; c = torch.zeros((len(a), H, H), device=a.device); m = torch.zeros_like(c)
    c[:, :n, :n] = a; m[:, :n, :n] = 1
    return c, m


def mncc(a, ma, b, mb, min_share, centre=None, win=6):
    """Masked, overlap-normalised Pearson for every shift (a pixel u matches b pixel u + s). a, ma: the shrunk fine tile
    on a canvas and its mask; b, mb: the coarse tile. Shifts where < min_share of a's mask lies on b are left out.
    Returns r, dy, dx, share of the fine tile, share of the coarse tile, each (B,)."""
    B, H, W = b.shape; P = (2 * H, 2 * W)

    def nz(x, m):
        k = m.sum((-2, -1), keepdim=True).clamp_min(1)
        mu = (x * m).sum((-2, -1), keepdim=True) / k
        sd = ((((x - mu) * m) ** 2).sum((-2, -1), keepdim=True) / k).sqrt().clamp_min(1e-3)
        return (x - mu) / sd * m
    a, b = nz(a, ma), nz(b, mb)
    f = lambda x: torch.fft.rfft2(x, s=P)
    cc = lambda x, y: torch.fft.irfft2(x.conj() * y, s=P)
    fma, fmb, fa, fb = f(ma), f(mb), f(a), f(b)
    n = cc(fma, fmb)
    sab, sa, sb = cc(fa, fb), cc(fa, fmb), cc(fma, fb)
    saa, sbb = cc(f(a * a), fmb), cc(fma, f(b * b))
    va, vb = n * saa - sa * sa, n * sbb - sb * sb
    r = (n * sab - sa * sb) / torch.sqrt((va * vb).clamp_min(1e-6))
    area = ma.sum((-2, -1))
    bad = (n < min_share * area[:, None, None] - 0.5) | (va < 1e-3 * n * n) | (vb < 1e-3 * n * n)
    r = torch.where(bad, torch.full_like(r, -2.0), r)
    ky = torch.arange(2 * H, device=a.device); ky = torch.where(ky < H, ky, ky - 2 * H)
    kx = torch.arange(2 * W, device=a.device); kx = torch.where(kx < W, kx, kx - 2 * W)
    if centre is not None:
        cy, cx = centre
        m = ((ky[None, :, None] - cy[:, None, None]).abs() > win) | ((kx[None, None, :] - cx[:, None, None]).abs() > win)
        r = torch.where(m, torch.full_like(r, -2.0), r)
    v, ix = r.reshape(B, -1).max(1)
    dy, dx = ky[ix // (2 * W)], kx[ix % (2 * W)]
    nn_ = n.reshape(B, -1).gather(1, ix[:, None])[:, 0]
    return v, dy, dx, nn_ / area, nn_ / (H * W)


def stage_a(cands, FT, CT, name, diag=frozenset()):
    """Returns the pairs passing STAGE_B_R, and the best row of every pair in diag (planted copies) whatever its r."""
    seen = {}; keys = list(cands)
    out, n_all = [], len(cands)
    if True:
        for s in range(0, len(keys), 128):
            ks = keys[s:s + 128]
            a = FT[[k[0] for k in ks]].to(DEV).float(); ts = [cands[k][1] for k in ks]
            for t in range(1, 8):
                m = [i for i, tt in enumerate(ts) if tt == t]
                if m: a[m] = tturn(a[m], t)
            b = blur(CT[[k[1] for k in ks]].to(DEV).float()[:, None])[:, 0]; mb = torch.ones_like(b)
            best = None
            for n in sorted({int(round(128 / (z * mult))) for z in ZOOMS for mult in A_STEPS}):
                ca, ma = place(blur(shrink(a, n)[:, None])[:, 0], 128)
                v, dy, dx, sf, sc = mncc(ca, ma, b, mb, MIN_SHARE)
                cur = torch.stack([v, dy.float(), dx.float(), sf, sc, torch.full_like(v, 128 / n)], 1)
                best = cur if best is None else torch.where((cur[:, :1] > best[:, :1]), cur, best)
            for k, row in zip(ks, best.tolist()):
                if row[0] >= STAGE_B_R: out.append((k, cands[k], row))
                if k in diag: seen[k] = (cands[k], row)
    log(f'{name}: stage A done, {len(out)} of {n_all} pass r >= {STAGE_B_R}')
    return out, seen


def stage_b(rows_a, fid, cid, fnull, cnull, name, sink, tag=None):
    rows_a.sort(key=lambda p: p[0])
    jobs = [(int(fid[f]), fnull, int(cid[c]), cnull, v[1]) for (f, c), v, _ in rows_a]
    t1, n = time.time(), 0
    with Pool(WORKERS) as pool:
        nxt = pool.map_async(_load, jobs[:CHUNK], chunksize=16)
        for c0 in range(0, len(jobs), CHUNK):
            got_all = nxt.get()
            if c0 + CHUNK < len(jobs): nxt = pool.map_async(_load, jobs[c0 + CHUNK:c0 + 2 * CHUNK], chunksize=16)
            new = []
            for p, g in zip(rows_a[c0:c0 + CHUNK], got_all):
                (fi, ci), (zi, t, votes), (rA, dyA, dxA, _, _, zA) = p
                of, sf, oc, sc = [torch.from_numpy(x).to(DEV).float() for x in g]
                ns = sorted({int(round(512 / (zA * m))) for m in B_STEPS})
                cas, mas = zip(*[place(blur(shrink(of[None], k)[:, None])[:, 0], 512) for k in ns])
                ca, ma = torch.cat(cas), torch.cat(mas)
                b = blur(oc[None, None])[0].expand(len(ns), -1, -1).contiguous(); mb = torch.ones_like(b)
                cy = torch.full((len(ns),), 4 * dyA, device=DEV); cx = torch.full((len(ns),), 4 * dxA, device=DEV)
                v, dy, dx, sh_f, sh_c = mncc(ca, ma, b, mb, MIN_SHARE, centre=(cy, cx))
                j = int(v.argmax()); r_opt = float(v[j]); k = ns[j]; y, x = int(dy[j]), int(dx[j])
                r_sar = -2.0
                if r_opt > -2:
                    s_small = blur(shrink(sf[None], k)[:, None])[0, 0]; scb = blur(sc[None, None])[0, 0]
                    ya, yb = max(0, -y), min(k, 512 - y); xa, xb = max(0, -x), min(k, 512 - x)
                    u = s_small[ya:yb, xa:xb].reshape(-1); w = scb[ya + y:yb + y, xa + x:xb + x].reshape(-1)
                    u, w = u - u.mean(), w - w.mean()
                    r_sar = float((u * w).sum() / torch.sqrt((u * u).sum() * (w * w).sum()).clamp_min(1e-6))
                row = {'fine': int(fid[fi]), 'coarse': int(cid[ci]), 'zoom_voted': ZOOMS[zi], 't': t, 'votes': votes,
                       'zoom_a': round(zA, 4), 'r_a': round(rA, 4), 'zoom': round(512 / k, 4), 'dy': y, 'dx': x,
                       'share_fine': round(float(sh_f[j]), 3), 'share_coarse': round(float(sh_c[j]), 3),
                       'r_opt': round(r_opt, 4), 'r_sar': round(r_sar, 4)}
                if tag: row['planted'] = tag(row)
                new.append(row)
            sink(new); n += len(new)
            rate = n / max(time.time() - t1, 1e-6)
            if c0 % (10 * CHUNK) == 0 or n >= len(jobs):
                log(f'{name}: stage B {n} / {len(jobs)} ({rate:.0f} pairs/s, ~{(len(jobs) - n) / max(rate, 1e-6) / 60:.0f} min left)')
    return n


COLS = ['fine', 'coarse', 'fine_split', 'coarse_split', 'zoom_voted', 't', 'votes', 'zoom_a', 'r_a', 'zoom', 'dy', 'dx',
        'share_fine', 'share_coarse', 'r_opt', 'r_sar', 'planted']


def appender(path, split, fnull=False, cnull=False):
    with open(path, 'w', newline='') as fh: csv.DictWriter(fh, fieldnames=COLS, lineterminator='\n').writeheader()
    def sink(rows):
        with open(path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, lineterminator='\n', extrasaction='ignore')
            for x in rows:
                x.setdefault('planted', '')
                x['fine_split'] = 'planted' if x['fine'] >= PLANT_BASE else split[x['fine']] + ('(null)' if fnull else '')
                x['coarse_split'] = 'planted' if x['coarse'] >= PLANT_BASE else split[x['coarse']] + ('(null)' if cnull else '')
                w.writerow(x)
    return sink

# ---------------------------------------------------------------- main


def paste(dst, src, y0, x0):
    H, W = dst.shape; h, w = src.shape
    ya, yb, xa, xb = max(0, y0), min(H, y0 + h), max(0, x0), min(W, x0 + w)
    dst[ya:yb, xa:xb] = src[ya - y0:yb - y0, xa - x0:xb - x0]


def jpeg(a):
    buf = io.BytesIO(); Image.fromarray(a).save(buf, 'JPEG', quality=85); return np.asarray(Image.open(buf).convert('L'))


def main(zpath, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    px.ZIP = zpath; px.MEMBERS = px.index(zpath)
    ids = sorted(i for i in px.MEMBERS if i <= px.HALF)
    if os.environ.get('LIMIT'): ids = ids[::max(1, len(ids) // int(os.environ['LIMIT']))]
    split = {i: px.MEMBERS[i]['split'] for i in px.MEMBERS}
    pos = {i: k for k, i in enumerate(ids)}
    with Pool(4) as pool: th = dict(pool.imap_unordered(px._thumb, ids, chunksize=64))
    T = torch.from_numpy(np.stack([th[i] for i in ids])); del th
    vt = [i for i in ids if split[i] != 'train']; tr = [i for i in ids if split[i] == 'train']
    ALL = os.environ.get('ALL_PAIRS') == '1'    # every image against every other (both roles): links for grouping
    qsrc = list(vt)                              # planted copies are made of val / test images either way
    if ALL: vt, tr = list(ids), list(ids)
    VT, TR = T[[pos[i] for i in vt]], T[[pos[i] for i in tr]]
    log('thumbnails', T.shape, 'val/test', len(vt), 'train', len(tr))
    res = {'images': len(ids), 'fine side': len(vt), 'coarse side': len(tr), 'all pairs': ALL, 'zooms': ZOOMS}

    # planted copies
    rng = np.random.default_rng(1)
    qv = [i for i in qsrc if float(T[pos[i]].float().std()) > 10]
    src = rng.choice(qv, min(N_PLANT, len(qv)), replace=False); truth = {}
    for n, q in enumerate(src):
        q = int(q); pid = PLANT_BASE + n; z = float(rng.uniform(1.1, 2.0)); m = int(round(512 / z))
        o, s = img(q, False)
        if n % 2 == 0:    # coarse copy in train: the image shrunk to m px, pasted into another train image
            fo, fs = img(int(rng.choice([i for i in tr if i != q][:50_000])), False); po, ps = fo.copy(), fs.copy()
            inside = n % 4 == 0
            y0, x0 = (int(v) for v in rng.integers(0, 512 - m + 1, 2))
            if not inside: y0 = 512 - m // 2 if rng.random() < 0.5 else -(m - m // 2)
            so = np.asarray(Image.fromarray(o).resize((m, m), Image.BOX)); ss = np.asarray(Image.fromarray(s).resize((m, m), Image.BOX))
            paste(po, so, y0, x0); paste(ps, ss, y0, x0)
            PLANTED[pid] = (jpeg(po), ps)
            truth[pid] = {'kind': 'coarse copy in train', 'fine': q, 'coarse': pid, 'zoom': 512 / m, 'dy': y0, 'dx': x0, 'inside': inside}
        else:             # fine copy in train: an m px crop of the image, enlarged to 512 px
            y0, x0 = (int(v) for v in rng.integers(0, 512 - m + 1, 2))
            po = np.asarray(Image.fromarray(o[y0:y0 + m, x0:x0 + m]).resize((512, 512), Image.BICUBIC))
            ps = np.asarray(Image.fromarray(s[y0:y0 + m, x0:x0 + m]).resize((512, 512), Image.BICUBIC))
            PLANTED[pid] = (jpeg(po), ps.copy())
            truth[pid] = {'kind': 'fine copy in train', 'fine': pid, 'coarse': q, 'zoom': 512 / m, 'dy': y0, 'dx': x0, 'inside': True}
    (out / 'planted_truth.json').write_text(json.dumps({str(k): v for k, v in truth.items()}))
    p1 = [p for p in sorted(truth) if truth[p]['kind'].startswith('coarse')]; p2 = [p for p in sorted(truth) if p not in p1]
    PT1 = torch.from_numpy(np.stack([small(PLANTED[p][0]) for p in p1])); PT2 = torch.from_numpy(np.stack([small(PLANTED[p][0]) for p in p2]))
    log('planted', len(p1), 'coarse copies,', len(p2), 'fine copies')

    def tag(row):         # right zoom, turn and shift?
        p = row['coarse'] if row['coarse'] >= PLANT_BASE else row['fine'] if row['fine'] >= PLANT_BASE else None
        if p is None: return 'other'
        t_ = truth[p]
        if t_['fine'] != row['fine'] or t_['coarse'] != row['coarse']: return 'other'
        ok = abs(row['zoom'] / t_['zoom'] - 1) <= 0.02 and row['t'] == 0 and abs(row['dy'] - t_['dy']) <= 3 and abs(row['dx'] - t_['dx']) <= 3
        return f"{t_['kind'].split()[0]}:{t_['zoom']:.3f}:{int(t_['inside'])}:{int(ok)}"

    rng2 = np.random.default_rng(2)
    nq1 = sorted(rng2.choice(len(vt), min(N_NULL, len(vt)), replace=False).tolist())
    nq2 = sorted(rng2.choice(len(tr), min(N_NULL, len(tr)), replace=False).tolist())
    NTR = torch.from_numpy(np.stack([shuffle(TR[k].numpy(), perm_of(i)) for k, i in enumerate(tr)]))
    vt_a, tr_a = np.array(vt), np.array(tr)
    fp1_ids = np.array([truth[p]['fine'] for p in p1])

    # direction 1: val / test fine, train coarse (+ the planted coarse copies, used by the planted run only)
    C1 = torch.cat([TR, PT1]); R, _ = build_ref(C1)
    c_main1 = cached('main1', lambda: candidates(VT, R, len(tr), 'main1', vt_a, tr_a), out)
    FP1 = torch.stack([T[pos[truth[p]['fine']]] for p in p1])
    c_pl1 = cached('planted1', lambda: candidates(FP1, R, len(C1), 'planted1', fp1_ids, np.array(tr + p1)), out)
    del R; torch.cuda.empty_cache()
    # direction 2: train fine (+ planted fine copies, + shuffled train for the null), val / test coarse
    R, _ = build_ref(VT)
    c_main2 = {} if ALL else cached('main2', lambda: candidates(TR, R, len(vt), 'main2', tr_a, vt_a), out)   # ALL: main1 has both roles
    c_pl2 = cached('planted2', lambda: candidates(PT2, R, len(vt), 'planted2', np.array(p2), vt_a), out)
    c_null2 = cached('null2', lambda: candidates(NTR[nq2], R, len(vt), 'null2', tr_a[nq2], vt_a), out)
    del R; torch.cuda.empty_cache()
    # null, direction 1: val / test fine against shuffled train
    R, _ = build_ref(NTR)
    c_null1 = cached('null1', lambda: candidates(VT[nq1], R, len(tr), 'null1', vt_a[nq1], tr_a), out)
    del R; torch.cuda.empty_cache()
    log(f'GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')

    sets = [('planted1', c_pl1, FP1, C1, fp1_ids, np.array(tr + p1), False, False),
            ('planted2', c_pl2, PT2, VT, np.array(p2), vt_a, False, False),
            ('null1', c_null1, VT[nq1], NTR, vt_a[nq1], tr_a, False, True),
            ('null2', c_null2, NTR[nq2], VT, tr_a[nq2], vt_a, True, False),
            ('main1', c_main1, VT, TR, vt_a, tr_a, False, False),
            ('main2', c_main2, TR, VT, tr_a, vt_a, False, False)][:5 if ALL else 6]
    own = {'planted1': {(k, len(tr) + k) for k in range(len(p1))},
           'planted2': {(k, vt.index(truth[p]['coarse'])) for k, p in enumerate(p2)}}
    for name, cands, FT, CT, fid, cid, fnull, cnull in sets:
        rows_a, seen = stage_a(cands, FT, CT, name, own.get(name, frozenset()))
        if name in own:
            o = own[name]; res[f'{name} own copy kept'] = {'candidates': len(o & set(cands)), 'stage A': len(o & {p[0] for p in rows_a}), 'planted': len(o)}
            log(name, res[f'{name} own copy kept'])
            pl = p1 if name == 'planted1' else p2
            with open(out / f'stageA_{name}.csv', 'w') as fh:
                fh.write('k,true_zoom,inside,voted_zoom,t,votes,r_a,dy,dx,zoom_a,true_dy,true_dx\n')
                for k_ in sorted(o):
                    tt = truth[pl[k_[0]]]
                    if k_ in seen:
                        (zi, t_, vo), rw = seen[k_]
                        fh.write(f"{k_[0]},{tt['zoom']:.3f},{int(tt['inside'])},{ZOOMS[zi]},{t_},{vo},{rw[0]:.4f},{int(rw[1])},{int(rw[2])},{rw[5]:.4f},{tt['dy']},{tt['dx']}\n")
                    else:
                        fh.write(f"{k_[0]},{tt['zoom']:.3f},{int(tt['inside'])},,,,,,,,{tt['dy']},{tt['dx']}\n")
        nb = stage_b(rows_a, fid, cid, fnull, cnull, name, appender(out / f'pairs_{name}.csv', split, fnull, cnull),
                     tag if name.startswith('planted') else None)
        res[name] = {'fine tiles': len(FT), 'candidates': len(cands), 'stage B pairs': nb}
        (out / 'search.json').write_text(json.dumps(res, indent=1))
    res['settings'] = {k: globals()[k] for k in ('ZOOMS', 'QSTRIDE', 'K_PATCH', 'PATCH_R', 'MIN_VOTES', 'K_VOTE', 'A_STEPS',
                                                 'B_STEPS', 'MIN_SHARE', 'STAGE_B_R', 'N_NULL', 'N_PLANT')}
    (out / 'search.json').write_text(json.dumps(res, indent=1)); log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
