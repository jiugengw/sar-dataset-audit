"""M4-SAR copy search on the pixels: exact, flipped, shifted and overlapping copies, between any two images.

Each tile comes in two versions (k: 10 m optical + VH; k + 56,087: 60 m optical + VV; same boxes, same split,
labeltwins.py), so the search runs on the 56,087 version-1 images and every link holds for both versions.

1. Candidates, on the optical image (no speckle), grey, area-averaged:
     whole image  32 px zero-mean unit-norm descriptors under the 8 flips / turns, top K_WHOLE refs per turn, r >= WHOLE_R;
     patch votes  16 px patches of the 64 px thumbnail, slightly blurred (query every 4 px, reference every 2 px, flat
                  patches left out);
                  each query patch's top K_PATCH reference patches with r >= PATCH_R vote for their image; images with
                  >= MIN_VOTES votes (top K_VOTE) are candidates. Finds shifted and partly overlapping tiles.
2. Stage A, 128 px, slightly blurred (sub-pixel shifts): Pearson over the shared region only, for every shift with
   >= MIN_OVL shared area (FFT); best shift.
3. Stage B, full resolution (512 px), for pairs with stage-A r >= STAGE_B_R: the same, within +-6 px of 4x the stage-A
   shift, on the optical image (r_opt); then the SAR images' Pearson over the same shared region at the same shift
   (r_sar). The same acquisition repeats its speckle (r_sar ~ 1); the same place on another date does not.

Controls, in the same job:
   null    val + test queries against a fake train pool that cannot hold their copies: every train image cut into 4 x 4
           blocks, shuffled with a fixed per-image order, and every block turned or flipped (one of the 7 non-identity
           turns), so no two blocks keep their layout (texture and objects survive, the scene layout does not).
           Any pair passing a rule there is a false positive of that rule.
   planted N_PLANT val / test images copied into the pool: re-saved (JPEG) and shifted so that 75 / 50 / 30 / 15% of
           the image is shared (the rest filled from another train image, optical and SAR alike). Recall = planted
           copies found with the right shift (+-2 px) that pass the rule.

    python pixsearch.py M4-SAR.zip OUT_DIR
Resume after a stop: CANDS=<earlier OUT_DIR>/cands_main.npz skips step 1 for main (~1.5 h), and
MAIN_DONE=<earlier OUT_DIR>/pairs_main.csv keeps the main pairs already checked and skips them.
Order: main candidates (saved to cands_main.npz), then the planted and null controls, then main stage A / B; stage B
pairs are appended to the CSVs as they are checked, with progress lines.
Output: OUT_DIR/pairs_main.csv, pairs_null.csv, pairs_planted.csv (every pair that reached stage B), search.json.
The rule (r_opt, r_sar, shared share) is chosen from the controls by summarise.py.
"""
import csv, io, json, os, sys, time, zipfile
from functools import lru_cache
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HALF = 56087
K_WHOLE, WHOLE_R = 10, 0.7
K_PATCH, PATCH_R, MIN_VOTES, K_VOTE = 48, 0.8, 2, 20
QSTRIDE, RSTRIDE = 4, 2
BLUR = 1.0            # Gaussian sigma (64 px thumbnail) before patches: tolerates shifts between grid points
FLAT = 2.0            # 64 px patch std (grey 0-255) below this never votes
MIN_OVL = 0.10        # shared share of the image
STAGE_B_R = 0.8
N_PLANT = 500
PLANT_BASE = 10 ** 6
DEV = 'cuda'
WORKERS = int(os.environ.get('WORKERS', 3))   # + the main process = 4 CPUs (the QOS limit)
REF_CHUNK = int(os.environ.get('REF_CHUNK', 250_000))   # reference patch rows per matrix product (GPU memory)
CHUNK = 4096         # stage B pairs read per step (bounded memory)
T0 = time.time()
ZIP, MEMBERS, PLANTED, NULL_PERM = None, None, {}, False

# ---------------------------------------------------------------- data


def index(zpath):
    m = {}
    for x in zipfile.ZipFile(zpath).namelist():
        p = x.split('/')
        if len(p) == 5 and p[2] == 'images' and '.' in p[4]:
            i = int(p[4].split('.')[0]); d = m.setdefault(i, {}); d[p[1]] = x; d['split'] = p[3]
    return m


_z, _zpid = None, None


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


@lru_cache(maxsize=256)  # stage B is sorted by query, so a worker reads each query image about once
def _read(member):          # one open zip per process (forked workers must not share the parent's file position)
    global _z, _zpid
    if _z is None or _zpid != os.getpid(): _z, _zpid = zipfile.ZipFile(ZIP), os.getpid()
    with Image.open(io.BytesIO(_z.read(member))) as im:
        im = im.convert('L')
        if im.size != (512, 512): im = im.resize((512, 512), Image.BILINEAR)
        return np.asarray(im, np.uint8)


def small(a):            # 512 -> 128, area mean
    return a.reshape(128, 4, 128, 4).mean((1, 3)).astype(np.float16)


def _thumb(i):
    return i, small(_read(MEMBERS[i]['optical']))


def perm_of(j):
    g = np.random.default_rng(j)
    return g.permutation(16), g.integers(1, 8, 16)


def shuffle(a, perm):    # 4 x 4 blocks of a 2-D image (size divisible by 4): reorder, and turn / flip every block
    order, turns = perm; s = a.shape[-1] // 4
    b = [a[(k // 4) * s:(k // 4 + 1) * s, (k % 4) * s:(k % 4 + 1) * s] for k in range(16)]
    b = [turn(b[k], int(t)) for k, t in zip(order, turns)]
    rows = [np.concatenate(b[r * 4:(r + 1) * 4], -1) for r in range(4)]
    return np.concatenate(rows, -2)


def turn(a, t):          # numpy (H, W): flip, then rot90 t % 4 (same as tturn)
    if t >= 4: a = a[:, ::-1]
    return np.ascontiguousarray(np.rot90(a, t % 4))


def tturn(x, t):         # torch (..., H, W)
    if t >= 4: x = x.flip(-1)
    return torch.rot90(x, t % 4, dims=(-2, -1))


def full(j):             # (optical, sar) uint8 512 px of a pool image (real, planted or null-shuffled)
    if j >= PLANT_BASE: return PLANTED[j]
    o, s = _read(MEMBERS[j]['optical']), _read(MEMBERS[j]['sar'])
    if NULL_PERM: p = perm_of(j); o, s = shuffle(o, p), shuffle(s, p)
    return o, s


def _pair(args):
    i, j, t = args
    oi, si = _read(MEMBERS[i]['optical']), _read(MEMBERS[i]['sar'])
    oj, sj = full(j)
    return turn(oi, t), oj, turn(si, t), sj

# ---------------------------------------------------------------- search


def norm_img(x):         # per image zero mean, unit std (B, H, W)
    x = x - x.mean((-2, -1), keepdim=True)
    return x / x.std((-2, -1), keepdim=True).clamp_min(1e-3)


def desc32(t128):
    x = F.avg_pool2d(t128[:, None].float(), 4)[:, 0].reshape(len(t128), -1)
    x = x - x.mean(1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)


_g = torch.exp(-torch.arange(-2, 3, dtype=torch.float32) ** 2 / (2 * BLUR ** 2)); _g = _g / _g.sum()


def blur(x):             # (N, 1, H, W) -> same, Gaussian sigma BLUR
    k = _g.to(x.device)
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode='reflect'), k.view(1, 1, 1, 5))
    return F.conv2d(F.pad(x, (0, 0, 2, 2), mode='reflect'), k.view(1, 1, 5, 1))


def patches(t64, stride):
    x = blur(t64[:, None].float())
    u = F.unfold(x, 16, stride=stride).transpose(1, 2)                        # (N, P, 256)
    u = u - u.mean(2, keepdim=True)
    sd = u.std(2, keepdim=True)
    u = u / u.norm(dim=2, keepdim=True).clamp_min(1e-6)
    return torch.where(sd > FLAT, u, torch.zeros_like(u))


def build_ref(t128, extra=0):
    n = len(t128); g = (64 - 16) // RSTRIDE + 1
    R = torch.zeros(((n + extra) * g * g, 256), dtype=torch.float16, device=DEV)
    D = torch.zeros((n + extra, 1024), dtype=torch.float32, device=DEV)
    for s in range(0, n, 1024):
        x = t128[s:s + 1024].to(DEV)
        R[s * g * g:(s + len(x)) * g * g] = patches(F.avg_pool2d(x[:, None].float(), 2)[:, 0], RSTRIDE).reshape(-1, 256).half()
        D[s:s + len(x)] = desc32(x)
    return R, D


def fill_ref(R, D, row, t128):
    g = (64 - 16) // RSTRIDE + 1; x = t128.to(DEV)
    R[row * g * g:(row + len(x)) * g * g] = patches(F.avg_pool2d(x[:, None].float(), 2)[:, 0], RSTRIDE).reshape(-1, 256).half()
    D[row:row + len(x)] = desc32(x)


def candidates(q128, qid, R, D, rid, nref):
    """q128 (NQ,128,128) cpu f16; qid: pool id of each query (or -1); R, D: ref patches / descriptors (first nref
    images used); rid: pool id of each ref row. Returns {(q index, ref index, t): source}."""
    g = (64 - 16) // RSTRIDE + 1; gq = (64 - 16) // QSTRIDE + 1
    Rv, Dv = R[:nref * g * g], D[:nref]
    rid_t = torch.as_tensor(rid[:nref], device=DEV)
    qpos = torch.arange(gq * gq, device=DEV)
    out = {}; t1 = time.time()
    for s in range(0, len(q128), 8):
        if s and s % 8000 == 0:
            el = time.time() - t1; log(f'candidates: {s} / {len(q128)} queries, ~{el / s * (len(q128) - s) / 60:.0f} min left')
        x = q128[s:s + 8].to(DEV); b = len(x)
        qids = torch.as_tensor(qid[s:s + b], device=DEV)
        # whole image, 8 turns
        for t in range(8):
            sim = desc32(tturn(x, t)) @ Dv.T
            sim[rid_t[None, :] == qids[:, None]] = -1
            v, ix = sim.topk(K_WHOLE, dim=1)
            ixc = ix.cpu()
            for a, r in torch.nonzero(v >= WHOLE_R).tolist():
                out.setdefault((s + a, int(ixc[a, r]), t), 'whole')
        # patch votes, no turn
        Q = patches(F.avg_pool2d(x[:, None].float(), 2)[:, 0], QSTRIDE).reshape(-1, 256).half()
        bv = torch.full((len(Q), K_PATCH), -2.0, device=DEV); bi = torch.zeros((len(Q), K_PATCH), dtype=torch.long, device=DEV)
        for c in range(0, len(Rv), REF_CHUNK):
            sim = Q @ Rv[c:c + REF_CHUNK].T
            v, ix = sim.topk(K_PATCH, dim=1)
            v = torch.cat([bv, v.float()], 1); ix = torch.cat([bi, ix + c], 1)
            bv, o = v.topk(K_PATCH, dim=1); bi = ix.gather(1, o)
        ref = bi // (g * g)
        qa = (torch.arange(len(Q), device=DEV) // (gq * gq))
        ok = (bv >= PATCH_R) & (rid_t[ref] != qids[qa][:, None])
        a, p = torch.nonzero(ok, as_tuple=True)
        if len(a) == 0: continue
        trip = torch.stack([qa[a], a, ref[a, p]], 1).unique(dim=0)                 # (query, query patch, ref) once
        qr, cnt = trip[:, [0, 2]].unique(dim=0, return_counts=True)
        keep = cnt >= MIN_VOTES; qr, cnt = qr[keep].tolist(), cnt[keep].tolist()
        per = {}
        for (qa_, r_), c_ in zip(qr, cnt): per.setdefault(qa_, []).append((c_, r_))
        for qa_, lst in per.items():
            for c_, r_ in sorted(lst, reverse=True)[:K_VOTE]:
                out.setdefault((s + qa_, r_, 0), 'patch')
    return out


def ncc(a, b, min_ovl, centre=None, win=6):
    """Overlap-normalised Pearson for every shift (a pixel u matches b pixel u + s). a, b: (B, H, W) f32.
    Returns r, dy, dx, shared share, each (B,)."""
    B, H, W = a.shape; P = (2 * H, 2 * W)
    a = norm_img(a); b = norm_img(b)
    fo = torch.fft.rfft2(torch.ones((1, H, W), device=a.device), s=P)
    fa, fb = torch.fft.rfft2(a, s=P), torch.fft.rfft2(b, s=P)
    cc = lambda x, y: torch.fft.irfft2(x.conj() * y, s=P)
    n = cc(fo, fo)
    sab, sa, sb = cc(fa, fb), cc(fa, fo), cc(fo, fb)
    saa, sbb = cc(torch.fft.rfft2(a * a, s=P), fo), cc(fo, torch.fft.rfft2(b * b, s=P))
    va, vb = n * saa - sa * sa, n * sbb - sb * sb
    r = (n * sab - sa * sb) / torch.sqrt((va * vb).clamp_min(1e-6))
    bad = (n < min_ovl * H * W - 0.5) | (va < 1e-3 * n * n) | (vb < 1e-3 * n * n)
    r = torch.where(bad, torch.full_like(r, -2.0), r)
    ky = torch.arange(2 * H, device=a.device); ky = torch.where(ky < H, ky, ky - 2 * H)
    kx = torch.arange(2 * W, device=a.device); kx = torch.where(kx < W, kx, kx - 2 * W)
    if centre is not None:
        cy, cx = centre
        m = ((ky[None, :, None] - cy[:, None, None]).abs() > win) | ((kx[None, None, :] - cx[:, None, None]).abs() > win)
        r = torch.where(m, torch.full_like(r, -2.0), r)
    v, ix = r.reshape(B, -1).max(1)
    dy, dx = ky[ix // (2 * W)], kx[ix % (2 * W)]
    ovl = n.reshape(-1)[ix] / (H * W)
    return v, dy, dx, ovl


def pearson_at(a, b, dy, dx):
    H, W = a.shape
    ya, yb = (0, dy) if dy >= 0 else (-dy, 0); xa, xb = (0, dx) if dx >= 0 else (-dx, 0)
    h, w = H - abs(dy), W - abs(dx)
    x, y = a[ya:ya + h, xa:xa + w].float().reshape(-1), b[yb:yb + h, xb:xb + w].float().reshape(-1)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1e-6))


def verify(cands, q128, r128, qid, rid, name, sink=None, dedupe=False, skip=frozenset(), workers=WORKERS):
    """Stage A on all candidates, stage B on those with r >= STAGE_B_R. dedupe: keep one of (a, b) and (b, a) (the
    higher stage-A r). skip: (q id, r id) pairs already checked by an earlier run. sink(rows) gets every checked chunk.
    Returns rows, candidates, stage B pairs."""
    keys = list(cands)
    stage_b = []
    qg, rg = q128.to(DEV), r128.to(DEV)
    for s in range(0, len(keys), 256):
        ks = keys[s:s + 256]
        qi = torch.tensor([k[0] for k in ks], device=DEV); ri = torch.tensor([k[1] for k in ks], device=DEV)
        a = qg[qi].float(); b = rg[ri].float()
        for t in range(1, 8):
            m = [n for n, k in enumerate(ks) if k[2] == t]
            if m: a[m] = tturn(a[m], t)
        v, dy, dx, ov = ncc(blur(a[:, None])[:, 0], blur(b[:, None])[:, 0], MIN_OVL)
        for k, vv, y, x, o in zip(ks, v.tolist(), dy.tolist(), dx.tolist(), ov.tolist()):
            if vv >= STAGE_B_R: stage_b.append((k, vv, y, x, o))
    del qg, rg; torch.cuda.empty_cache()
    n_a = len(stage_b)
    if dedupe:
        best = {}
        for p in stage_b:
            (q, r, t) = p[0]; u = (min(int(qid[q]), int(rid[r])), max(int(qid[q]), int(rid[r])))
            if u not in best or p[1] > best[u][1]: best[u] = p
        stage_b = list(best.values())
    if skip: stage_b = [p for p in stage_b if (int(qid[p[0][0]]), int(rid[p[0][1]])) not in skip]
    stage_b.sort(key=lambda p: (p[0][0], p[0][1]))
    log(f'{name}: stage A done, {n_a} of {len(keys)} pass r >= {STAGE_B_R}; {len(stage_b)} to stage B')
    jobs = [(int(qid[q]), int(rid[r]), t) for (q, r, t), *_ in stage_b]
    rows, t1, last = [], time.time(), 0
    with Pool(workers) as pool:
        nxt = pool.map_async(_pair, jobs[:CHUNK], chunksize=16)
        for c in range(0, len(jobs), CHUNK):
            got_all = nxt.get()
            if c + CHUNK < len(jobs): nxt = pool.map_async(_pair, jobs[c + CHUNK:c + 2 * CHUNK], chunksize=16)
            new = []
            for s in range(0, len(got_all), 16):
                part, got = stage_b[c + s:c + s + 16], got_all[s:s + 16]
                oa = torch.stack([torch.from_numpy(g[0]) for g in got]).to(DEV).float()
                ob = torch.stack([torch.from_numpy(g[1]) for g in got]).to(DEV).float()
                cy = torch.tensor([4 * p[2] for p in part], device=DEV); cx = torch.tensor([4 * p[3] for p in part], device=DEV)
                v, dy, dx, ov = ncc(oa, ob, MIN_OVL, centre=(cy, cx))
                for p, g, vv, y, x, o in zip(part, got, v.tolist(), dy.tolist(), dx.tolist(), ov.tolist()):
                    (q, r, t), rA = p[0], p[1]
                    r_sar = pearson_at(torch.from_numpy(g[2]).to(DEV), torch.from_numpy(g[3]).to(DEV), y, x) if vv > -2 else -2.0
                    new.append({'q': int(qid[q]), 'r': int(rid[r]), 'source': cands[p[0]], 't': t, 'dy': y, 'dx': x,
                                'shared': round(o, 3), 'r128': round(rA, 4), 'r_opt': round(vv, 4), 'r_sar': round(r_sar, 4)})
            del got_all
            rows += new
            if sink: sink(new)
            done = c + CHUNK
            if done - last >= 50_000 or done >= len(jobs):
                last = done; rate = min(done, len(jobs)) / max(time.time() - t1, 1e-6)
                log(f'{name}: stage B {min(done, len(jobs))} / {len(jobs)} ({rate:.0f} pairs/s, '
                    f'~{(len(jobs) - min(done, len(jobs))) / max(rate, 1e-6) / 60:.0f} min left)')
    return rows, len(keys), len(stage_b)


COLS = ['q', 'r', 'q_split', 'r_split', 'source', 't', 'dy', 'dx', 'shared', 'r128', 'r_opt', 'r_sar', 'planted']


def write(rows, path, split, mode='w'):
    with open(path, mode, newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator='\n', extrasaction='ignore')
        if mode == 'w': w.writeheader()
        for x in rows:
            x.setdefault('planted', ''); x['q_split'] = split.get(x['q'], '?'); x['r_split'] = split.get(x['r'], 'planted' if x['r'] >= PLANT_BASE else '?')
            w.writerow(x)


def appender(path, split):   # header now, rows as they come
    write([], path, split)
    return lambda rows: write(rows, path, split, mode='a')


def main(zpath, out):
    global ZIP, MEMBERS, NULL_PERM
    out = Path(out); out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    ZIP = zpath; MEMBERS = index(zpath)
    ids = sorted(i for i in MEMBERS if i <= HALF)
    if os.environ.get('LIMIT'): ids = ids[::max(1, len(ids) // int(os.environ['LIMIT']))]   # a quick trial run
    split = {i: MEMBERS[i]['split'] for i in MEMBERS}
    pos = {i: k for k, i in enumerate(ids)}
    with Pool(4) as pool:
        th = dict(pool.imap_unordered(_thumb, ids, chunksize=64))
    T = torch.from_numpy(np.stack([th[i] for i in ids])); del th
    log('thumbnails', T.shape)
    res = {'version-1 images': len(ids), 'per split': dict(Counter(split[i] for i in ids))}

    # planted copies (built now so their pool rows exist)
    rng = np.random.default_rng(0)
    qv = [i for i in ids if split[i] != 'train' and float(T[pos[i]].float().std()) > 10]
    train = [i for i in ids if split[i] == 'train']
    src = rng.choice(qv, min(N_PLANT, len(qv)), replace=False); kinds = [1.0, 0.75, 0.5, 0.3, 0.15]
    truth, pthumbs = {}, []
    for n, q in enumerate(src):
        f = kinds[n % len(kinds)]; pid = PLANT_BASE + n
        o, s = _read(MEMBERS[int(q)]['optical']), _read(MEMBERS[int(q)]['sar'])
        if f == 1.0:
            buf = io.BytesIO(); Image.fromarray(o).save(buf, 'JPEG', quality=85); o2 = np.asarray(Image.open(buf).convert('L'))
            PLANTED[pid] = (o2, s.copy()); dy = dx = 0
        else:
            mode = n % 3; d = int(round(512 * (1 - (f if mode < 2 else np.sqrt(f)))))
            dy, dx = [(0, d), (d, 0), (d, d)][mode]
            if rng.random() < 0.5: dy = -dy
            if rng.random() < 0.5: dx = -dx
            fid = int(rng.choice(train)); fo, fs = _read(MEMBERS[fid]['optical']), _read(MEMBERS[fid]['sar'])
            po, ps = fo.copy(), fs.copy()
            # planted pixel u + (dy, dx) = query pixel u
            ys, yd = (slice(0, 512 - dy), slice(dy, 512)) if dy >= 0 else (slice(-dy, 512), slice(0, 512 + dy))
            xs, xd = (slice(0, 512 - dx), slice(dx, 512)) if dx >= 0 else (slice(-dx, 512), slice(0, 512 + dx))
            po[yd, xd] = o[ys, xs]; ps[yd, xd] = s[ys, xs]; PLANTED[pid] = (po, ps)
        truth[pid] = (int(q), f, dy, dx); pthumbs.append(small(PLANTED[pid][0]))
    PT = torch.from_numpy(np.stack(pthumbs))
    log('planted', len(PT))

    # main search: every version-1 image against every other (+ planted rows, used by the planted run only)
    R, D = build_ref(T, extra=len(PT)); fill_ref(R, D, len(T), PT)
    rid = np.array(ids + sorted(truth))
    allT = torch.cat([T, PT])
    log('reference built', tuple(R.shape))
    if os.environ.get('CANDS'):
        z = np.load(os.environ['CANDS']); assert z['n_ids'] == len(ids) and z['n_rid'] == len(rid)
        src_names = [str(x) for x in z['names']]
        cands = {(int(a), int(b), int(c)): src_names[d] for a, b, c, d in zip(z['q'], z['r'], z['t'], z['src'])}
        log('main candidates loaded from', os.environ['CANDS'], len(cands))
    else:
        cands = candidates(T, np.array(ids), R, D, rid, len(T))
        k = np.array(list(cands), dtype=np.int64).reshape(-1, 3); names = ['whole', 'patch']
        np.savez(out / 'cands_main.npz', q=k[:, 0], r=k[:, 1], t=k[:, 2], src=np.array([names.index(v) for v in cands.values()]),
                 names=np.array(names), n_ids=len(ids), n_rid=len(rid))
    log('main candidates', len(cands), dict(Counter(cands.values())))
    main_cands = cands

    # planted: the planted copies' queries against the whole pool including the planted copies
    qs = [int(truth[p][0]) for p in sorted(truth)]
    uq = sorted(set(qs)); QT = torch.stack([T[pos[i]] for i in uq])
    cands = candidates(QT, np.array(uq), R, D, rid, len(rid))
    log(f'GPU peak so far {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')
    del R, D; torch.cuda.empty_cache()     # stage A / B do not need the reference (fits a 24 GB GPU)
    rows, n_c, _ = verify(cands, QT, allT, np.array(uq), rid, 'planted')
    rows = [x for x in rows if x['r'] >= PLANT_BASE and truth[x['r']][0] == x['q']]
    for x in rows:
        _, f, dy, dx = truth[x['r']]; x['planted'] = f'{f}:{dy}:{dx}:{int(abs(x["dy"] - dy) <= 2 and abs(x["dx"] - dx) <= 2 and x["t"] == 0)}'
    write(rows, out / 'pairs_planted.csv', split); log('planted stage B pairs', len(rows))
    res['planted'] = {'planted copies': len(truth), 'per shared share': dict(Counter(str(v[1]) for v in truth.values())),
                      'found at stage B (any shift)': len({x['r'] for x in rows})}
    (out / 'planted_truth.json').write_text(json.dumps({str(k): v for k, v in truth.items()}))

    # null: val + test queries against the block-shuffled train images
    NULL_PERM = True
    tr = [i for i in ids if split[i] == 'train']
    NT = torch.from_numpy(np.stack([shuffle(T[pos[i]].numpy(), perm_of(i)) for i in tr]))
    R, D = build_ref(NT)
    q = [i for i in ids if split[i] != 'train']; QT = torch.stack([T[pos[i]] for i in q])
    cands = candidates(QT, np.array(q), R, D, np.array(tr), len(tr))
    del R, D; torch.cuda.empty_cache()
    rows, n_c, n_b = verify(cands, QT, NT, np.array(q), np.array(tr), 'null', sink=appender(out / 'pairs_null.csv', split))
    log('null stage B pairs', len(rows))
    res['null'] = {'queries': len(q), 'shuffled train images': len(tr), 'candidates': n_c, 'stage B pairs': len(rows)}
    NULL_PERM = False
    res['settings'] = {k: globals()[k] for k in ('K_WHOLE', 'WHOLE_R', 'K_PATCH', 'PATCH_R', 'MIN_VOTES', 'K_VOTE',
                                                 'QSTRIDE', 'RSTRIDE', 'BLUR', 'FLAT', 'MIN_OVL', 'STAGE_B_R', 'N_PLANT')}
    (out / 'search.json').write_text(json.dumps(res, indent=1))

    # main stage A / B last (the longest part; pairs written as they are checked)
    sink, skip = appender(out / 'pairs_main.csv', split), set()
    if os.environ.get('MAIN_DONE'):      # pairs_main.csv of an earlier run that stopped part-way: keep its rows, skip them
        with open(os.environ['MAIN_DONE']) as fh: old = [dict(x, q=int(x['q']), r=int(x['r']), planted='') for x in csv.DictReader(fh)]
        skip = {(int(x['q']), int(x['r'])) for x in old}; sink(old); log('main: kept', len(old), 'pairs from', os.environ['MAIN_DONE'])
    rows, n_c, n_b = verify(main_cands, T, allT, np.array(ids), rid, 'main', sink=sink, dedupe=True, skip=skip)
    log('main stage B pairs', len(rows))
    res['main'] = {'candidates': n_c, 'by source': dict(Counter(main_cands.values())),
                   'stage B pairs (one of each (a, b) / (b, a))': n_b}
    (out / 'search.json').write_text(json.dumps(res, indent=1)); log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
