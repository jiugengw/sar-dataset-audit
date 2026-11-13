"""Small images inside larger ones (GPU): a 256 px chip cut from a 512 / 800 / 1000 px image, maybe rescaled.

Every other search compares whole images (or one image shifted against another of the same size). A chip cut out
of a larger image, at the same scale or resized, is never found that way. Here every square "chip" (side <= 512) is
searched inside every larger square "host" (side >= 512), under the 8 flips / turns and the SCALES k (the host
window that became the chip is k x the chip's side: k < 1 means the chip was enlarged, k > 1 shrunk).

1. Coarse (GPU): per scale, each host is area-shrunk so that the k x chip-side window becomes 32 px; every 32 x 32
   patch of it (stride 1) is compared with every chip's 32 px thumbnail by normalised correlation (one GEMM).
   Kept per chip: the TOP best (host, scale, flip, place) over all hosts and scales, if r >= MIN_COARSE.
2. Fine: the chip is matched at full resolution in a margin around the coarse window (the host region resized to
   the chip's scale; normalised correlation at every offset), then at the best offset full_r, low_r (8x averaged)
   and bg_r (bright targets masked, as background.bg_r).
Rules as for the whole-image search: full_r >= 0.95 'contained copy', low_r >= 0.95 'contained re-processed';
confirmed when bg_r >= 0.5.

--null cuts every host into 4 x 4 blocks and shuffles them (fixed per-image order), as shift_null.py does: no chip
truly sits inside such a host, so whatever passes is a false positive of search + rules. --chips N searches a fixed
random N chips only (for the null).

    python contained.py IMAGES_DIR resplit_split.csv OUT.csv [--null] [--chips N] [--hosts N]
IMAGES_DIR/{train,val,test}/ holds the images by official split; resplit_split.csv gives each file's official
split and source. --hosts N keeps a fixed random N hosts (tests only).
"""
import csv, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

SCALES = (1.0, 0.75, 1.25, 1.5, 2.0, 0.5)  # k = 1 first: the most likely, and the run's first numbers
D = 32; TOP = 10; MIN_STD = 1.0; MIN_COARSE = 0.5; CHUNK = 8192; MAX_ELEMS = 6e8
TRANSFORMS = ['id', 'rot90', 'rot180', 'rot270', 'lr', 'ud', 'transpose', 'antitranspose']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HALF = torch.float16 if dev.type == 'cuda' else torch.float32
FOLDER = None


def tf(a, t):
    """Flip / turn the last two axes (same names and meaning as verify.F)."""
    if t == 'id': return a
    if t == 'rot90': return torch.rot90(a, 1, (-2, -1))
    if t == 'rot180': return torch.rot90(a, 2, (-2, -1))
    if t == 'rot270': return torch.rot90(a, 3, (-2, -1))
    if t == 'lr': return a.flip(-1)
    if t == 'ud': return a.flip(-2)
    if t == 'transpose': return a.transpose(-2, -1)
    return torch.rot90(a, 2, (-2, -1)).transpose(-2, -1)


def grey(path):
    return np.asarray(Image.open(Path(FOLDER) / path).convert('L'))  # uint8 (path: '<official split>/<file>')


def shuffle_blocks(a, name, grid=4):
    h, w = a.shape; bh, bw = h // grid, w // grid
    perm = np.random.default_rng(zlib.crc32(name.encode())).permutation(grid * grid)
    out = a.copy()
    for dst, src in enumerate(perm):
        out[dst // grid * bh:(dst // grid + 1) * bh, dst % grid * bw:(dst % grid + 1) * bw] = \
            a[src // grid * bh:(src // grid + 1) * bh, src % grid * bw:(src % grid + 1) * bw]
    return out


def load(args):
    name, path, null, role = args
    a = grey(path)
    if a.ndim != 2 or a.shape[0] != a.shape[1]: return name, None
    if role == 'chip':  # chips: 32 px thumbnail only (full image re-read for the fine step)
        t = Fn.adaptive_avg_pool2d(torch.from_numpy(a.astype(np.float32))[None, None], D)[0, 0].numpy()
        return name, (a.shape[0], t)
    return name, (shuffle_blocks(a, name) if null else a)


def unit(x):
    x = x - x.mean(1, keepdim=True)
    return x / (x.norm(dim=1, keepdim=True) + 1e-6)


def coarse(cside, cnames, cthumbs, hosts):
    """-> {chip name: [(r, host name, k, transform, y, x, window)]} (host pixels)."""
    n = len(cnames)
    th = torch.from_numpy(np.stack(cthumbs)).to(dev)
    ok = th.flatten(1).std(1) >= MIN_STD
    Q = torch.cat([unit(tf(th, t).reshape(n, -1)) * ok[:, None] for t in TRANSFORMS]).to(HALF)  # (8n, 1024)
    chip_pos = {c: i for i, c in enumerate(cnames)}
    top = torch.full((n, TOP), -2.0, device=dev)
    topi = torch.zeros((n, TOP, 6), dtype=torch.long, device=dev)  # host side, host index, scale, transform, py, px
    sides = sorted(hosts)
    for si, hside in enumerate(sides):
        hn, ha = hosts[hside]
        for ki, k in enumerate(SCALES):
            win = k * cside
            if win > hside or win == hside: continue  # a window as large as the host is a whole-image match
            gs = int(round(hside * D / win)); ny = gs - D + 1
            B = min(128, max(1, int(MAX_ELEMS // (ny * ny * CHUNK))))
            for b0 in range(0, len(hn), B):
                hb = torch.from_numpy(np.stack(ha[b0:b0 + B])).to(dev).float()[:, None]
                small = Fn.interpolate(hb, size=(gs, gs), mode='area')[:, 0]
                P = small.unfold(1, D, 1).unfold(2, D, 1).reshape(-1, D * D)  # (b * ny * ny, 1024)
                b = hb.shape[0]
                pstd = P.std(1); P = (unit(P) * (pstd >= MIN_STD)[:, None]).to(HALF)
                m = torch.empty((b, Q.shape[0]), device=dev); w = torch.empty((b, Q.shape[0]), dtype=torch.long, device=dev)
                for c0 in range(0, Q.shape[0], CHUNK):
                    s = (P @ Q[c0:c0 + CHUNK].T).view(b, ny * ny, -1)
                    mm, ww = s.max(1); m[:, c0:c0 + CHUNK] = mm.float(); w[:, c0:c0 + CHUNK] = ww
                for j, h in enumerate(hn[b0:b0 + b]):  # never match an image with itself (512 chip in 512 host)
                    if h in chip_pos: m[j, chip_pos[h]::n] = -2
                m = m.view(b, len(TRANSFORMS), n); w = w.view(b, len(TRANSFORMS), n)
                mv, mt = m.max(1)  # best flip per (host, chip): (b, n)
                wp = w.gather(1, mt[:, None])[:, 0]
                idx = torch.stack([torch.full_like(mt, si), torch.arange(b0, b0 + b, device=dev)[:, None].expand(-1, n),
                                   torch.full_like(mt, ki), mt, wp // ny, wp % ny], -1)  # (b, n, 6)
                allv = torch.cat([top, mv.T], 1); alli = torch.cat([topi, idx.transpose(0, 1)], 1)
                o = allv.topk(TOP, 1).indices
                top = allv.gather(1, o); topi = alli.gather(1, o[..., None].expand(-1, -1, 6))
            print(f'  chips {cside} in hosts {hside} at k {k}: {time.strftime("%H:%M:%S")}', flush=True)
    out = {}
    for c, (vals, rows) in enumerate(zip(top.tolist(), topi.tolist())):
        for r, (si, hi, ki, t, py, px) in zip(vals, rows):
            if r < MIN_COARSE: continue
            hside = sides[si]; k = SCALES[ki]; f = hside / round(hside * D / (k * cside))
            out.setdefault(cnames[c], []).append((r, hosts[hside][0][hi], k, TRANSFORMS[t], py * f, px * f, k * cside))
    return out


def fine(chip, host, k, t, y, x, win):
    """Full-resolution match around (y, x) -> (full_r, low_r, bg_r, y, x) or None."""
    a = tf(torch.from_numpy(chip.astype(np.float32)).to(dev), t); cside = a.shape[0]; hside = host.shape[0]
    m = int(np.ceil(win / D)) + 2  # coarse step + slack
    y0, x0 = max(0, int(y) - m), max(0, int(x) - m)
    y1, x1 = min(hside, int(np.ceil(y + win)) + m), min(hside, int(np.ceil(x + win)) + m)
    sh, sw = int(round((y1 - y0) / k)), int(round((x1 - x0) / k))
    if sh < cside or sw < cside: return None
    reg = torch.from_numpy(host[y0:y1, x0:x1].astype(np.float32)).to(dev)[None, None]
    reg = Fn.interpolate(reg, size=(sh, sw), mode='area' if k > 1 else 'bilinear', **({} if k > 1 else {'align_corners': False}))
    az = a - a.mean()
    num = Fn.conv2d(reg, az[None, None])[0, 0]
    ones = torch.ones((1, 1, cside, cside), device=dev)
    s1 = Fn.conv2d(reg, ones)[0, 0]; s2 = Fn.conv2d(reg ** 2, ones)[0, 0]
    var = (s2 - s1 ** 2 / cside ** 2).clamp_min(1e-6)
    ncc = num / (az.norm() * var.sqrt() + 1e-6)
    j = int(ncc.argmax()); oy, ox = divmod(j, ncc.shape[1])
    b = reg[0, 0, oy:oy + cside, ox:ox + cside]
    pa, pb = Fn.avg_pool2d(a[None, None], 8).flatten(), Fn.avg_pool2d(b[None, None], 8).flatten()
    lr = float(torch.corrcoef(torch.stack([pa, pb]))[0, 1])
    an, bn = a.cpu().numpy(), b.cpu().numpy()
    mask = (an > np.quantile(an, 0.98)) | (bn > np.quantile(bn, 0.98))
    keep = Fn.max_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None], 9, 1, 4)[0, 0].numpy() == 0
    if keep.sum() < 0.2 * keep.size or an[keep].std() < 1e-3 or bn[keep].std() < 1e-3:
        bg = float('nan')
    else:
        bg = float(np.corrcoef(an[keep], bn[keep])[0, 1])
    return float(ncc.max()), lr, bg, y0 + oy * k, x0 + ox * k


def init(folder):
    global FOLDER; FOLDER = folder


def main(folder, split_list, out, null=False, nchips=None, nhosts=None):
    init(folder); t0 = time.time()
    rows = list(csv.DictReader(open(split_list))); info = {r['file_name']: r for r in rows}
    path = {n: f"{r['official_split']}/{n}" for n, r in info.items()}
    names = sorted(info)
    with Pool(None, init, (folder,)) as p:
        sizes = dict(p.imap_unordered(_size, [(n, path[n]) for n in names], chunksize=256))
        chip_names = [n for n in names if sizes[n] and sizes[n] <= 512]
        host_names = [n for n in names if sizes[n] and sizes[n] >= 512]
        rng = np.random.default_rng(0)
        if nchips: chip_names = sorted(rng.permutation(chip_names)[:nchips].tolist())
        if nhosts: host_names = sorted(rng.permutation(host_names)[:nhosts].tolist())
        chips = dict(p.imap_unordered(load, [(n, path[n], False, 'chip') for n in chip_names], chunksize=256))
        hosts = {}
        for n, a in p.imap_unordered(load, [(n, path[n], null, 'host') for n in host_names], chunksize=64):
            hosts.setdefault(a.shape[0], ([], [])); hosts[a.shape[0]][0].append(n); hosts[a.shape[0]][1].append(a)
    print(len(chips), 'chips,', {s: len(v[0]) for s, v in hosts.items()}, 'hosts', f'{time.time() - t0:.0f}s', flush=True)
    best = {}
    for cside in sorted({c[0] for c in chips.values()}):
        cn = [n for n, c in chips.items() if c[0] == cside]
        print(f'chips of side {cside}: {len(cn)}', flush=True)
        best.update(coarse(cside, cn, [chips[n][1] for n in cn], hosts))
    print(len(best), 'chips with coarse candidates', f'{time.time() - t0:.0f}s', flush=True)
    himg = {n: a for hn, ha in hosts.values() for n, a in zip(hn, ha)}
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['chip', 'chip_split', 'chip_source', 'host', 'host_split', 'host_source', 'transform', 'scale',
                    'y', 'x', 'window', 'coarse_r', 'full_r', 'low_r', 'bg_r'])
        for i, (c, cands) in enumerate(best.items()):
            a = grey(path[c])
            for r, h, k, t, y, x, win in sorted(cands, reverse=True):
                res = fine(a, himg[h], k, t, y, x, win)
                if res is None: continue
                fr, lr, bg, fy, fx = res
                w.writerow([c, info[c]['official_split'], info[c]['source'], h, info[h]['official_split'],
                            info[h]['source'], t, k, f'{fy:.0f}', f'{fx:.0f}', f'{win:.0f}', f'{r:.4f}', f'{fr:.4f}',
                            f'{lr:.4f}', f'{bg:.4f}'])
            if i % 5000 == 0: print(i, 'of', len(best), 'chips verified', f'{time.time() - t0:.0f}s', flush=True)
    print('->', out, f'{time.time() - t0:.0f}s', flush=True)


def _size(args):
    name, path = args
    with Image.open(Path(FOLDER) / path) as im:
        return name, im.size[0] if im.size[0] == im.size[1] else None


if __name__ == '__main__':
    a = sys.argv[1:]
    opt = lambda k: int(a[a.index(k) + 1]) if k in a else None
    main(a[0], a[1], a[2], '--null' in a, opt('--chips'), opt('--hosts'))
