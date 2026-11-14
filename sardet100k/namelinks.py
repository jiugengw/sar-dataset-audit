"""Link tiles by their original file names (mapping.json), then check the pixels where the names say they overlap.

Names that give a position:
  ShipDataset   Sen_ship_hh_0 + date (8) + scene (Sen 2 digits, Gao 4) + grid row + grid column. Tiles are 256 px,
                one grid step is 128 px, so tiles one step apart (also diagonally) share 50% / 25%. Row and column are
                zero-padded to the same width when the rest has an even length; with an odd length both splits are
                tried and the pixels decide.
  HRSID         P0005_4200_5000_6600_7400: scene and the 800 px window in it (both axis orders are tried).
  tiled names   SARDet-100K's own tiles of one original (SAR-AIRcraft, AIR-SARShip, part of MSAR): same original.
Also ShipDataset tiles at the same scene and grid position in another polarization (hh / hv / vh / vv): the same
place at the same moment, so the same ships, with different speckle.

Every name pair gets a pixel check at the offset the names give: Pearson r over the shared part, at full resolution
(r_full) and 8x averaged (r_low), the best over the axis orders and signs. 'confirmed' needs r_full >= 0.95; a shared
part with almost no texture (std < 1) is 'flat'. Polarization pairs are compared whole (r_low), not confirmed.

    ~/.venvs/sar-audit/bin/python sardet100k/namelinks.py   # -> results/name_links.csv, results/name_links.json
"""
import csv, gzip, json, re, sys
from collections import Counter, defaultdict
from itertools import product
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import groups as G
import provenance as P
import verify as V

D = HERE / 'results'
SHIP_STEP = 128


def ship_parses(stem):
    """[(scene, pol, row, col)] for a ShipDataset stem like 'Sen_ship_hh_02017112701029044'."""
    parts = stem.split('_')
    if len(parts) != 4 or not parts[3].isdigit(): return []  # 969 'newship...' / 'ship...' names: no scene or grid
    sat, _, pol, d = parts
    k = 11 if sat == 'Sen' else 13
    scene, rest = f'{sat}_{d[1:k]}', d[k:]
    n = len(rest)
    cuts = [n // 2] if n % 2 == 0 else [n // 2, n // 2 + 1]
    return [(scene, pol, int(rest[:c]), int(rest[c:])) for c in cuts if 0 < c < n]


def overlap_r(a, b, oy, ox):
    """a's pixel (i + oy, j + ox) sits on b's pixel (i, j). -> (r_full, r_low, share, std)."""
    H, W = b.shape
    if a.shape != b.shape: return float('nan'), float('nan'), 0.0, 0.0
    y0, y1, x0, x1 = max(0, -oy), min(H, H - oy), max(0, -ox), min(W, W - ox)
    if y1 - y0 < 8 or x1 - x0 < 8: return float('nan'), float('nan'), 0.0, 0.0
    pb = b[y0:y1, x0:x1]; pa = a[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
    share = pb.size / b.size; std = float(min(pa.std(), pb.std()))
    if std < 1e-3: return float('nan'), float('nan'), share, std
    full = float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1])
    h, w = (pa.shape[0] // 8) * 8, (pa.shape[1] // 8) * 8
    if h >= 16 and w >= 16:
        la = pa[:h, :w].reshape(h // 8, 8, w // 8, 8).mean((1, 3)); lb = pb[:h, :w].reshape(h // 8, 8, w // 8, 8).mean((1, 3))
        low = float(np.corrcoef(la.ravel(), lb.ravel())[0, 1]) if la.std() > 1e-3 and lb.std() > 1e-3 else float('nan')
    else: low = float('nan')
    return full, low, share, std


def check(job):
    kind, a_name, b_name, sa, sb, offsets = job
    try:
        a, b = V.grey(sa, a_name), V.grey(sb, b_name)
        best = (float('nan'), float('nan'), 0.0, 0.0, None)
        for oy, ox in offsets:
            full, low, share, std = overlap_r(a, b, oy, ox)
            if best[4] is None or (full == full and not (best[0] >= full)): best = (full, low, share, std, (oy, ox))
        full, low, share, std, off = best
        if kind == 'other polarization': verdict = 'compared'
        elif full == full and full >= 0.95: verdict = 'confirmed'
        elif std < 1: verdict = 'flat'
        else: verdict = 'not matching'
        return kind, a_name, b_name, sa, sb, off, share, full, low, std, verdict
    except Exception as e:  # noqa: BLE001
        return kind, a_name, b_name, sa, sb, None, 0, float('nan'), float('nan'), 0, f'error {e}'


def name_pairs(prov, split):
    jobs = []
    ship = defaultdict(set)  # (scene, pol, row, col) -> files
    hrsid = defaultdict(list)
    orig = defaultdict(list)
    for f, v in prov.items():
        stem = v['orig_name'].split('_', 1)[1].rsplit('.', 1)[0]
        if v['source'] == 'Shipdataset':
            for key in ship_parses(stem): ship[key].add(f)
        elif v['source'] == 'HRSID':
            m = re.match(r'(P\d+)_(\d+)_(\d+)_(\d+)_(\d+)$', stem)
            if m: hrsid[m.group(1)].append((f, *map(int, m.groups()[1:])))
        if '__' in v['orig_name']: orig[v['original']].append(f)
    seen = set()
    def add(kind, a, b, offsets):
        k = (kind,) + tuple(sorted((a, b)))
        if a != b and k not in seen:
            seen.add(k); jobs.append((kind, a, b, split[a], split[b], offsets))
    for (scene, pol, r, c), files in ship.items():
        for a in files:
            for dr, dc in product((-1, 0, 1), repeat=2):
                if (dr, dc) == (0, 0): continue
                for b in ship.get((scene, pol, r + dr, c + dc), ()):
                    s = SHIP_STEP
                    add('ShipDataset grid neighbour', a, b,
                        [(s * dr, s * dc), (-s * dr, -s * dc), (s * dc, s * dr), (-s * dc, -s * dr)])
            for p2 in ('hh', 'hv', 'vh', 'vv'):
                if p2 != pol:
                    for b in ship.get((scene, p2, r, c), ()): add('other polarization', a, b, [(0, 0)])
    for scene, tiles in hrsid.items():
        for i, (a, p, q, u, w) in enumerate(tiles):
            for b, p2, q2, u2, w2 in tiles[i + 1:]:
                if min(q, q2) > max(p, p2) and min(w, w2) > max(u, u2):  # windows overlap
                    d1, d2 = p2 - p, u2 - u
                    add('HRSID window overlap', a, b, [(d1, d2), (-d1, -d2), (d2, d1), (-d2, -d1)])
    for o, files in orig.items():
        for a in files[1:]:
            add('same original (SARDet tiles)', files[0], a, [])
    return jobs


def gpu_links():
    out = set()
    conf = lambda r: G.confirmed_first(r) if r['kind'] == 'whole' else G.confirmed_shift(r)
    for q, rr in [('val', 'train'), ('test', 'train'), ('val', 'test'), ('val', 'val'), ('test', 'test'), ('train', 'train')]:
        p = D / f'gp_{q}_vs_{rr}.csv'
        fh = p.open() if p.exists() else gzip.open(str(p) + '.gz', 'rt')
        for r in csv.DictReader(fh):
            if r['query'] != r['ref'] and conf(r): out.add(frozenset((r['query'], r['ref'])))
    return out


def main():
    prov = P.load()
    split = {r['file_name']: r['official_split'] for r in csv.DictReader((D / 'resplit_split.csv').open())}
    jobs = name_pairs(prov, split)
    print(Counter(j[0] for j in jobs), flush=True)
    pix = [j for j in jobs if j[5]]
    with Pool() as p:
        res = list(p.imap_unordered(check, pix, chunksize=64))
    res += [(j[0], j[1], j[2], j[3], j[4], None, '', '', '', '', 'by name') for j in jobs if not j[5]]
    cols = ['kind', 'a', 'b', 'split_a', 'split_b', 'offset', 'share', 'r_full', 'r_low', 'std', 'verdict']
    f = lambda v: f'{v:.4f}' if isinstance(v, float) else v
    with (D / 'name_links.csv').open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in res: w.writerow([f(v) for v in r])
    gl = gpu_links()
    summ = defaultdict(lambda: defaultdict(int))
    for kind, a, b, sa, sb, off, share, full, low, std, v in res:
        s = summ[kind]; s['pairs'] += 1; s[v] += 1
        if v in ('confirmed', 'by name', 'flat'):
            cross = sa != sb
            known = frozenset((a, b)) in gl
            s[f'{v}: crosses official splits'] += cross
            s[f'{v}: new (not a GPU-search link)'] += not known
            s[f'{v}: new and crosses splits'] += cross and not known
    if 'other polarization' in summ:
        lows = np.array([r[8] for r in res if r[0] == 'other polarization' and r[8] == r[8]])
        summ['other polarization']['r_low quantiles 10/50/90'] = [round(float(x), 3) for x in np.quantile(lows, [0.1, 0.5, 0.9])]
        summ['other polarization']['crosses official splits'] = sum(r[3] != r[4] for r in res if r[0] == 'other polarization')
    out = {k: dict(v) for k, v in summ.items()}
    (D / 'name_links.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
