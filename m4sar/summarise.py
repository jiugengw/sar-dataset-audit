"""M4-SAR: link rules from the controls, leak rates, and groups for leak-free folds, from pixsearch.py's output.

A pair (version-1 images, aligned on the optical image) is an OPTICAL link when r_opt passes and a RADAR link when r_sar
passes, over the shared region at the found shift. Thresholds are tiered by the shared share and set from the null
control (a block-shuffled train pool that holds no copies) and checked on the planted copies:
    shared >= 0.25:        r >= R_WIDE
    0.10 <= shared < 0.25: r >= R_THIN     (thin overlaps: the null reaches higher there)
Leak rate of val / test: images with a link to a train image. Radar-only use counts radar links; optical or fusion use
counts optical OR radar links. Image k and k + 56,087 are the same tile (labeltwins.py), so each rate holds for both
versions. Groups (union-find): every link of either kind, the label twins (identical non-empty box lists) and the
k / k + 56,087 pairs; folds deal groups largest first to the emptiest fold.

    python summarise.py SEARCH_DIR LABEL_GROUPS_CSV M4-SAR.zip OUT_DIR
Output: OUT_DIR/summary.json, links.csv, folds.csv (id, fold, split; all 112,174 ids), examples_*.png.
"""
import csv, io, json, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HALF = 56087
R_WIDE, R_THIN, WIDE = 0.97, 0.99, 0.25
SHARES = [0.15, 0.3, 0.5, 0.75, 1.0]


def passes(df, col):
    return ((df.shared >= WIDE) & (df[col] >= R_WIDE)) | ((df.shared >= 0.1) & (df.shared < WIDE) & (df[col] >= R_THIN))


class UF:
    def __init__(self): self.p = {}
    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a: self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def sheet(rows, zpath, title, path, n=8):
    z = zipfile.ZipFile(zpath); names = {}
    for x in z.namelist():
        p = x.split('/')
        if len(p) == 5 and p[2] == 'images' and '.' in p[4]: names[(p[1], int(p[4].split('.')[0]))] = x
    rd = lambda m, i: Image.open(io.BytesIO(z.read(names[(m, i)]))).convert('L').resize((192, 192))
    rows = rows.head(n); W = Image.new('L', (4 * 196, 22 + len(rows) * 214), 255); d = ImageDraw.Draw(W)
    d.text((4, 4), title + '   (query optical | train optical | query SAR | train SAR)', fill=0)
    for k, (_, x) in enumerate(rows.iterrows()):
        y = 22 + k * 214
        for c, (m, i) in enumerate([('optical', x.q), ('optical', x.r), ('sar', x.q), ('sar', x.r)]):
            W.paste(rd(m, int(i)), (c * 196, y))
        d.text((2, y + 194), f'{x.q}({x.q_split}) vs {x.r}({x.r_split})  shared {x.shared:.2f}  r_opt {x.r_opt:.3f}  r_sar {x.r_sar:.3f}  shift {x.dy},{x.dx} t{x.t}', fill=0)
    W.save(path)


def main(search, groups_csv, zpath, out):
    search, out = Path(search), Path(out); out.mkdir(parents=True, exist_ok=True); res = {}
    null = pd.read_csv(search / 'pairs_null.csv'); pl = pd.read_csv(search / 'pairs_planted.csv'); m = pd.read_csv(search / 'pairs_main.csv')
    truth = json.loads((search / 'planted_truth.json').read_text())
    nq = json.loads((search / 'search.json').read_text())['null']['queries']

    # ---- controls
    res['rule'] = {'shared >= %.2f' % WIDE: f'r >= {R_WIDE}', '0.10 <= shared < %.2f' % WIDE: f'r >= {R_THIN}'}
    nl = {}
    for col in ('r_opt', 'r_sar'):
        ok = null[passes(null, col)]
        nl[col] = {'null pairs passing': len(ok), 'null queries with a false link': int(ok.q.nunique()),
                   'false link rate (% of val / test queries)': round(100 * ok.q.nunique() / nq, 3),
                   'by shared': ok.shared.round(2).value_counts().sort_index().astype(int).to_dict()}
    either = null[passes(null, 'r_opt') | passes(null, 'r_sar')]
    nl['either'] = {'null queries with a false link': int(either.q.nunique()), 'rate %': round(100 * either.q.nunique() / nq, 3)}
    res['null control'] = nl
    pl['f'] = pl.planted.str.split(':').str[0].astype(float); pl['ok'] = pl.planted.str.split(':').str[3].astype(int)
    rec = {}
    for col in ('r_opt', 'r_sar'):
        g = pl[(pl.ok == 1) & passes(pl, col)]
        rec[col] = {str(f): int(g[g.f == f].r.nunique()) for f in SHARES}
    rec['planted per share'] = {str(f): int(sum(1 for v in truth.values() if v[1] == f)) for f in SHARES}
    rec['reached stage B with the right shift'] = {str(f): int(pl[(pl.ok == 1) & (pl.f == f)].r.nunique()) for f in SHARES}
    res['planted recall'] = rec

    # ---- links
    m = m[m.r_opt > -2].copy()
    m['opt'] = passes(m, 'r_opt'); m['sar'] = passes(m, 'r_sar')
    L = m[m.opt | m.sar].copy()
    L['kind'] = np.where(L.opt & L.sar, 'both', np.where(L.sar, 'radar only', 'optical only'))
    L.to_csv(out / 'links.csv', index=False)
    res['links'] = {'pairs checked': len(m), 'links': len(L), 'by kind': L.kind.value_counts().to_dict(),
                    'by kind, pairs touching train and val/test': L[(L.q_split == 'train') != (L.r_split == 'train')].kind.value_counts().to_dict(),
                    'by shared (links)': L.shared.round(1).value_counts().sort_index().astype(int).to_dict()}

    # ---- leak rates (version-1 ids; hold for k + 56,087)
    split = {}
    for x in m.itertuples(): split[x.q] = x.q_split; split[x.r] = x.r_split
    with open(groups_csv) as fh: G = list(csv.DictReader(fh))
    for g in G:
        for s in g['members'].split(';'): a, b = s.split('/'); split[int(b)] = a
    cross = L[(L.q_split == 'train') != (L.r_split == 'train')]
    vt = lambda row: row.r if row.q_split == 'train' else row.q
    leaked = {'radar': set(), 'optical': set(), 'either': set()}
    for x in cross.itertuples():
        i = vt(x)
        if x.sar: leaked['radar'].add(i)
        if x.opt: leaked['optical'].add(i)
        leaked['either'].add(i)
    twin = set()
    for g in G:
        if int(g['n_boxes']) == 0: continue
        mem = [s.split('/') for s in g['members'].split(';')]
        if any(a == 'train' for a, _ in mem): twin |= {int(b) - HALF if int(b) > HALF else int(b) for a, b in mem if a != 'train'}
    n_split = Counter(split[i] for i in range(1, HALF + 1) if i in split)
    rates = {}
    for sp in ('val', 'test'):
        ids = {i for i in range(1, HALF + 1) if split.get(i) == sp}
        f = lambda s: {'images (per version)': len(s & ids), '%': round(100 * len(s & ids) / len(ids), 1)}
        rates[sp] = {'images per version': len(ids), 'radar-only use (radar link)': f(leaked['radar']),
                     'optical or fusion use (either link)': f(leaked['either']), 'optical link': f(leaked['optical']),
                     'label twins (identical boxes)': f(twin), 'radar link or label twin': f(leaked['radar'] | twin),
                     'either link or label twin': f(leaked['either'] | twin)}
    res['leak rate'] = rates

    # ---- one-sided examples
    one = cross[cross.opt != cross.sar]
    sheet(one[one.sar].sort_values('r_opt'), zpath, 'radar link, optical below the rule', out / 'examples_radar_only.png')
    sheet(one[one.opt].sort_values('r_sar'), zpath, 'optical link, radar below the rule', out / 'examples_optical_only.png')
    sheet(cross[cross.opt & cross.sar].sample(8, random_state=0), zpath, 'both links (random)', out / 'examples_both.png')
    res['one-sided, val/test to train'] = {
        'radar only: r_opt quantiles (5/25/50%)': one[one.sar].r_opt.quantile([.05, .25, .5]).round(3).tolist(),
        'optical only: r_sar quantiles (5/25/50%)': one[one.opt].r_sar.quantile([.05, .25, .5]).round(3).tolist()}

    # ---- groups and folds over all 112,174 ids
    uf = UF(); allids = sorted(set(split) | {i + HALF for i in split if i <= HALF})
    for i in allids: uf.find(i)
    for i in range(1, HALF + 1):
        if i in split: uf.union(i, i + HALF)
    for x in L.itertuples(): uf.union(int(x.q), int(x.r))
    for g in G:
        if int(g['n_boxes']) == 0: continue
        mem = [int(s.split('/')[1]) for s in g['members'].split(';')]
        for b in mem[1:]: uf.union(mem[0], b)
    groups = defaultdict(list)
    for i in allids: groups[uf.find(i)].append(i)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    res['groups'] = {'images': len(allids), 'groups': len(groups), 'largest': sizes[:10],
                     'images in groups of >= 100': sum(s for s in sizes if s >= 100)}
    load = [0] * 5; fold = {}
    for g in sorted(groups.values(), key=len, reverse=True):
        k = int(np.argmin(load)); load[k] += len(g)
        for i in g: fold[i] = k
    sp_full = lambda i: split[i if i <= HALF else i - HALF]
    with open(out / 'folds.csv', 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n'); w.writerow(['id', 'fold', 'split'])
        for i in allids: w.writerow([i, fold[i], sp_full(i)])
    bad = sum(fold[int(x.q)] != fold[int(x.r)] for x in L.itertuples())
    res['folds'] = {'images per fold': load, 'links crossing folds (must be 0)': int(bad),
                    'official split per fold': {k: dict(Counter(sp_full(i) for i in allids if fold[i] == k)) for k in range(5)}}
    (out / 'summary.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
