"""Leak-free re-split of SARDet-100K: group images that share pixels, then split whole groups.

Links (only proven ones join images):
  gp_{q}_vs_{r}.csv(.gz) (gpupairs.py)  every match for all split pairs; confirmed by the background rule
                                         (groups.confirmed_first / confirmed_shift; the ship-pixel rule is not used:
                                         the shuffled control showed it confirms almost as many false pairs as real)
  gp_msar_numbers.csv                    MSAR files with nearby numbers (gpupairs.py --pairs), same rule
  name_links.csv.gz (namelinks.py)       tiles the file names place next to each other, with matching pixels where
                                         they overlap ('confirmed'), or a blank shared part ('flat': kept together,
                                         not counted as a leak); SARDet's own tiles of one original; ShipDataset tiles
                                         in another polarization with every box matching (polcheck.py)
  contained.csv.gz (contained.py)        chips inside larger images, full_r or low_r >= 0.95 and bg_r >= 0.5
  all_*.csv, verify_*/shifts_*_bg.csv    the earlier CPU searches, same background rule
Matches that pass a pixel rule but not the background check are counted as uncertain and do not join.

Groups are union-find components, with the source of most of their images. Per source, val and test get the
official number of that source's images, filled with whole groups:
  - a group goes to val (test) only if it holds at most CAP_SHARE of that split's target; bigger groups (the two
    largest SAR-AIRcraft airports) stay in train;
  - the remaining groups are binned by size (BANDS) and each band gives val and test the same share of its images,
    so val and test get small, medium and large groups in the same mix as train (no size bias).

    ~/.venvs/sar-audit/bin/python sardet100k/resplit.py      # -> results/resplit.json, results/resplit_split.csv
    ~/.venvs/sar-audit/bin/python sardet100k/resplit.py check    # confirmed links that cross the saved split
"""
import csv, gzip, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import assign as A
import groups as G

D = HERE / 'results'; ROOT = HERE / 'data' / 'SARDet_100K'
CAP_SHARE = 0.5
BANDS = [(1, 1), (2, 5), (6, 20), (21, 100), (101, 500), (501, 10 ** 9)]
SEED = 0


class UF:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[max(a, b)] = min(a, b)


def true_source():
    m = json.loads((ROOT / 'mapping.json').read_text())  # original name -> SARDet file name
    name = lambda k: 'AIR-SARShip' if k.lstrip('0123456789').startswith('AIR_SARShip') else k.lstrip('0123456789').split('_')[0]
    return {v: name(k) for k, v in m.items()}  # '6SAR-AIRcraft_0001872...' -> 'SAR-AIRcraft', '10SIVED_...' -> 'SIVED'


def images():
    out = {}
    for split in ('train', 'val', 'test'):
        p = ROOT / 'Annotations' / f'{split}.json'; p = p if p.exists() else ROOT / f'{split}.json'
        d = json.loads(p.read_text())
        out.update({(split, im['file_name']): im for im in d['images']})
    return out


LINKS = [  # file, query split, ref split, kind
    ('verify_val_bg.csv', 'val', 'train', 'first'), ('shifts_val_bg.csv', 'val', 'train', 'shift'),
    ('verify_test_bg.csv', 'test', 'train', 'first'), ('shifts_test_bg.csv', 'test', 'train', 'shift'),
    ('verify_train_vs_train_bg.csv', 'train', 'train', 'first'), ('shifts_train_bg.csv', 'train', 'train', 'shift'),
    ('verify_val_vs_val_bg.csv', 'val', 'val', 'first'), ('verify_test_vs_test_bg.csv', 'test', 'test', 'first'),
    ('verify_val_vs_test_bg.csv', 'val', 'test', 'first'), ('shifts_val_vs_test_bg.csv', 'val', 'test', 'shift'),
]


PAIRS = [('train', 'train'), ('val', 'train'), ('test', 'train'), ('val', 'val'), ('test', 'test'), ('val', 'test')]


def _open(name):
    p = D / name
    if p.exists(): return p.open()
    if Path(str(p) + '.gz').exists(): return gzip.open(str(p) + '.gz', 'rt')
    return None


def links():
    """(file, a, b, confirmed?) for every link that passes a rule; files not made yet are skipped."""
    for qs, rs in PAIRS:
        for name in (f'gp_{qs}_vs_{rs}.csv', f'all_{qs}_vs_{rs}.csv'):
            fh = _open(name)
            if fh is None:
                print('missing', name); continue
            for r in csv.DictReader(fh):
                if r['query'] == r['ref']: continue
                g = G.first_pass(r) if r['kind'] == 'whole' else G.shifted(r)
                c = G.confirmed_first(r) if r['kind'] == 'whole' else G.confirmed_shift(r)
                if g or c: yield name, (qs, r['query']), (rs, r['ref']), bool(c)
    fh = _open('gp_msar_numbers.csv')
    if fh is not None:
        sp = {r['file_name']: r['official_split'] for r in csv.DictReader((D / 'resplit_split.csv').open())}
        for r in csv.DictReader(fh):
            g = G.first_pass(r) if r['kind'] == 'whole' else G.shifted(r)
            c = G.confirmed_first(r) if r['kind'] == 'whole' else G.confirmed_shift(r)
            if g or c: yield 'gp_msar_numbers.csv', (sp[r['query']], r['query']), (sp[r['ref']], r['ref']), bool(c)
    fh = _open('name_links.csv')
    if fh is not None:
        import polcheck, verify as V
        boxes = {}
        for s_ in ('train', 'val', 'test'): boxes.update(V.coco(s_)[1])
        for r in csv.DictReader(fh):
            if r['kind'] == 'other polarization':
                c = (polcheck.matched(boxes.get(r['a'], []), boxes.get(r['b'], [])) or 0) >= 0.999
            elif r['verdict'] == 'not matching' or r['verdict'].startswith('error'): continue
            else: c = True  # confirmed, flat (kept together), or SARDet's own tiles (by name)
            yield f"name_links: {r['kind']}", (r['split_a'], r['a']), (r['split_b'], r['b']), c
    fh = _open('contained.csv')
    if fh is not None:
        for r in csv.DictReader(fh):
            f = lambda k: float(r[k]) if r[k] not in ('', 'nan') else -9.0
            if max(f('full_r'), f('low_r')) < 0.95: continue
            yield 'contained.csv', (r['chip_split'], r['chip']), (r['host_split'], r['host']), f('bg_r') >= G.BG
    for name, qs, rs, kind in LINKS:
        fh = _open(name)
        if fh is None: continue
        for r in csv.DictReader(fh):
            if not r['ref']: continue
            g = G.first_pass(r) if kind == 'first' else G.shifted(r)
            c = G.confirmed_first(r) if kind == 'first' else G.confirmed_shift(r)
            if g or c: yield name, (qs, r['query']), (rs, r['ref']), bool(c)


def check():
    """Confirmed links whose two images sit in different splits of the saved re-split."""
    new = {(r['official_split'], r['file_name']): r['new_split']
           for r in csv.DictReader((D / 'resplit_split.csv').open())}
    cross = Counter(); imgs = defaultdict(set)
    for name, a, b, c in links():
        if c and new[a] != new[b]:
            pair = '-'.join(sorted((new[a], new[b]))); cross[(name, pair)] += 1; imgs[pair] |= {a, b}
    out = {'links': {f'{n} | {p}': k for (n, p), k in sorted(cross.items())},
           'images': {p: {s: sum(1 for k in v if new[k] == s) for s in ('train', 'val', 'test')} for p, v in imgs.items()}}
    (D / 'resplit_check.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def main():
    ims = images(); src_of = true_source()
    src = {k: src_of.get(k[1], '?') for k in ims}
    conf, unc = UF(), UF()
    stats = {}
    for name, a, b, c in links():
        stats.setdefault(name, Counter())['confirmed' if c else 'uncertain'] += 1
        unc.union(a, b)
        if c: conf.union(a, b)
    stats = {k: dict(v) for k, v in stats.items()}
    comp = defaultdict(list)
    for k in ims: comp[conf.find(k)].append(k)
    ucomp = Counter(unc.find(k) for k in ims)
    sizes = Counter(len(v) for v in comp.values())

    def size_table(cs):
        bins = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 50), (51, 500), (501, 10 ** 9)]
        return {f'{lo}-{hi}' if hi < 10 ** 9 else f'>{lo - 1}':
                {'groups': sum(1 for s in cs if lo <= s <= hi), 'images': sum(s for s in cs if lo <= s <= hi)}
                for lo, hi in bins}

    # official split, how many images in groups that cross splits
    cross = Counter()
    for members in comp.values():
        splits = {s for s, _ in members}
        for s, _ in members:
            cross[(s, len(splits) > 1)] += 1

    # new split, per source: big groups stay in train, the rest is dealt band by band
    new, band_report = A.assign(comp, src, ims, SEED, CAP_SHARE, BANDS)
    moved = Counter((s, new[(s, f)]) for s, f in ims)
    per_source = defaultdict(dict)
    for (s, f), d in new.items():
        per_source[src[(s, f)]][d] = per_source[src[(s, f)]].get(d, 0) + 1
    off = defaultdict(dict)
    for (s, f) in ims:
        off[src[(s, f)]][s] = off[src[(s, f)]].get(s, 0) + 1
    # sanity: no confirmed link crosses the new split
    leaks = sum(1 for k in ims if len({new[m] for m in comp[conf.find(k)]}) > 1)

    out = {'links': stats, 'images': len(ims), 'groups': len(comp),
           'largest_groups': sorted((len(v) for v in comp.values()), reverse=True)[:10],
           'group_sizes': size_table([len(v) for v in comp.values()]),
           'with_uncertain_links': {'groups': len(ucomp), 'largest': sorted(ucomp.values(), reverse=True)[:5]},
           'official_split_images_in_cross_split_groups': {f'{s} {"crosses" if c else "own"}': n
                                                           for (s, c), n in sorted(cross.items())},
           'new_split_sizes': dict(Counter(new.values())),
           'official_to_new': {f'{a} -> {b}': n for (a, b), n in sorted(moved.items())},
           'per_source_official': dict(off), 'per_source_new': dict(per_source),
           'images_in_groups_split_across_new_splits': leaks, 'bands (images per size band)': band_report,
           'cap_share': CAP_SHARE, 'seed': SEED}
    (D / 'resplit.json').write_text(json.dumps(out, indent=1))
    with (D / 'resplit_split.csv').open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['official_split', 'file_name', 'source', 'group', 'group_size', 'new_split'])
        for k in sorted(ims):
            r = conf.find(k)
            w.writerow([k[0], k[1], src[k], f'{r[0]}/{r[1]}', len(comp[r]), new[k]])
    print(json.dumps({k: out[k] for k in out if k not in ('per_source_official', 'per_source_new',
                                                          'bands (images per size band)')}, indent=1))
    print('->', D / 'resplit.json')


if __name__ == '__main__':
    check() if sys.argv[1:2] == ['check'] else main()
