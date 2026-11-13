"""Where each SARDet-100K image comes from, using the dataset's own mapping.json (shipped in the Kaggle copy at
data/SARDet_100K/mapping.json: original file name -> SARDet-100K file name, for all 116,598 images).

Original names look like '6SAR-AIRcraft_0001872__512__488___312.jpg': a source index and name, the source image id,
and for tiled sources the tile size and position (x, y). parse() splits them into
(source, original image, x, y). For ShipDataset chips ('Sen_ship_hh_0201612140104404 6'-like names) the original is
the whole chip name; scene() tries to group chips of one acquisition.

    ~/.venvs/sar-audit/bin/python sardet100k/provenance.py   # -> results/provenance.json
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
ROOT = HERE / 'data' / 'SARDet_100K'
TILE = re.compile(r'^(?P<orig>.+?)__(?P<size>\d+)__(?P<x>-?\d+)___(?P<y>-?\d+)$')


def parse(orig_name):
    stem = orig_name.rsplit('.', 1)[0]
    m = re.match(r'^(\d+)(.+?)_(.*)$', stem)
    idx, source, rest = m.group(1), m.group(2), m.group(3)
    t = TILE.match(rest)
    if t:
        return source, t['orig'], int(t['x']), int(t['y'])
    return source, rest, 0, 0


def load():
    """{sardet file name: {'source', 'original', 'x', 'y', 'orig_name'}}"""
    m = json.loads((ROOT / 'mapping.json').read_text())
    out = {}
    for orig_name, new in m.items():
        s, o, x, y = parse(orig_name)
        out[new] = {'source': s, 'original': f'{s}/{o}', 'x': x, 'y': y, 'orig_name': orig_name}
    return out


def splits():
    out = {}
    for s in ('train', 'val', 'test'):
        p = ROOT / 'Annotations' / f'{s}.json'
        p = p if p.exists() else ROOT / f'{s}.json'
        for im in json.loads(p.read_text())['images']:
            out[im['file_name']] = s
    return out


def main():
    import groups as G
    prov = load(); split = splits()
    res = {'images_in_mapping': len(prov), 'images_in_splits': len(split),
           'missing_from_mapping': sorted(set(split) - set(prov))[:20]}
    # 1. originals whose tiles fall in more than one split
    by_orig = defaultdict(set); src_imgs = Counter()
    for f, p in prov.items():
        by_orig[p['original']].add(split.get(f)); src_imgs[p['source']] += 1
    cross = Counter(); origs = Counter()
    for o, ss in by_orig.items():
        s = o.split('/')[0]; origs[s] += 1
        if len(ss - {None}) > 1: cross[s] += 1
    res['per_source'] = {s: {'images': src_imgs[s], 'originals': origs[s], 'originals_split_across_splits': cross[s]}
                         for s in sorted(src_imgs)}
    # 2. inferred source (groups / verify.source) vs true source, on val
    g = G.load('val')
    agree = Counter((v['source'], prov[q]['source']) for q, v in g.items())
    res['inferred_vs_true_source_val'] = {f'{a} -> {b}': n for (a, b), n in agree.most_common()}
    # 3. leaked val images: is the train match the same original, another original of the same source, or another source?
    rel = defaultdict(Counter)
    for q, v in g.items():
        if v['group'] == 'clean': continue
        f, sh = v['first'], v['shift']
        ref = sh['ref'] if v['group'] in ('wrapped copy', 'overlapping tile') and sh else f['ref']
        if v['group'] == 'same target (labels)' and sh and not G.first_pass(f):
            ref = sh['ref']
        a, b = prov[q], prov[ref]
        k = ('same original' if a['original'] == b['original'] else
             'same source, other original' if a['source'] == b['source'] else 'other source')
        rel[v['group']][k] += 1; rel['ALL leaked'][k] += 1
    res['leaked_val_match_relation'] = {k: dict(c) for k, c in rel.items()}
    (HERE / 'results' / 'provenance.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('images_in_mapping', 'images_in_splits')}))
    print('missing from mapping:', len(set(split) - set(prov)))
    print(f"{'source':16s} {'images':>7s} {'originals':>9s} {'split across splits':>20s}")
    for s, d in res['per_source'].items():
        print(f"{s:16s} {d['images']:7d} {d['originals']:9d} {d['originals_split_across_splits']:20d}")
    print('\ninferred -> true source (val):')
    for k, n in res['inferred_vs_true_source_val'].items(): print(f'  {n:6d}  {k}')
    print('\nleaked val images, relation to their train match:')
    for k, c in res['leaked_val_match_relation'].items(): print(f'  {k:24s} {dict(c)}')


if __name__ == '__main__':
    main()
