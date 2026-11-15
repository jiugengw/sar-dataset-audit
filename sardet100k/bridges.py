"""Which links join different sources into one group? For the largest group of the rebuilt re-split: the confirmed
links whose two images have different true sources, by link file, with their pixel scores; and the largest group
when a kind of link is left out (the new rules BG_LOW / THIN, gp_msar_numbers.csv, contained.csv).

    python bridges.py OUT.json
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import groups as G
import resplit as R


def comps(links, ims):
    uf = R.UF()
    for name, a, b, c in links:
        if c: uf.union(a, b)
    out = defaultdict(list)
    for k in ims: out[uf.find(k)].append(k)
    return sorted(out.values(), key=len, reverse=True)


def main(out):
    ims = R.images(); src_of = R.true_source(); src = {k: src_of.get(k[1], '?') for k in ims}
    links = [l for l in R.links() if l[3]]
    big = set(comps(links, ims)[0])
    cross = [l for l in links if l[1] in big and src[l[1]] != src[l[2]]]
    res = {'largest group': len(big), 'its sources': dict(Counter(src[k] for k in big)),
           'cross-source links inside it, by file': dict(Counter(l[0] for l in cross)),
           'cross-source links inside it, by source pair': dict(Counter(' / '.join(sorted((src[l[1]], src[l[2]]))) for l in cross))}
    pairs = {(l[1][1], l[2][1]) for l in cross} | {(l[2][1], l[1][1]) for l in cross}
    ex = []
    for name in sorted({l[0] for l in cross if l[0].endswith('.csv') and l[0].startswith('gp_')}):
        fh = R._open(name)
        for r in csv.DictReader(fh):
            if (r['query'], r['ref']) in pairs and len(ex) < 60:
                ex.append({'file': name, **{k: r.get(k) for k in ('kind', 'query', 'ref', 'full_r', 'low_r', 'bg_r', 'bg_low_r',
                                                                   'ovl_frac', 'ovl_full', 'ovl_low', 'bg_ovl', 'bg_low_ovl',
                                                                   'wrap_full', 'wrap_low', 'bg_wrap', 'bg_low_wrap')},
                           'src': f"{src_of.get(r['query'])} / {src_of.get(r['ref'])}"})
    res['examples'] = ex
    tests = {'all links': lambda l: True,
             'without gp_msar_numbers.csv': lambda l: l[0] != 'gp_msar_numbers.csv',
             'without contained.csv': lambda l: l[0] != 'contained.csv',
             'without cross-source links': lambda l: src[l[1]] == src[l[2]]}
    for name, keep in tests.items():
        res[f'largest groups, {name}'] = [len(c) for c in comps([l for l in links if keep(l)], ims)[:5]]
    G.BG_LOW, G.THIN_BG = None, 9  # switch the thin rule off too
    res['largest groups, without the thin rule'] = [len(c) for c in comps(list(R.links()), ims)[:5]]
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'examples'}, indent=1))
    for e in ex[:25]: print(e)


if __name__ == '__main__':
    main(sys.argv[1])
