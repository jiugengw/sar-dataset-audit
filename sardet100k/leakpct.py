"""Share of official val / test images with a confirmed link (groups.py rules, every link file resplit.py uses) to
an official train image: directly, or through a chain (in a group that holds train images). Overall and per source.

    python leakpct.py OUT.json
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import resplit as R


def main(out):
    ims = R.images(); src_of = R.true_source(); src = {k: src_of.get(k[1], '?') for k in ims}
    uf = R.UF(); direct = set()
    for name, a, b, c in R.links():
        if not c: continue
        uf.union(a, b)
        if a[0] == 'train' and b[0] != 'train': direct.add(b)
        if b[0] == 'train' and a[0] != 'train': direct.add(a)
    has_train = {uf.find(k) for k in ims if k[0] == 'train'}
    res = {}
    for s in ('val', 'test'):
        part = [k for k in ims if k[0] == s]
        chain = {k for k in part if uf.find(k) in has_train}
        row = {'images': len(part), 'direct link to train': len(direct & set(part)),
               'direct %': round(100 * len(direct & set(part)) / len(part), 1),
               'in a group with train (chains too)': len(chain), 'group %': round(100 * len(chain) / len(part), 1),
               'per source (direct %)': {}}
        n = Counter(src[k] for k in part); d = Counter(src[k] for k in part if k in direct)
        for x in sorted(n, key=lambda x: -n[x]): row['per source (direct %)'][x] = f'{d[x]}/{n[x]} = {100 * d[x] / n[x]:.1f}%'
        res[s] = row
    Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
