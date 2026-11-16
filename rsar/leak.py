"""RSAR's leak rate, from the SARDet-100K audit's links. RSAR re-splits SARDet-100K images (class-wise stratified
sampling), so every confirmed SARDet link (any pair of SARDet splits, sardet100k/groups.py rules, the same links as
sardet100k/leakpct.py) is also a link between RSAR images once each RSAR image is mapped to its SARDet image
(match.py, by pixels).

An RSAR val / test image is leaked when
  - the same pixels are also an RSAR train image (a duplicate across RSAR's own splits), or
  - its SARDet image has a direct confirmed link to the SARDet image of an RSAR train image.
Chains (in a union-find group with RSAR train) are reported too, as for SARDet, but are not the leak rate.

    python leak.py rsar_to_sardet.csv SARDET_AUDIT_DIR OUT.json
SARDET_AUDIT_DIR holds resplit.py, groups.py and results/ (as staged in ~/sbatch-staging/audit).
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path


def main(match_csv, audit_dir, out):
    sys.path.insert(0, str(Path(audit_dir)))
    import resplit as R
    src = {r['file_name']: r['source'] for r in csv.DictReader((Path(audit_dir) / 'results' / 'resplit_split.csv').open())}
    rs = []                                  # (rsar split, rsar name, [sardet keys])
    of_sd = defaultdict(set)                 # sardet key -> {rsar split}
    for r in csv.DictReader(open(match_csv)):
        keys = [tuple(x.split('/', 1)) for x in r['sardet'].split(';') if x]
        rs.append((r['rsar_split'], r['rsar_name'], keys))
        for k in keys: of_sd[k].add(r['rsar_split'])
    in_train = {k for k, s in of_sd.items() if 'train' in s}
    # direct links between SARDet images, both ways; union-find for chains
    nbr = defaultdict(set); uf = R.UF(); n_links = Counter()
    for name, a, b, c in R.links():
        if not c: continue
        nbr[a].add(b); nbr[b].add(a); uf.union(a, b); n_links[name.split(':')[0]] += 1
    for _, _, keys in rs:                    # same pixels = one image
        for k in keys[1:]: uf.union(keys[0], k)
    train_roots = {uf.find(k) for k in in_train}
    res = {'confirmed SARDet links used': dict(n_links), 'splits': {}}
    for split in ('val', 'test'):
        part = [x for x in rs if x[0] == split]
        matched = [x for x in part if x[2]]
        dup = {x[1] for x in matched if any(k in in_train for k in x[2])}
        direct = {x[1] for x in matched if x[1] in dup or any(nb in in_train for k in x[2] for nb in nbr[k])}
        chain = {x[1] for x in matched if any(uf.find(k) in train_roots for k in x[2])}
        s_of = {x[1]: src.get(x[2][0][1], '?') for x in matched}
        n = Counter(s_of.values()); d = Counter(s_of[x] for x in direct)
        res['splits'][split] = {
            'images': len(part), 'matched to SARDet': len(matched),
            'same pixels as an RSAR train image': len(dup),
            'direct link to RSAR train (incl. same pixels)': len(direct),
            'direct %': round(100 * len(direct) / max(len(matched), 1), 1),
            'in a group with RSAR train (chains too) %': round(100 * len(chain) / max(len(matched), 1), 1),
            'per source (direct %)': {k: f'{d[k]}/{n[k]} = {100 * d[k] / n[k]:.1f}%' for k in sorted(n, key=lambda k: -n[k])}}
    Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
