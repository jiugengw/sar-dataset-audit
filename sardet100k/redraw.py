"""Redraw the leak-free re-split with another seed, from the groups saved in results/resplit_split.csv (no links
needed, so it runs on the cluster). Same rule as resplit.py (assign.py); seed 0 must give back the saved split.

    python redraw.py resplit_split.csv SEED split.txt DATA_ROOT   # '<official split> <file> <new split>' per image
DATA_ROOT is SARDet_100K (Annotations/{split}.json, or {split}.json at the top, as resplit.py looks): images are taken in their order, as resplit.py does, so that the
same seed deals the groups in the same order.
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assign as A

CAP_SHARE = 0.5
BANDS = [(1, 1), (2, 5), (6, 20), (21, 100), (101, 500), (501, 10 ** 9)]  # as in resplit.py


def main(csv_path, seed, out, root):
    ann = lambda s: next(p for p in (Path(root) / 'Annotations' / f'{s}.json', Path(root) / f'{s}.json') if p.exists())
    by = {(r['official_split'], r['file_name']): r for r in csv.DictReader(open(csv_path))}
    rows = [by[(s, im['file_name'])] for s in ('train', 'val', 'test')
            for im in json.loads(ann(s).read_text())['images']]
    assert len(rows) == len(by), (len(rows), len(by))
    ims = [(r['official_split'], r['file_name']) for r in rows]
    src = {(r['official_split'], r['file_name']): r['source'] for r in rows}
    comp = defaultdict(list)
    for r in rows: comp[r['group']].append((r['official_split'], r['file_name']))
    new, _ = A.assign(comp, src, ims, int(seed), CAP_SHARE, BANDS)
    same = sum(new[(r['official_split'], r['file_name'])] == r['new_split'] for r in rows)
    with open(out, 'w') as fh:
        for s, f in ims: fh.write(f'{s} {f} {new[(s, f)]}\n')
    per = defaultdict(Counter)
    for k, v in new.items(): per[src[k]][v] += 1
    print(json.dumps({'seed': int(seed), 'images': len(rows), 'same new split as the saved one': same,
                      'sizes': dict(Counter(new.values())),
                      'SAR-AIRcraft groups in val / test': {s: len({r['group'] for r in rows if r['source'] == 'SAR-AIRcraft'
                                                                     and new[(r['official_split'], r['file_name'])] == s})
                                                            for s in ('val', 'test')},
                      'per source': {k: dict(v) for k, v in sorted(per.items())}}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
