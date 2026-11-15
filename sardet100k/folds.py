"""5-fold group cross-validation for the leak-free evaluation: every group of results/resplit_split.csv (images joined
by proven links, resplit.py) goes whole into one fold, so no link ever crosses train and test, and every image -
the largest airports included - is tested once, by the model trained on the other four folds.

Per source, groups are dealt largest first to the fold that holds the fewest images of that source so far (ties
broken by a seeded shuffle), so each fold gets ~20% of every source and the big airports land in different folds.
Each fold's model trains on ~80% of the images, the share of the official train split (81%).

    python folds.py resplit_split.csv DATA_ROOT OUT.csv [K=5] [SEED=0]   # official_split,file_name,source,group,fold
DATA_ROOT (SARDet_100K, for the annotation order) makes the result independent of the CSV's row order.
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def main(csv_path, root, out, k=5, seed=0):
    k, seed = int(k), int(seed)
    ann = lambda s: next(p for p in (Path(root) / 'Annotations' / f'{s}.json', Path(root) / f'{s}.json') if p.exists())
    by = {(r['official_split'], r['file_name']): r for r in csv.DictReader(open(csv_path))}
    rows = [by[(s, im['file_name'])] for s in ('train', 'val', 'test') for im in json.loads(ann(s).read_text())['images']]
    groups = defaultdict(list)
    for r in rows: groups[r['group']].append(r)
    gsrc = {g: Counter(r['source'] for r in m).most_common(1)[0][0] for g, m in groups.items()}
    rng = np.random.default_rng(seed)
    fold = {}
    for source in sorted(set(gsrc.values())):
        gs = [g for g in groups if gsrc[g] == source]
        gs = [gs[i] for i in rng.permutation(len(gs))]
        gs.sort(key=lambda g: -len(groups[g]))  # stable: equal sizes keep the shuffled order
        fill = [0] * k
        for g in gs:
            f = min(range(k), key=lambda i: (fill[i], i)); fold[g] = f; fill[f] += len(groups[g])
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['official_split', 'file_name', 'source', 'group', 'fold'])
        for r in rows: w.writerow([r['official_split'], r['file_name'], r['source'], r['group'], fold[r['group']]])
    per = defaultdict(Counter); big = defaultdict(list)
    for r in rows: per[r['source']][fold[r['group']]] += 1
    for g, m in groups.items():
        if len(m) >= 500: big[fold[g]].append(f"{gsrc[g]} {len(m)}")
    print(json.dumps({'images per fold': dict(sorted(Counter(fold[r['group']] for r in rows).items())),
                      'per source': {s: [c[i] for i in range(k)] for s, c in sorted(per.items())},
                      'groups of 500+ images per fold': {i: big[i] for i in range(k)}}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:6])
