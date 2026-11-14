"""Are ShipDataset tiles at the same scene and grid position in another polarization (namelinks.py 'other
polarization') the same moment, i.e. the same ships in the same places?

For each pair, the share of each image's boxes that a box of the other image covers (IoU >= 0.5), at zero offset.
Control: the same pairs with the second image swapped for a random ShipDataset tile of another scene that also has
boxes. Pairs where either image has no box are counted apart.

    ~/.venvs/sar-audit/bin/python sardet100k/polcheck.py   # -> results/pol_check.json
"""
import csv, gzip, json, random, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import verify as V

D = HERE / 'results'


def matched(A, B):
    if not A or not B: return None
    hit = lambda X, Y: sum(any(V.iou(x[1:], y[1:]) >= 0.5 for y in Y) for x in X)
    return (hit(A, B) + hit(B, A)) / (len(A) + len(B))


def summary(v):
    v = np.array([x for x in v if x is not None])
    return {'pairs with boxes on both': len(v), 'all boxes matched': round(float((v >= 0.999).mean()), 3),
            'at least half matched': round(float((v >= 0.5).mean()), 3), 'none matched': round(float((v == 0).mean()), 3)}


def main():
    boxes = {}
    for s in ('train', 'val', 'test'): boxes.update(V.coco(s)[1])
    rows = [r for r in csv.DictReader(gzip.open(D / 'name_links.csv.gz', 'rt')) if r['kind'] == 'other polarization']
    real = [matched(boxes.get(r['a'], []), boxes.get(r['b'], [])) for r in rows]
    pool = [f for r in rows for f in (r['a'], r['b']) if boxes.get(f)]
    rng = random.Random(0)
    null = [matched(boxes.get(r['a'], []), boxes[rng.choice(pool)]) for r in rows]
    out = {'pairs': len(rows), 'no box on one side': sum(x is None for x in real), 'real': summary(real),
           'control (random other tile)': summary(null),
           'real, all boxes matched, across official splits': sum(1 for r, x in zip(rows, real)
                                                                  if x is not None and x >= 0.999 and r['split_a'] != r['split_b'])}
    (D / 'pol_check.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
