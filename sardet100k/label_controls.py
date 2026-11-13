"""Chance level of the 'same target (labels)' rule: does a RANDOM train image of the same source also carry boxes that
land on this image's boxes?

For every labelled val / test image, pick a random train image of the same true source (mapping.json, provenance.py),
move its boxes with the transform of the image's real best match (verify.py), and count the image's boxes hit at
IoU >= 0.5 with the same class. The real rule needs box recall >= 0.5 (plus speckle-reduced r >= 0.8).

    ~/.venvs/sar-audit/bin/python sardet100k/label_controls.py   # -> results/label_controls.json
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import provenance as P
import verify as V

TRIALS = 3


def recall(qb, tb, t, rw, rh, qw, qh):
    moved = [(b[0], V.move_box(b[1:], t, rw, rh, qw, qh)) for b in tb]
    moved = [(c, m) for c, m in moved if m]
    return sum(any(c == qc and V.iou(m, qbb) >= 0.5 for c, m in moved) for qc, *qbb in qb) / len(qb)


def run(split, prov, tims, tboxes, by_src, rng):
    qims, qboxes = V.coco(split)
    real = {r['query']: r for r in csv.DictReader((V.OUT / f'verify_{split}.csv').open())}
    n = Counter(); passed = Counter(); real_pass = Counter(); rec_sum = Counter()
    for q, im in qims.items():
        qb = qboxes.get(q)
        if not qb: continue
        s = prov[q]['source']; t = real[q]['transform'] if q in real and real[q]['transform'] else 'id'
        r = real.get(q, {})
        if r.get('box_recall') and float(r['box_recall']) >= 0.5: real_pass[s] += 1
        for _ in range(TRIALS):
            ref = by_src[s][rng.integers(len(by_src[s]))]
            x = recall(qb, tboxes.get(ref, []), t, tims[ref]['width'], tims[ref]['height'], im['width'], im['height'])
            n[s] += 1; rec_sum[s] += x; passed[s] += x >= 0.5
    per = {s: {'random_pairs': n[s], 'mean_recall': round(rec_sum[s] / n[s], 4),
               'recall_at_least_half': round(passed[s] / n[s], 4),
               'real_best_match_recall_at_least_half': round(real_pass[s] / (n[s] / TRIALS), 4)} for s in sorted(n)}
    tot = sum(n.values())
    return {'all': {'random_pairs': tot, 'mean_recall': round(sum(rec_sum.values()) / tot, 4),
                    'recall_at_least_half': round(sum(passed.values()) / tot, 4),
                    'real_best_match_recall_at_least_half': round(sum(real_pass.values()) / (tot / TRIALS), 4)},
            'per_source': per}


def main():
    rng = np.random.default_rng(0); prov = P.load()
    tims, tboxes = V.coco('train')
    by_src = defaultdict(list)
    for f in tims:
        if tboxes.get(f): by_src[prov[f]['source']].append(f)
    out = {s: run(s, prov, tims, tboxes, by_src, rng) for s in ('val', 'test')}
    (V.OUT / 'label_controls.json').write_text(json.dumps(out, indent=1))
    for s, o in out.items():
        print(s, 'ALL', o['all'])
        for k, v in o['per_source'].items():
            print(f"   {k:14s} random recall>=.5 {v['recall_at_least_half']:.3f}  (mean {v['mean_recall']:.3f})"
                  f"   real match {v['real_best_match_recall_at_least_half']:.3f}")


if __name__ == '__main__':
    main()
