"""Chance baseline and visual check for the shift search (shifts.py).

The shift search tries every offset, so it has more chances to line two unrelated images up by luck than the
whole-image search. Chance baseline: 400 random val images, each paired with a random train image of the same
source; the best shift is found exactly as in shifts.py (check()), and the leak rules are applied. Anything that
passes is luck. Also draws 6 overlapping-tile and 4 wrapped-copy pairs from shifts_val.csv for a visual check.

    ~/.venvs/sar-audit/bin/python sardet100k/shift_controls.py
    -> results/shift_controls.json, results/sheets/val_shifted_pairs.png
"""
import csv, json, random, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import shifts as S
import verify as V

D = HERE / 'results'


def rule(sy, sx, wf, wl, of, ol, frac):
    """The shifted-copy rules used for the results (labels left out: chance label transfer is checked elsewhere)."""
    if abs(sy) + abs(sx) <= 2: return None
    if wf >= 0.95 or wl >= 0.95: return 'shifted copy'
    if frac >= 0.25 and max(of, ol) >= 0.95: return 'overlapping tile'
    return None


def main():
    random.seed(3)
    sh = list(csv.DictReader((D / 'shifts_val.csv').open()))
    vims, vb = V.coco('val'); tims, tb = V.coco('train')
    src = lambda n, ims, b: V.source(ims[n]['width'], ims[n]['height'], n.rsplit('.', 1)[1].lower(), {x[0] for x in b.get(n, [])})
    pool = {}
    for n in tims:
        pool.setdefault(src(n, tims, tb), []).append(n)
    qs = random.sample([r['query'] for r in sh], 400)
    jobs = [('val', q, [(random.choice(pool[src(q, vims, vb)]), 'id')]) for q in qs]
    with Pool(8) as p:
        res = p.map(S.check, jobs)
    hits = {}
    for q, (score, ref, t, sy, sx, wf, wl, of, ol, frac) in res:
        g = rule(sy, sx, wf, wl, of, ol, frac)
        if g: hits[g] = hits.get(g, 0) + 1
    out = {'random_pairs': len(res), 'passed_rules_by_chance': hits}
    real = {}
    for r in sh:
        g = rule(int(r['shift_y']), int(r['shift_x']), *(float(r[k]) for k in ('wrap_full', 'wrap_low', 'ovl_full', 'ovl_low', 'ovl_frac')))
        if g: real[g] = real.get(g, 0) + 1
    out['real_val_matches'] = real; out['val_images'] = len(vims)
    (D / 'shift_controls.json').write_text(json.dumps(out, indent=1)); print(out)

    def g(r):
        return rule(int(r['shift_y']), int(r['shift_x']), *(float(r[k]) for k in ('wrap_full', 'wrap_low', 'ovl_full', 'ovl_low', 'ovl_frac')))
    picks = random.sample([r for r in sh if g(r) == 'overlapping tile'], 6) + random.sample([r for r in sh if g(r) == 'shifted copy'], 4)
    s = 170; sheet = Image.new('L', (2 * s + 10, len(picks) * (s + 16)), 255); d = ImageDraw.Draw(sheet)
    for k, r in enumerate(picks):
        a = np.ascontiguousarray(V.F[r['transform']](V.grey('val', r['query']))); b = V.grey('train', r['ref'])
        y = k * (s + 16)
        d.text((2, y + 1), f"{g(r)} | shift {r['shift_y']},{r['shift_x']} | shared {float(r['ovl_frac']):.2f} | "
                           f"r {float(r['ovl_full']):.2f}/{float(r['ovl_low']):.2f} | boxes {r['box_recall']}", fill=0)
        for x0, im in ((0, a), (s + 10, b)):
            sheet.paste(Image.fromarray(im.clip(0, 255).astype(np.uint8)).resize((s, s)), (x0, y + 14))
    sheet.save(D / 'sheets' / 'val_shifted_pairs.png'); print('->', D / 'sheets' / 'val_shifted_pairs.png')


if __name__ == '__main__':
    main()
