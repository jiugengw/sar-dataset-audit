"""One leak group per image, from all checks (verify_*.csv and shifts_*.csv). Used by score_groups.py and summary.py.

Groups, strongest first (an image gets the first that applies):
  exact copy             full-resolution r >= 0.95 (same position, possibly flipped / turned)
  re-processed copy      speckle-reduced r >= 0.95
  wrapped copy           shifted with wrap-round, r >= 0.95 (full or speckle-reduced)
  overlapping tile       shifted without wrap, shared region >= 25% of the image, r >= 0.95 there
  same target (labels)   val only: speckle-reduced r >= 0.8 and >= 50% of the val boxes recovered by the
                         moved train boxes (same position, or after the shift)
  clean                  none of the above

Evidence (background.py, after shift_null.py showed that sparse images - one bright ship on dark sea - pass the
speckle-reduced and shift rules with *other* train images): a leak is 'confirmed' when the background, with the bright
targets masked, also repeats at full resolution (bg r >= 0.5; 13 of 9,607 queries reach this against a shuffled train
pool). Otherwise it is 'uncertain'. 'same target (labels)' is uncertain unless the ship check below confirms it.

Ship pixels (ship.py): sparse chips have too little background to judge, so an uncertain match is also confirmed
when the target's own pixels (the brightest 2%, grown by 4 px) repeat at full resolution: fg_r >= SHIP. Different
ships lined up by the search (the shuffled-pool control, real copies left out) reach that in 0.3% of pairs; true
copies in 83%. 'confirmed_by' says which check confirmed an image.
"""
import csv
from pathlib import Path

D = Path(__file__).resolve().parent / 'results'
ORDER = ['exact copy', 're-processed copy', 'wrapped copy', 'overlapping tile', 'same target (labels)', 'clean']
BG = 0.5
SHIP = 0.95
OVL_FULL = 0.9  # a confirmed overlapping tile also needs full-resolution r >= 0.9 over the shared region: searching
                # every offset (gpupairs.py) finds ~30% overlaps of smooth sea at r 0.7-0.8 with the background
                # passing; 98% of the overlaps confirmed by the old search reach 0.9, planted ones 1.0


def _f(r, k):
    return float(r[k]) if r.get(k) not in (None, '') else None


def first_pass(r):
    fr, lr, rec = _f(r, 'full_r'), _f(r, 'low_r'), _f(r, 'box_recall')
    if fr is None: return None
    if fr >= 0.95: return 'exact copy'
    if lr >= 0.95: return 're-processed copy'
    if lr >= 0.8 and rec is not None and rec >= 0.5: return 'same target (labels)'
    return None


def shifted(r):
    if abs(int(r['shift_y'])) + abs(int(r['shift_x'])) <= 2: return None
    wf, wl, of, ol, frac, rec = (_f(r, k) for k in ('wrap_full', 'wrap_low', 'ovl_full', 'ovl_low', 'ovl_frac', 'box_recall'))
    if wf >= 0.95 or wl >= 0.95: return 'wrapped copy'
    if frac >= 0.25 and max(of, ol) >= 0.95: return 'overlapping tile'
    if max(wl, ol) >= 0.8 and rec is not None and rec >= 0.5: return 'same target (labels)'
    return None


def confirmed_first(r):
    g = first_pass(r)
    return g if g in ('exact copy', 're-processed copy') and (_f(r, 'bg_r') or 0) >= BG else None


def confirmed_shift(r):
    g = shifted(r)
    if g == 'wrapped copy' and (_f(r, 'bg_wrap') or 0) >= BG: return g
    if g == 'overlapping tile' and (_f(r, 'bg_ovl') or 0) >= BG and (_f(r, 'ovl_full') or 0) >= OVL_FULL: return g
    return None


def load(split):
    """{file_name: {'group', 'evidence', 'confirmed_group', 'source', 'first': verify row, 'shift': shifts row or None}}
    for every image of split. 'group' is the strongest rule passed; 'confirmed_group' the strongest confirmed one."""
    vb, sb = D / f'verify_{split}_bg.csv', D / f'shifts_{split}_bg.csv'
    vp = vb if vb.exists() else D / f'verify_{split}.csv'
    first = {r['query']: r for r in csv.DictReader(vp.open())}
    sp = sb if sb.exists() else D / f'shifts_{split}.csv'
    sh = {r['query']: r for r in csv.DictReader(sp.open())} if sp.exists() else {}
    shp = D / f'ship_{split}.csv'
    ship = {r['query']: float(r['fg_r']) for r in csv.DictReader(shp.open())
            if r['set'] == 'uncertain' and r['fg_r'] not in ('', 'nan')} if shp.exists() else {}
    out = {}
    for q, r in first.items():
        s = sh.get(q)
        cands = [g for g in (first_pass(r), shifted(s) if s else None) if g]
        conf = [g for g in (confirmed_first(r), confirmed_shift(s) if s else None) if g]
        group = min(cands, key=ORDER.index) if cands else 'clean'
        cg = min(conf, key=ORDER.index) if conf else None
        by = 'background' if cg else None
        if not cg and group != 'clean' and ship.get(q, 0) >= SHIP:
            cg, by = group, 'ship pixels'
        out[q] = {'group': group, 'confirmed_group': cg, 'confirmed_by': by,
                  'evidence': 'clean' if group == 'clean' else 'confirmed' if cg else 'uncertain',
                  'source': r['source'], 'first': r, 'shift': s}
    return out
