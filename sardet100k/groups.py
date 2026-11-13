"""One leak group per image, from all checks (verify_*.csv and shifts_*.csv). Used by score_groups.py and summary.py.

Groups, strongest first (an image gets the first that applies):
  exact copy             full-resolution r >= 0.95 (same position, possibly flipped / turned)
  re-processed copy      speckle-reduced r >= 0.95
  wrapped copy           shifted with wrap-round, r >= 0.95 (full or speckle-reduced)
  overlapping tile       shifted without wrap, shared region >= 25% of the image, r >= 0.95 there
  same target (labels)   val only: speckle-reduced r >= 0.8 and >= 50% of the val boxes recovered by the
                         moved train boxes (same position, or after the shift)
  clean                  none of the above
"""
import csv
from pathlib import Path

D = Path(__file__).resolve().parent / 'results'
ORDER = ['exact copy', 're-processed copy', 'wrapped copy', 'overlapping tile', 'same target (labels)', 'clean']


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


def load(split):
    """{file_name: {'group', 'source', 'first': verify row, 'shift': shifts row or None}} for every image of split."""
    first = {r['query']: r for r in csv.DictReader((D / f'verify_{split}.csv').open())}
    sp = D / f'shifts_{split}.csv'
    sh = {r['query']: r for r in csv.DictReader(sp.open())} if sp.exists() else {}
    out = {}
    for q, r in first.items():
        cands = [g for g in (first_pass(r), shifted(sh[q]) if q in sh else None) if g]
        out[q] = {'group': min(cands, key=ORDER.index) if cands else 'clean', 'source': r['source'],
                  'first': r, 'shift': sh.get(q)}
    return out
