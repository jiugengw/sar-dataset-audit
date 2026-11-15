"""Version 2 of the GPU search (job23): how many planted copies each confirmation rule finds, and how many false
links it makes against the block-shuffled train pool (the null), next to the real val / test -> train links.

Rules: 'strict' is groups.py (background bg >= 0.5 at full resolution; thin overlaps included). 'bg_low >= t' also
accepts a speckle-reduced match (low r >= 0.9) whose 4x averaged background repeats (bg_low >= t): copies with fresh
speckle, where the full-resolution background no longer matches.

    python gp2_eval.py RUN_DIR recall_truth.csv OUT.json
Run (job27) with groups.py before BG_LOW / THIN_BG were added; the chosen rules are now in groups.py.
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import groups as G

BG_LOW = [None, 0.9, 0.8, 0.7]
THIN = {'thin off': None, 'thin as groups.py': (0.5, 0.95), 'thin bg 0.8 r 0.98': (0.8, 0.98), 'thin bg 0.9 r 0.99': (0.9, 0.99)}
f = lambda r, k: float(r[k]) if r.get(k) not in (None, '', 'nan') else float('nan')


def group(r, t, thin):
    """Which rule confirms row r (None if none)."""
    if r['kind'] == 'whole':
        g = G.confirmed_first(r)
        if g: return g
        if t is not None and f(r, 'low_r') >= 0.9 and f(r, 'bg_low_r') >= t: return 'speckle (bg_low)'
        return None
    g = G.shifted(r)
    if g == 'thin overlap':
        if thin is None: return None
        bg, rr = thin
        return g if f(r, 'bg_ovl') >= bg and f(r, 'ovl_full') >= rr else None
    g = G.confirmed_shift(r)
    if g: return g
    if t is not None:
        if f(r, 'wrap_low') >= 0.9 and f(r, 'bg_low_wrap') >= t: return 'speckle (bg_low)'
        if f(r, 'ovl_low') >= 0.9 and f(r, 'ovl_frac') >= 0.25 and f(r, 'bg_low_ovl') >= t: return 'speckle (bg_low)'
    return None


def links(rows, t, thin):
    hit, why = defaultdict(set), defaultdict(set)
    for r in rows:
        if r['query'] == r['ref']: continue
        g = group(r, t, thin)
        if g: hit[r['query']].add(r['ref']); why[r['query']].add(g)
    return hit, why


def main(run, truth_path, out):
    run = Path(run); truth = list(csv.DictReader(open(truth_path)))
    rows = {k: list(csv.DictReader(open(run / f'gp_{k}.csv'))) for k in
            ('planted_vs_train', 'val_vs_train_null', 'val_vs_train', 'test_vs_train')}
    res = {'queries searched': {q: len({r['query'] for r in rows[f'{q}_vs_train']}) for q in ('val', 'test')}}
    for tn, thin in THIN.items():
        for t in BG_LOW:
            name = f"{tn} | {'no bg_low' if t is None else f'bg_low >= {t}'}"
            hit, _ = links(rows['planted_vs_train'], t, thin)
            n, got = Counter(), Counter()
            for x in truth:
                n[x['kind']] += 1; got[x['kind']] += x['original'] in hit[x['query']]
            nh, nwhy = links(rows['val_vs_train_null'], t, thin)
            row = {'planted found': round(sum(got.values()) / len(truth), 4),
                   'planted found by kind': {k: round(got[k] / n[k], 3) for k in sorted(n)},
                   'null: val images with a false link': len(nh),
                   'null: false links by rule': dict(Counter(g for v in nwhy.values() for g in v))}
            for q in ('val', 'test'):
                row[f'{q} -> train: images with a link'] = len(links(rows[f'{q}_vs_train'], t, thin)[0])
            res[name] = row
            print(name, json.dumps(row), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
