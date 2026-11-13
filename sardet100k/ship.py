"""Target-pixel check for the uncertain matches (sparse chips: one bright ship on dark sea, where bg_r is undefined
or low).

A re-saved or re-processed copy repeats the target's own pixels (its speckle, side lobes, shape) at full resolution;
a *different* ship that the search lined up does not, even when the shapes agree. For each pair, lined up exactly
as its rule matched it (background.aligned), fg_r is the Pearson r over the target pixels only: the brightest 2% of
either image, grown by 4 px (exactly the pixels bg_r leaves out). Also fg_n, the number of those pixels.

Three sets of pairs (val):
  confirmed   pairs confirmed by the background check (true copies)                      -> what copies look like
  uncertain   the strongest rule matched, but not confirmed                              -> to settle
  null        shift_null.py passes against the block-shuffled pool whose reference is NOT one of the query's real
              matches (1,584; the other 1,002 passes were the query's own copy, its ship kept inside a block)
              -> what a different, lined-up ship looks like. Some of these references are real copies the main
              search missed (a wrap by whole 64 px blocks survives the shuffle). On the original images, the pair is
              re-aligned and checked with the confirmed rule (full-resolution r >= 0.95 or background r >= 0.5:
              'unshuffled_full', 'unshuffled_bg'); the null summary leaves those out.

    ~/.venvs/sar-audit/bin/python sardet100k/ship.py real [test]  # confirmed + uncertain -> results/ship_{val,test}.csv
    ~/.venvs/sar-audit/bin/python sardet100k/ship.py null    # -> results/ship_val_null.csv (separate process: the
                                                             #    null patches verify.grey for the whole process)
    ~/.venvs/sar-audit/bin/python sardet100k/ship.py summary # -> results/ship.json
"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import background as B
import groups as G

D = HERE / 'results'
MIN_PIX = 30


def fg_r(a, b):
    m = (a > np.quantile(a, B.TOP)) | (b > np.quantile(b, B.TOP))
    fg = maximum_filter(m, size=B.GROW)
    if fg.sum() < MIN_PIX or a[fg].std() < 1e-3 or b[fg].std() < 1e-3:
        return float('nan'), int(fg.sum())
    return float(np.corrcoef(a[fg], b[fg])[0, 1]), int(fg.sum())


def mode_of(r):
    if 'shift_y' not in r: return 'same'
    g = G.shifted(r)
    if g == 'wrapped copy': return 'wrap'
    if g == 'overlapping tile': return 'overlap'
    return 'wrap' if float(r['wrap_low']) >= float(r['ovl_low']) else 'overlap'


def job(args):
    kind, row, null, split = args
    try:
        row = dict(row); row['_mode'] = mode_of(row)
        a, b = B.aligned(split, row, null)
        if min(a.shape) < 16: return kind, row, float('nan'), 0
        r, n = fg_r(a, b)
        if null:  # is the original (unshuffled) reference a copy anyway?
            import shift_null as SN, shifts as S, verify as V
            a0 = np.ascontiguousarray(V.F[row['transform']](SN._grey('val', row['query']))); b0 = SN._grey('train', row['ref'])
            if a0.shape == b0.shape:
                sy, sx = S.find_shift(a0, b0); w = np.roll(a0, (sy, sx), (0, 1))
                row['unshuffled_full'] = f'{V.pearson(w, b0):.4f}'; row['unshuffled_bg'] = f'{B.bg_r(w, b0):.4f}'
        return kind, row, r, n
    except Exception as e:  # noqa: BLE001
        return kind, row, f'error {e}', 0


def real_pairs(split='val'):
    """The pair behind each image's group: confirmed first, else the rule that gave its (uncertain) group."""
    out = []
    for q, g in G.load(split).items():
        if g['evidence'] == 'clean': continue
        f, s = g['first'], g['shift']
        if g['evidence'] == 'confirmed':
            row = f if G.confirmed_first(f) else s
        else:
            row = f if G.first_pass(f) == g['group'] else s
        out.append(('confirmed' if g['evidence'] == 'confirmed' else 'uncertain', {**row, 'source': g['source'],
                                                                                   'group': g['group']}))
    return out


def null_pairs():
    real = {}
    for name in ('verify_val_bg.csv', 'shifts_val_bg.csv', 'all_val_vs_train.csv'):
        for r in csv.DictReader((D / name).open()):
            real.setdefault(r['query'], set()).add(r['ref'])
    src = {q: g['source'] for q, g in G.load('val').items()}
    return [('null', {**r, 'source': src.get(r['query'], ''), 'group': G.shifted(r)})
            for r in csv.DictReader((D / 'shifts_val_null_bg.csv').open())
            if G.shifted(r) and r['ref'] not in real.get(r['query'], set())]


def run(which, split='val'):
    null = which == 'null'
    pairs = null_pairs() if null else real_pairs(split)
    if null:
        import shift_null  # noqa: F401  (patch before the pool starts, so every worker reads shuffled train)
    with Pool() as p:
        res = list(p.imap_unordered(job, [(k, r, null, split) for k, r in pairs], chunksize=16))
    errs = [r for r in res if isinstance(r[2], str)]
    print(len(res), 'pairs,', len(errs), 'errors', [e[2] for e in errs[:3]])
    out = D / ('ship_val_null.csv' if null else f'ship_{split}.csv')
    with out.open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['set', 'query', 'ref', 'source', 'group', 'mode', 'fg_r', 'fg_n', 'unshuffled_full', 'unshuffled_bg'])
        for kind, row, r, n in res:
            w.writerow([kind, row['query'], row['ref'], row['source'], row['group'], row['_mode'],
                        r if isinstance(r, str) else f'{r:.4f}', n, row.get('unshuffled_full', ''), row.get('unshuffled_bg', '')])
    print('->', out)


def summary():
    null = list(csv.DictReader((D / 'ship_val_null.csv').open()))
    f = lambda r, k: float(r[k]) if r[k] not in ('', 'nan') else 0.0
    missed = [r for r in null if f(r, 'unshuffled_full') >= 0.95 or f(r, 'unshuffled_bg') >= 0.5]
    rows = list(csv.DictReader((D / 'ship_val.csv').open())) + [r for r in null if r not in missed]
    val = lambda r: float(r['fg_r']) if r['fg_r'] not in ('', 'nan') and not r['fg_r'].startswith('error') else np.nan
    out = {}
    for s in ('confirmed', 'uncertain', 'null'):
        v = np.array([val(r) for r in rows if r['set'] == s])
        ok = v[~np.isnan(v)]
        out[s] = {'pairs': len(v), 'undefined': int(np.isnan(v).sum()),
                  'quantiles': {q: round(float(np.quantile(ok, q)), 4) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)} if len(ok) else {},
                  'share_at_least': {t: round(float((ok >= t).mean()), 4) for t in (0.8, 0.85, 0.9, 0.95)} if len(ok) else {}}
    out['null references that are real copies (left out)'] = len(missed)
    (D / 'ship.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    summary() if sys.argv[1] == 'summary' else run(*sys.argv[1:3])
