"""Background check for matched pairs: Pearson r at full resolution with the bright targets masked out.

Sparse images (one bright ship on dark sea) can pass the correlation rules with a *different* train image, because
the ship dominates the variance once the two ships are lined up (shift_null.py found this). A true copy, wrapped copy
or overlapping tile also repeats the background (sea clutter, noise, JPEG blocks) pixel for pixel; a different image
does not. So each pair is lined up exactly as the rule matched it, the brightest pixels of either image (top 2%, grown
by 4 px) are masked, and r is taken over the rest ("bg_r"), at full resolution.

    ~/.venvs/sar-audit/bin/python sardet100k/background.py shifts_val.csv          # adds bg_r -> *_bg.csv
    ~/.venvs/sar-audit/bin/python sardet100k/background.py shifts_val_null.csv --null
    ~/.venvs/sar-audit/bin/python sardet100k/background.py verify_val.csv
"""
import csv, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import verify as V
import shifts as S

TOP, GROW = 0.98, 9  # mask the brightest 2% of each image, grown by a 9 px window (4 px each side)


def bg_r(a, b):
    m = (a > np.quantile(a, TOP)) | (b > np.quantile(b, TOP))
    keep = ~maximum_filter(m, size=GROW)
    if keep.sum() < 0.2 * keep.size or a[keep].std() < 1e-3 or b[keep].std() < 1e-3:
        return float('nan')  # too little (or flat) background to judge
    return V.pearson(a[keep], b[keep])


def aligned(qsplit, row, null, rsplit='train'):
    if null:
        import shift_null  # noqa: F401  (patches V.grey so train images come back block-shuffled)
    a0 = V.grey(qsplit, row['query']); b = V.grey(rsplit, row['ref'])
    a = np.ascontiguousarray(V.F[row['transform']](a0))
    if a.shape != b.shape:
        a = np.asarray(Image.fromarray(a).resize(b.shape[::-1], Image.BILINEAR), dtype=np.float32)
    if 'shift_y' in row:
        sy, sx = int(row['shift_y']), int(row['shift_x'])
        if row.get('_mode') == 'overlap':
            oa, ob, _ = S.overlap(a, b, sy, sx)
            return oa, ob
        a = np.roll(a, (sy, sx), (0, 1))
    return a, b


def job(args):
    qsplit, row, null, rsplit = args
    if not row['ref']:
        return row['query'], ''
    try:
        out = []
        for mode in (('wrap', 'overlap') if 'shift_y' in row else ('same',)):
            row['_mode'] = mode; a, b = aligned(qsplit, row, null, rsplit)
            out.append(bg_r(a, b) if min(a.shape) >= 16 else float('nan'))
        return row['query'], ';'.join(f'{x:.4f}' for x in out)
    except Exception as e:
        return row['query'], f'error {e}'


def main(name, null=False):
    path = HERE / 'results' / name
    rows = list(csv.DictReader(path.open()))
    parts = Path(name).stem.split('_'); qsplit = parts[1]
    rsplit = parts[3] if len(parts) > 3 and parts[2] == 'vs' else 'train'  # verify_val_vs_val -> val
    with Pool() as p:
        res = dict(p.imap_unordered(job, [(qsplit, r, null, rsplit) for r in rows], chunksize=32))
    errs = [v for v in res.values() if v.startswith('error')]
    print(len(rows), 'pairs,', len(errs), 'errors', errs[:3])
    out = path.with_name(path.stem + '_bg.csv')
    with out.open('w', newline='') as fh:
        cols = list(rows[0].keys()); extra = ['bg_wrap', 'bg_ovl'] if 'shift_y' in rows[0] else ['bg_r']
        w = csv.writer(fh); w.writerow(cols + extra)
        for r in rows:
            v = res[r['query']].split(';') if res[r['query']] and not res[r['query']].startswith('error') else [''] * len(extra)
            w.writerow([r[c] for c in cols] + v)
    print('->', out)


if __name__ == '__main__':
    main(sys.argv[1], '--null' in sys.argv)
