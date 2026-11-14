"""The group-to-split draw of the leak-free re-split, on its own so the cluster can redraw it with another seed.

comp: group -> its images (official split, file name); src: image -> true source; ims: every image.
Per source, val and test get the official number of that source's images, filled with whole groups: a group bigger
than cap_share of a split's target stays in train; the rest is dealt band by band (bands of group sizes), each band
giving val and test the same share of its images. -> ({image: new split}, {band: counts}).
"""
from collections import Counter
import numpy as np


def assign(comp, src, ims, seed, cap_share, bands):
    rng = np.random.default_rng(seed)
    target = Counter((s, src[(s, f)]) for s, f in ims if s != 'train')
    gsrc = {r: Counter(src[k] for k in m).most_common(1)[0][0] for r, m in comp.items()}
    new, band_report = {}, {}
    for source in sorted(set(gsrc.values())):
        tv, tt = target[('val', source)], target[('test', source)]
        roots = [r for r in comp if gsrc[r] == source]
        cap = {'val': cap_share * tv, 'test': cap_share * tt}
        elig = [r for r in roots if len(comp[r]) <= max(cap.values())]
        E = sum(len(comp[r]) for r in elig) or 1
        share = {'val': min(1.0, tv / E), 'test': min(1.0, tt / E)}
        fill = Counter()
        for r in roots:
            for k in comp[r]: new[k] = 'train'
        for lo, hi in bands:
            band = [r for r in elig if lo <= len(comp[r]) <= hi]
            rng.shuffle(band)
            n_band = sum(len(comp[r]) for r in band)
            got = Counter()
            for r in band:
                n = len(comp[r])
                need = {s_: share[s_] * n_band - got[s_] for s_ in ('val', 'test')
                        if n <= cap[s_] and fill[s_] + n <= (tv if s_ == 'val' else tt) + n // 2}
                if not need: continue
                s_ = max(need, key=need.get)
                if need[s_] >= n / 2:
                    got[s_] += n; fill[s_] += n
                    for k in comp[r]: new[k] = s_
            band_report[f'{source} | {lo}-{hi}'] = {'images': n_band, 'val': got['val'], 'test': got['test']}
    return new, band_report
