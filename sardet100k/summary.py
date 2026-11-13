"""Collect the SARDet-100K audit numbers into results/summary.json (leak groups per split and source, duplicates
inside each split, controls, and the copy-the-labels baseline).

    ~/.venvs/sar-audit/bin/python sardet100k/summary.py
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import groups as G

D = HERE / 'results'


def within_split(split):
    """Images with a copy inside the same split (same rules as groups.first_pass), merged into duplicate groups."""
    par = {}

    def find(x):
        while par.get(x, x) != x:
            x = par[x]
        return x
    kinds = Counter()
    rows = list(csv.DictReader((D / f'verify_{split}_vs_{split}.csv').open()))
    for r in rows:
        g = G.first_pass(r)
        if g:
            kinds[g] += 1; par[find(r['query'])] = find(r['ref'])
    members = set(par) | {find(x) for x in par}; groups = len({find(x) for x in members})
    return {'images': len(rows), 'images_with_a_copy_in_split': sum(kinds.values()), 'by_kind': dict(kinds),
            'images_in_duplicate_groups': len(members), 'duplicate_groups': groups,
            'distinct_images': len(rows) - len(members) + groups,
            'redundant_share': round((len(members) - groups) / len(rows), 4)}


def main():
    out = {'splits': 'official (Annotations/train.json, val.json, JPEGImages/test); nothing re-split',
           'group_rules': G.__doc__.strip()}
    for s in ('val', 'test'):
        g = G.load(s); c = Counter(v['group'] for v in g.values()); n = len(g)
        per = defaultdict(Counter)
        for v in g.values():
            per[v['source']][v['group']] += 1
        out[s] = {'images': n, 'leaked': n - c['clean'], 'leaked_share': round((n - c['clean']) / n, 4),
                  'leaked_image_rules_only': n - c['clean'] - c['same target (labels)'],
                  'groups': {k: c[k] for k in G.ORDER},
                  'per_source': {k: {**dict(v), 'images': sum(v.values())} for k, v in sorted(per.items())},
                  'duplicates_inside_split': within_split(s)}
    for name in ('controls.json', 'shift_controls.json', 'scores_copy.json', 'scores_model.json'):
        p = D / name
        if p.exists():
            out[name.removesuffix('.json')] = json.loads(p.read_text())
    (D / 'summary.json').write_text(json.dumps(out, indent=1))
    for s in ('val', 'test'):
        o = out[s]; d = o['duplicates_inside_split']
        print(f"{s}: {o['leaked']} of {o['images']} leaked ({100 * o['leaked_share']:.1f}%), {o['groups']}; "
              f"inside {s}: {100 * d['redundant_share']:.1f}% redundant")
    print('->', D / 'summary.json')


if __name__ == '__main__':
    main()
