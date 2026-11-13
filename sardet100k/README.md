# SARDet-100K leakage audit

**SARDet-100K** (NeurIPS 2024) merges 10 SAR detection datasets into 116,598 images. This audit uses the official
split: train 94,493, val 10,492 and test 11,613 images. Nothing was re-split. The Kaggle copy also contains
`SARDet_100K/test.json` (test labels, 24,203 boxes) and `SARDet_100K/mapping.json` (each image's original source
file name and tile position). These sit at the top level of the dataset folder, not under `Annotations/`.

## Main findings (13 Nov 2026)

| | val | test |
|---|---|---|
| Images that repeat train content | **7,535 of 10,492 (71.8%)** | **6,376 of 11,613 (54.9%)** |
| ... exact copies (often flipped or turned) | 1,531 | 1,321 |
| ... re-processed copies (same image, small pixel changes) | 1,921 | 1,915 |
| ... wrapped copies (train chip slid round, edges wrapping) | 1,293 | 1,338 |
| ... overlapping tiles (share at least 25% of the image) | 1,950 | 1,802 |
| ... same target, other look (needs labels) | 840 | n/a |
| Using pixel rules only (comparable across splits) | 63.8% | 54.9% |
| Copies inside the same split (redundant images) | 20.3% | 15.4% |

- **Copying labels, with no model, scores 36.2 mAP on val** (AP50 58.5): each val image gets the boxes of its best
  train match, moved the matched way. Leaked images score 46.2 mAP; clean images score 4.1 (luck level).
- **Some copies are byte-identical files** saved under different names: 3,419 groups inside train, 348 inside val,
  738 inside test, and 128 groups shared by train and val.
- **Copies carry disagreeing labels.**
  - OGSOD's flipped copies have their boxes moved exactly **2 px along the flipped axis** (0% exact; unflipped copies
    match 100%), which is an off-by-one in the flip formula.
  - 4,572 byte-identical SAR-AIRcraft tile pairs carry two hand-drawn sets of boxes. Box centres agree to within
    2 px, but one box is about 6% larger at the median. Boxes far from the image edge disagree as much as boxes cut
    by an edge.
- **The leaks come from the source datasets, not from SARDet-100K's tiling** (`provenance.py`, using
  `mapping.json`).
  - No original image has tiles in more than one split: SARDet-100K split per original file.
  - But every leaked val image matches a *different* original file: 7,300 (97%) another original of the same
    source, 235 another source dataset.
  - So OGSOD, MSAR, ShipDataset, SAR-AIRcraft and SADD each contain copies, flipped copies, wrapped copies or
    overlapping crops under different file names, and a per-file split cannot keep them apart.
  - SAR-AIRcraft example: its 4,368 originals (all of SAR-AIRcraft-1.0) overlap each other. Val tiles overlap train
    tiles at small offsets (12, 24, 112 px ...), never at SARDet's own tile step (312 px).
- **By source:**
  - Nearly all of OGSOD and SADD val repeats train.
  - The 256 px ship chips (ShipDataset/MSAR) and SAR-AIRcraft are heavily affected through wrapped copies and
    overlapping tiles.
  - HRSID is least affected.
  - Per-source counts are in `results/summary.json`.

## Method

| Step | Script | What it does |
|---|---|---|
| 1 | `candidates.py` | 32 px area-averaged thumbnails (averaging removes most speckle), FAISS search under all 8 flips and 90-degree turns |
| 2 | `verify.py` | Full-resolution and speckle-reduced correlation; train boxes moved onto the val image (label transfer) |
| 3 | `shifts.py` | Shift-proof search (\|FFT\| of thumbnails), exact shift by FFT cross-correlation, wrap-round and overlap scores, label transfer after the shift |
| 4 | `groups.py` | One group per image, strongest first: exact, re-processed, wrapped, overlapping tile, same target (labels), clean |
| 5 | `score_groups.py` | COCO mAP per group and source: the copy-the-labels baseline, and a trained detector's val predictions |
| 6 | `provenance.py` | Original file and tile position of each image (from `mapping.json`); checks whether any original spans splits, and where each leak's match comes from |
| 7 | `summary.py` | Writes `results/summary.json` |

The source of each image was first inferred from its size, format and classes, because the annotations have no
source field. `mapping.json` confirmed every assignment. The two ambiguous groups resolve as follows: the 256 px ship
chips are 3,973 ShipDataset and 1,286 MSAR images, and "512 other" is MSAR and AIR-SARShip.

## Controls (is a match just luck?)

- **Positive control** (`controls.py`). 1,000 train images were corrupted on purpose and searched for:
  - found 76-87% of the time after new speckle, gain changes, JPEG or flips, mostly landing in the
    exact/re-processed groups;
  - shifts of 4 px or more were mostly missed by whole-image search, which is why step 3 exists.
- **Negative control** (`controls.py`). Val was searched against a fake train pool, each image cut into 4 x 4 pieces
  and shuffled: 79 / 225 pairs reach the exact / re-processed rules, against 1,531 / 1,921 real ones.
  - A first version used half-rolled images instead. It "matched" 860 val images pixel for pixel, which is how the
    wrapped copies were found (kept as `negative_roll_half` in `results/controls.json`).
- **Shift rules** (`shift_controls.py`). 0 of 400 random same-source pairs pass, even with the best shift chosen.
- **Label transfer from a random same-source train image** recovers 0.00-0.01 of the val boxes in every source
  except SIVED (0.19).
- **Visual checks:** `results/sheets/`.

## Still to do

- **Score a trained detector on leaked vs clean val.** YOLO11s is training on the SMU cluster (`cluster/job_a.sbatch`;
  the data goes in as one tar unpacked on the node's local disk, because of the 100k-file scratch quota). Then run
  `score_groups.py model results/val_predictions.json`.
- **Leak-free re-split.**
  - `mapping.json` gives the original files, but the leaks run *between* originals. So groups still have to be
    rebuilt by linking files that share ground (train against train too), then following the links.
  - Then split whole groups and measure the gap.
- **Test labels.** `test.json` is in the Kaggle copy, so the label-based checks (same target, copy-the-labels mAP)
  can now run on test too.

## Reproduce

```bash
python3 -m venv ~/.venvs/sar-audit && ~/.venvs/sar-audit/bin/pip install kaggle six faiss-cpu numpy pillow
~/.venvs/sar-audit/bin/kaggle datasets download greatbird/sardet-100k -p sardet100k/data --unzip
P=~/.venvs/sar-audit/bin/python
$P sardet100k/candidates.py thumbs
for q in val test; do $P sardet100k/candidates.py search $q train; $P sardet100k/candidates.py search $q $q; done
for q in val test; do $P sardet100k/verify.py $q; $P sardet100k/verify.py $q $q; $P sardet100k/shifts.py $q; done
$P sardet100k/controls.py; $P sardet100k/shift_controls.py
~/.venvs/sar-yolo/bin/python sardet100k/score_groups.py copy   # needs pycocotools (the sar-yolo env)
$P sardet100k/provenance.py
$P sardet100k/summary.py
```

The Kaggle download needs a Kaggle API token in `~/.kaggle/`.
