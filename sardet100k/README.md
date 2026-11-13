# SARDet-100K leakage audit

**SARDet-100K** (NeurIPS 2024) merges 10 SAR detection datasets into 116,598 images. This audit uses the official
split: train 94,493 and val 10,492 with labels, and test 11,613 images (the Kaggle copy has no test labels). Nothing
was re-split.

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
- **Overlaps come from the sources.** SAR-AIRcraft val tiles overlap train tiles at small, repeating offsets (12,
  24, 112 px ...) and never at SARDet's own tile step (312 px). So the overlapping tiles come from different
  SAR-AIRcraft-1.0 originals covering the same ground, which the 8:1:1 split did not keep together.
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
| 6 | `summary.py` | Writes `results/summary.json` |

The source of each image is inferred from its size, format and classes, because the annotations have no source
field. The counts match the official per-source table.

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
- **Rebuild source groups** (tiles linked by overlap, then grouped by following the links), then use them to make a
  leak-free re-split and measure the gap.
  - The maintainer's `mapping.json` (GitHub issue #35) would give these groups directly, but it is not public.
- **Test labels.** `test.json` exists only on the Baidu share and needs a Baidu account to download.

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
$P sardet100k/summary.py
```

The Kaggle download needs a Kaggle API token in `~/.kaggle/`.
