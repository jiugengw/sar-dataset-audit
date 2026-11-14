# SARDet-100K leakage audit

**SARDet-100K** (NeurIPS 2024) merges 10 SAR detection datasets into 116,598 images. This audit uses the official
split: train 94,493, val 10,492 and test 11,613 images, and builds a leak-free re-split of the same sizes
([below](#leak-free-re-split)). The Kaggle copy also contains
`SARDet_100K/test.json` (test labels, 24,203 boxes) and `SARDet_100K/mapping.json` (each image's original source
file name and tile position). These sit at the top level of the dataset folder, not under `Annotations/`.

## Main findings (13 Nov 2026)

| | val | test |
|---|---|---|
| **Confirmed leaks** (the background repeats too) | **5,101 of 10,492 (48.6%)** | **4,816 of 11,613 (41.5%)** |
| ... exact copies (often flipped or turned) | 1,531 | 1,321 |
| ... re-processed copies (same image, small pixel changes) | 644 | 662 |
| ... shifted copies (a neighbouring crop; first read as wrapped round, see Controls) | 1,082 | 1,163 |
| ... overlapping tiles (share at least 25% of the image) | 1,844 | 1,670 |
| Uncertain (a rule matched, but chance matches are common here) | 2,434 (23.2%) | 2,461 (21.2%) |
| All rules (upper bound) | 71.8% | 62.7% |
| Confirmed copies inside the same split (redundant images) | 10.3% | 14.8% |
| Confirmed copy in the other split (val against test) | 755 val images (7.2%) | 607 test images (5.2%) |

- **Why two levels.**
  - A control against a shuffled train pool (`shift_null.py`) showed the problem. In sparse images (one bright ship
    on dark sea), the search across 94k train images often finds a *different* ship that lines up. That match
    passes the speckle-reduced and shift rules.
  - A leak counts as confirmed only when the background, with the bright targets masked, also matches at full
    resolution (`background.py`). Against the shuffled pool, 13 of 9,607 images reach that.
  - Uncertain matches are almost all 256 px ShipDataset/MSAR chips (about 2,200 per split), HRSID (about 140) and
    SSDD (about 50).
  - Some uncertain matches are real copies whose background was re-processed (`results/sheets/val_reprocessed_background.png`).
    So the true share lies between the two rows.
  - The earlier 71.8% / 62.7% headline counted all of them and was too high.
- **A detector trained on train (YOLO11s, 24 epochs) scores higher on leaked val images than on clean images from
  the same source** (`score_groups.py model`, `results/scores_model.json`):

  | val mAP | confirmed leak | uncertain | clean |
  |---|---|---|---|
  | SAR-AIRcraft | **64.8** (1,054) | - | 49.1 (865) |
  | ShipDataset/MSAR 256 px chips | **62.3** (2,013) | 62.8 (2,192) | 53.9 (1,054) |
  | HRSID | 46.3 (15) | 68.4 (141) | 59.4 (825) |
  | OGSOD | 22.8 (1,815) | 27.0 (13) | 24.8 (6) |
  | All val | 47.3 | 59.9 | 56.2 |

  - Within a source, leaked images score 9-16 mAP higher (SAR-AIRcraft +15.7, ship chips +8.4).
  - Over all of val, leaked images score lower, because most of them are OGSOD, which every image of scores about 23.
  - Uncertain chips score like confirmed ones. That suggests many of them are real repeats, but the model score
    alone cannot tell a copy from an easy image.
- **Copying labels, with no model, scores 36.2 mAP on val and 34.0 on test** (AP50 58.5 / 55.1). Each image gets
  the boxes of its best train match, moved the matched way.

  | Copy-the-labels mAP (AP50) | val | test |
  |---|---|---|
  | All images | 36.2 (58.5) | 34.0 (55.1) |
  | Confirmed leaks | 49.1 (70.2) | 49.1 (69.3) |
  | Clean (luck level) | 4.1 (9.6) | 2.9 (7.1) |

  - Uncertain matches also score well (54.4 / 43.7). But that says nothing about whether they are copies: the boxes
    were lined up by the same ship that made the match.
  - Per group and source: `results/scores_copy.json` (val) and `results/scores_copy_test.json` (test).
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
  - But every confirmed leak in val matches a *different* original file of the same source dataset.
    - Only 16 cross sources, all between ShipDataset and MSAR.
    - The 235 other cross-source matches (mostly HRSID with ShipDataset) fail the background check.
      `results/sheets/val_cross_source.png` shows them: different ships lined up.
  - So OGSOD, MSAR, ShipDataset, SAR-AIRcraft and SADD each contain copies, flipped copies, shifted copies or
    overlapping crops under different file names, and a per-file split cannot keep them apart.
  - SAR-AIRcraft example: its 4,368 originals (all of SAR-AIRcraft-1.0) overlap each other. Val tiles overlap train
    tiles at small offsets (12, 24, 112 px ...), never at SARDet's own tile step (312 px).
- **By source (val, confirmed):**
  - OGSOD 99%, SADD 93%, MSAR 72%, SAR-AIRcraft 55%, ShipDataset/MSAR chips 38% (+42% uncertain).
  - HRSID 2% (+14% uncertain).
  - Per-source counts are in `results/summary.json` (`per_source_evidence`).

## Method

| Step | Script | What it does |
|---|---|---|
| 1 | `candidates.py` | 32 px area-averaged thumbnails (averaging removes most speckle), FAISS search under all 8 flips and 90-degree turns |
| 2 | `verify.py` | Full-resolution and speckle-reduced correlation; train boxes moved onto the val image (label transfer) |
| 3 | `shifts.py` | Shift-proof search (\|FFT\| of thumbnails), exact shift by FFT cross-correlation, wrap-round and overlap scores, label transfer after the shift |
| 3b | `shift_null.py` | Control: the whole shift search against a block-shuffled train pool (no real scene can match) |
| 3c | `background.py` | Full-resolution r with the bright targets masked, for every matched pair; >= 0.5 confirms a leak |
| 3d | `wrapcheck.py` | Do 'wrapped' matches really wrap round? Scores the wrapped-round part and the rest separately (`results/wrap_check.json`) |
| 4 | `groups.py` | One group per image, strongest first: exact, re-processed, shifted, overlapping tile, same target (labels), clean |
| 5 | `score_groups.py` | COCO mAP per group and source: the copy-the-labels baseline, and a trained detector's val predictions |
| 6 | `label_controls.py` | Chance level of the label rule: boxes from random same-source train images |
| 7 | `provenance.py` | Original file and tile position of each image (from `mapping.json`); checks whether any original spans splits, and where each leak's match comes from |
| 8 | `summary.py` | Writes `results/summary.json` |
| 9 | `resplit.py` | Groups from confirmed links only (union-find over all checks), then a per-source re-split of whole groups |
| 10 | `yolo_resplit.py`, `cluster/job_c.sbatch`, `score_resplit.py` | The same YOLO11s run on the re-split, scored per source next to the official split |

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
  - **Correction: they are not wrapped.** `wrapcheck.py` scores the part that wrapped round and the rest separately.
    Of 29,449 confirmed 'wrapped' matches (val and test pairs), 28,532 match in one part only: plain shifts, i.e.
    neighbouring crops. A half-rolled 256 px chip lines up with its half-tile neighbour, which is why the rolled
    control matched. None is clearly rolled; the 71 where both parts correlate are 1-2 px or 24 px shifts and
    streaky MSAR frames that look alike at any shift. ShipDataset half-tile neighbours mostly have file names one
    grid step apart. The rule is now called 'shifted copy' (older result files say 'wrapped copy').
- **Shift rules against random pairs** (`shift_controls.py`). 0 of 400 random same-source pairs pass, even with the
  best shift chosen. This turned out too weak: the real search keeps the best of 94k train images, not a random one.
- **Shift rules against a shuffled pool** (`shift_null.py`, `results/shift_null.json`). This is the full search,
  with every train image (and its boxes) cut into 4 x 4 blocks and shuffled. Ships survive inside a block; scenes do
  not.
  - Val images that still pass: 1,694 shifted, 252 overlapping and 640 same target, against 2,420 / 2,126 / 403 on
    the real pool.
  - Nearly all of them are 256 px ship chips, HRSID and SSDD.
  - With the background check, 13 pass. So the background check is what confirms a leak.
  - The whole-image negative control above shuffles blocks too. But a centred ship is split across four blocks
    there, so it could not show this.
- **Label rule against random pairs** (`label_controls.py`). Boxes from a random train image of the same source
  (true source from `mapping.json`), moved the same way as the real match, reach the rule's box recall of 0.5 for
  0.35% of val and 0.44% of test images, against 52% / 45% for the real best matches. SIVED (vehicles) is
  highest by chance, at 10%; every other source is at 2.3% or less.
- **Visual checks:** `results/sheets/`.

## Leak-free re-split

`mapping.json` gives the original files, but the leaks run *between* originals, so a per-file split cannot remove
them. `resplit.py` links every image to its confirmed copies and splits whole groups instead.

- **Links.** Every check's best match per image: val and test against train, train against train (whole image and
  shift search), within val, within test, and val against test. Only confirmed links (the background repeats too)
  join images.
  - Uncertain links are left out on purpose. With them, chance matches between sparse ship chips chain **31,202
    images into one group**. With confirmed links only, the largest group has 64 images.
- **Train is as redundant as val and test.** 56,558 train images (60%) have a confirmed copy elsewhere in train.
- **Groups.** 116,598 images form 62,989 groups: 40,004 single images, 11,403 pairs, 9,301 of 3-5 images, 2,281 of
  6-64.
  - Under the official split, 5,214 val images (49.7%) and 4,920 test images (42.4%) sit in a group that also has
    images in another split.
- **New split** (`results/resplit_split.csv`, seed 0). For each source, groups are shuffled and handed to val and
  test until each has as many of that source's images as the official split. Groups over 50 images stay in train.
  - Sizes: train 94,488, val 10,495, test 11,615. Every source is within 5 images of its official count per split.
  - No confirmed group crosses the new splits. Uncertain links still can, so sparse chips may still leak.
- **The gap.** One YOLO11s run on the re-split (`cluster/job_c.sbatch`, the same recipe as `job_a`) against the
  official split's 46.17 val mAP. Running; scored by `score_resplit.py` into `results/scores_resplit.json`.

## Still to do

- **The re-split's val mAP** (job above), per source against the official split.
- **Val against test with the shift search.** Only whole-image matches were checked between val and test, so shifted
  copies and overlapping tiles across them are not linked yet.
- **Sparse chips.** A better test for the uncertain ShipDataset/MSAR/HRSID matches, for example by comparing the
  ship's own pixels at full resolution.

## Reproduce

```bash
python3 -m venv ~/.venvs/sar-audit && ~/.venvs/sar-audit/bin/pip install kaggle six faiss-cpu numpy pillow
~/.venvs/sar-audit/bin/kaggle datasets download greatbird/sardet-100k -p sardet100k/data --unzip
P=~/.venvs/sar-audit/bin/python
$P sardet100k/candidates.py thumbs
for q in val test; do $P sardet100k/candidates.py search $q train; $P sardet100k/candidates.py search $q $q; done
$P sardet100k/candidates.py search train train; $P sardet100k/candidates.py search val test
for q in val test; do $P sardet100k/verify.py $q; $P sardet100k/verify.py $q $q; $P sardet100k/shifts.py $q; done
$P sardet100k/verify.py train train; $P sardet100k/verify.py val test; $P sardet100k/shifts.py train
for f in shifts_val shifts_test shifts_train verify_val verify_test verify_val_vs_val verify_test_vs_test \
         verify_train_vs_train verify_val_vs_test; do $P sardet100k/background.py $f.csv; done
$P sardet100k/shift_null.py val; $P sardet100k/background.py shifts_val_null.csv --null
$P sardet100k/controls.py; $P sardet100k/shift_controls.py
for q in val test; do ~/.venvs/sar-yolo/bin/python sardet100k/score_groups.py copy $q; done   # needs pycocotools (sar-yolo env)
$P sardet100k/provenance.py; $P sardet100k/label_controls.py
$P sardet100k/summary.py
$P sardet100k/resplit.py; $P sardet100k/yolo_resplit.py   # then cluster/job_c.sbatch, then score_resplit.py
```

The Kaggle download needs a Kaggle API token in `~/.kaggle/`.
