# SARDet-100K leakage audit

**SARDet-100K** (NeurIPS 2024) merges 10 SAR detection datasets into 116,598 images. The official split is train
94,493, val 10,492 and test 11,613 images (81 / 9 / 10). The Kaggle copy also contains `SARDet_100K/test.json` (test
labels, 24,203 boxes) and `SARDet_100K/mapping.json` (each image's original source file name and tile position).
These sit at the top level of the dataset folder, not under `Annotations/`.

## Main findings (16 Nov 2026)

| | val | test |
|---|---|---|
| **Images with a confirmed copy in train** (direct link, background repeats too) | **9,659 of 10,492 (92.1%)** | **9,672 of 11,613 (83.3%)** |
| Images in a group that reaches train through a chain of copies | 96.1% | 96.3% |
| YOLO11s mAP, official split (train seed 0 / seed 1) | 46.17 / 46.08 | 43.48 / 43.16 |
| YOLO11s mAP, leak-free re-split (3 draws) | 33.1 / 31.2 / 38.5 | 35.6 / 32.5 / 33.1 |
| **YOLO11s mAP, leak-free 5-fold group cross-validation** | **34.7 ± 3.4** (folds 29.5-37.8) | |

- **Almost every val and test image has a copy in train.** Copies include exact and flipped copies, re-processed
  copies, shifted crops, overlapping tiles, chips cut from larger images and the same scene in another polarization.
  - The leak rate counts **direct links only**: the image itself has a confirmed copy in train. Chains (A copies B,
    B overlaps C, ...) are not counted: the 60th image of a chain shares nothing with the first. The chain figures
    above matter only for the re-split.
  - It is a **lower bound**: the search finds about 80% of planted copies ([Controls](#controls)). The shuffled-pool
    control gives at most ~3% false links, most of them real pairs that survived the shuffle.
- **The leaks inflate the official score by about 9 mAP** (43.5 on official test against 34.7 leak-free).
  - Training noise is about 0.3 mAP (two seeds on the official split).
  - The leak-free score moves much more between draws (29.5 to 38.5): it depends on which scenes land in test. So
    a single leak-free split is not enough; the cross-validation mean is the estimate to use.
- **The drop is not where the leaks are.** Ship datasets leak the most but barely drop. Airports (SAR-AIRcraft) and
  OGSOD drop by about half: the model memorises the scenes, not the targets.

  | mAP | leak rate (val / test) | official test | cross-validation mean (fold range) |
  |---|---|---|---|
  | SAR-AIRcraft | 79.6% / 48.8% | 45.8 | **27.4** (20.3-36.7) |
  | OGSOD | 99.2% / 99.5% | 22.6 | **11.4** (9.5-14.6) |
  | SADD | 100% / 100% | 51.7 | 37.2 (12.6-55.9) |
  | MSAR | 98.6% / 98.9% | 50.8 | 40.7 (33.8-49.1) |
  | ShipDataset | 96.8% / 97.2% | 60.0 | 57.6 (57.0-59.1) |
  | HRSID | 90.2% / 90.2% | 58.9 | 59.8 (51.3-67.0) |
  | SIVED | 48.1% / 50.5% | 64.2 | 61.5 (59.9-62.8) |
  | SSDD | 16.4% / 19.0% | 69.0 | 68.5 (66.8-70.5) |
  | AIR-SARShip | 18.4% / 43.6% | 27.8 | 29.2 (23.5-32.8) |
  | **All** | **92.1% / 83.3%** | **43.5** | **34.7** (29.5-37.8) |

- **Airports differ hugely.** Each big SAR-AIRcraft group (one airport's tiles), scored by the cross-validation model
  that never saw it:

  | airport group (images) | 2,560 | 2,056 | 1,824 | 1,544 | 1,388 | 1,158 | 980 | 955 | 929 | 767 | 741 | 739 | 729 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | mAP | 27.4 | 24.0 | 31.2 | **7.7** | 17.0 | 31.6 | 32.3 | 41.3 | 29.4 | 36.2 | **5.4** | 44.2 | 37.6 |

  Whether an easy or a hard airport lands in test moves the headline by several points. This is most of the spread
  between draws and folds. All groups of 100+ images: `results/cv_summary.json` (`groups`).
- **Inside the official split, leaked airport tiles score twice as high as clean ones.** The official model on
  official val / test, with images split by whether they have a direct confirmed link to train
  (`score_leak.py`, `cluster/job_aa`, `results/scores_leak.json`):

  | mAP (images) | val leaked | val clean | test leaked | test clean |
  |---|---|---|---|---|
  | SAR-AIRcraft | **63.1** (1,530) | **26.7** (393) | **56.5** (1,460) | **31.2** (1,529) |
  | ShipDataset | 61.2 (3,845) | 51.8 (128) | 60.3 (3,859) | 49.3 (113) |
  | HRSID | 59.7 (885) | 55.9 (96) | 58.6 (885) | 64.3 (96) |
  | SSDD | 76.5 (19) | 66.4 (97) | 68.4 (22) | 69.5 (94) |
  | All | 46.4 (9,659) | 33.7 (833) | 44.7 (9,672) | 38.4 (1,941) |

  - Clean airport tiles score 27-31, the same as the leak-free cross-validation (27.4). So the leak-free number is
    what the model really does on an airport it has not seen.
  - ShipDataset's clean images also score about 10 lower, but removing the leaks barely changes ShipDataset
    (cross-validation 57.6). Images with no copy are probably unusual ones, so this gap is about which images
    have copies, not memorisation. The cross-validation, not this split, measures the inflation.
  - OGSOD, MSAR and SADD have 10-31 clean images each: too few to compare. 'All' mixes sources and means little.

Leak rates per source: `results/leakpct.json`. Scores: `results/scores_resplit.json` (official),
`results/scores_official_trainseed1.json`, `results/scores_final_seed{0,1,2}.json` (re-split draws; their keys say
'redrawn re-split' also for the official retrain), `results/cv_summary.json` and `results/cv_pooled.json`.

## First pass (13 Nov 2026)

The first pass kept only each image's best match and found **48.6% of val and 41.5% of test** confirmed. The search
in [Method](#method) (every match, low-contrast images, chips inside larger images, file-name links) raised that to
92.1% / 83.3%. Findings from the first pass that still hold:

- **A detector scores higher on leaked images than on clean images of the same source**: SAR-AIRcraft 64.8 vs 49.1,
  ShipDataset/MSAR chips 62.3 vs 53.9 val mAP (`score_groups.py model`, `results/scores_model.json`). Redone with
  the final links above.
- **Copying labels, with no model, scores 36.2 mAP on val and 34.0 on test** (AP50 58.5 / 55.1). Each image gets the
  boxes of its best train match, moved the matched way. Clean images score 4.1 / 2.9 this way (luck level).
  Per group and source: `results/scores_copy.json`, `results/scores_copy_test.json`.
- **Some copies are byte-identical files** under different names: 3,419 groups inside train, 348 inside val, 738
  inside test, and 128 groups shared by train and val.
- **Copies carry disagreeing labels.**
  - OGSOD's flipped copies have their boxes moved exactly **2 px along the flipped axis** (unflipped copies match
    100%): an off-by-one in the flip formula.
  - 4,572 byte-identical SAR-AIRcraft tile pairs carry two hand-drawn sets of boxes. Centres agree within 2 px, but
    one box is about 6% larger at the median.
- **The leaks come from the source datasets, not from SARDet-100K's tiling** (`provenance.py`, using `mapping.json`).
  No original file has tiles in more than one split: SARDet-100K split per original file. But the source datasets
  contain copies, shifted crops and overlapping crops under *different* file names, which a per-file split cannot
  keep apart. Example: SAR-AIRcraft's 4,368 originals overlap each other at small offsets (12, 24, 112 px ...).

## Method

| Step | Script | What it does |
|---|---|---|
| 1 | `candidates.py`, `verify.py` | 32 px averaged thumbnails (averaging removes most speckle), FAISS search under all 8 flips / turns; full-resolution and speckle-reduced correlation |
| 2 | `shifts.py` | Shift-proof search (\|FFT\| of thumbnails), exact shift by cross-correlation, scores on the shared region |
| 3 | `background.py` | Full-resolution correlation with the bright targets masked; >= 0.5 confirms a link |
| 4 | `allpairs.py`, `gpupairs.py` (`cluster/job_f`, `job_g`) | Every match, not only the best, on the GPU, including low-contrast images that step 1 skipped. Version 2 keeps a longer shortlist |
| 5 | `contained.py` (`cluster/job_e`) | Chips cut from larger images, at the same scale or resized |
| 6 | `namelinks.py`, `polcheck.py` | Links by original file name: ShipDataset grid neighbours, HRSID windows, SARDet's own tiles, MSAR neighbouring frame numbers, the same tile in another polarization; the pixels (or boxes) are checked where the names say they overlap |
| 7 | `groups.py` | The rules that make a match a confirmed link (below) |
| 8 | `recall.py`, `recall_diag.py`, `gp2_eval.py` (`cluster/job_j`) | Planted copies: how many the search finds, and why it misses the rest |
| 9 | `shift_null.py`, `gp2_null.py` (`cluster/job_k`) | The search against a block-shuffled train pool: false links |
| 10 | `leakpct.py` (`cluster/job_n`) | Leak rate: val / test images with a direct confirmed link to train, per source |
| 11 | `resplit.py`, `assign.py`, `bridges.py` (`cluster/job_l`, `job_m`) | Union-find groups over all confirmed links; the leak-free re-split; checks that no link joins unrelated sources |
| 12 | `cluster/job_a`, `job_b`, `job_c`, `job_h`, `redraw.py`, `score_resplit.py`, `score_draw.py` | YOLO11s (24 epochs, 512 px) on the official split and on re-split draws, scored per source |
| 13 | `folds.py`, `score_fold.py` (`cluster/job_o`, `job_p`, `cv_rest.sh`) | 5-fold group cross-validation; per-fold, pooled and per-group scores |

**Link rules** (`groups.py`). Every rule needs the background (targets masked) to repeat: bg >= 0.5.

| Link | Rule |
|---|---|
| Exact copy | full-resolution r >= 0.95 (any flip / turn) |
| Re-processed copy | speckle-reduced r >= 0.95 |
| Shifted copy | r >= 0.95 on the shifted image |
| Overlapping tile | shares >= 25% of the image, r >= 0.95 there, full-resolution r on the overlap >= 0.9 |
| Thin overlap | shares 10-25%: background r on the overlap >= 0.9 and full-resolution r >= 0.99 |

A looser 'new speckle' rule (speckle-reduced background >= 0.8-0.9) was tried and dropped: it merged unrelated
sources into one 6,541-image group.

The source of each image was first inferred from its size, format and classes (the annotations have no source
field); `mapping.json` confirmed every assignment.

## Controls

- **Recall: about 80%.** 120 train images per source were edited in 8 ways, saved as a fake split and searched for
  (`recall.py`, `results/gp2_eval.json`). Found: flip + JPEG 98.5%, gain + JPEG 91.3%, shifted copy 92.6%,
  new speckle 67-72%, overlap 75% / 50% / 30% of the image 94.8% / 88.3% / 31.7%. Overall 79.8%.
  - Misses are mostly thin overlaps that never reach the shortlist, and near-empty sea under JPEG, where the
    background check fails on purpose. A longer shortlist (version 2) added little, so recall was left at 80%.
- **False links: at most ~3%.** The whole search against a train pool cut into 4 x 4 shuffled blocks
  (`gp2_null.py`): 350 val images (3.3%) still get a link, nearly all as overlapping tiles. Most are real pairs that
  survive the shuffle (a sea block still matches its own scene), so the true false rate is lower. The test is weak
  for smooth low-resolution sea.
- **Earlier controls** (`controls.py`, `shift_controls.py`, `shift_null.py`, `label_controls.py`, `wrapcheck.py`):
  - Without the background rule, the shuffled pool still passes 1,694 val images as shifted copies (sparse chips:
    one bright ship lines up with another). With it, 13. The background rule is what confirms a link.
  - 'Wrapped' copies are not wrapped: 28,532 of 29,449 match in one part only, i.e. neighbouring crops. The rule is
    called 'shifted copy' (older result files say 'wrapped copy').
  - Random same-source boxes reach the label rule for 0.35% / 0.44% of val / test, against 52% / 45% for real
    matches.
- **Visual checks:** `results/sheets/`.

## Leak-free re-split

- **Groups.** Union-find over every confirmed link joins the 116,598 images into **10,318 groups**
  (`results/resplit.json`). The largest are whole airports: 2,560, 2,056, 1,824, 1,544 and 1,388 images, all
  SAR-AIRcraft. `bridges.py` checked that no link joins different sources into them, and that no single kind of link
  makes them.
  - With uncertain links too, 80,858 images chain into one group, which is why only confirmed links count.
- **Whole chains stay in one split.** A chain's ends need not share any pixels, but splitting a chain anywhere puts
  copies on both sides of the cut. So each group goes to one split.
- **The draw** (`resplit.py`, `assign.py`, `results/resplit_split.csv`, seed 0). Per source and size band, groups are
  dealt to val and test until each matches the official count. No group may take more than half of a band's val or
  test share, so the biggest airports stay in train.
  - Sizes: train 94,417, val 10,580, test 11,601. No group crosses the new splits.
  - 1,163 official val and 883 official test images stay where they were; the rest move.
- **Why cross-validation as well.** One draw moves by up to 7 mAP against another (val 31.2 to 38.5) and keeps the
  big airports in train. The 5-fold group cross-validation (`folds.py`) uses every image as test exactly once:
  - per source, groups are dealt largest first to the fold with the fewest images of that source, so each fold holds
    about 20% of every source (23,284-23,379 images);
  - the big airports land in different folds;
  - each model trains on the other four folds (about 93,300 images, near the official 94,493), with the same recipe
    and nothing tuned, so no val set is needed;
  - pooled over all folds (each image predicted by the model that never saw it) it scores 34.69, the same as the
    fold mean.

## Still to do

- **The 20% the search misses**, mostly thin overlaps: the leak rate is a lower bound.
- **Other benchmarks**: RSAR and M4-SAR (top-level README), and the source datasets' own splits.

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
$P sardet100k/namelinks.py; $P sardet100k/provenance.py
```

Then on the cluster (`cluster/`, staged to `~/sbatch-staging/`; each job's header says what it reads and writes):

1. `job_e`: chips inside larger images.
2. `job_g`: the GPU pair search.
3. `job_j`, `job_k`: recall and false links.
4. `job_l`: groups and re-split.
5. `job_m`, `job_n`: bridges and leak rate.
6. `job_a`, `job_b`: the official split.
7. `job_c`, `job_h`: re-split draws.
8. `job_o` for `FOLD=0..4`, then `job_p`: cross-validation.

The datasets have more files than the cluster's 100k-file quota, so jobs unpack a single tar on the node's local
disk. The Kaggle download needs a Kaggle API token in `~/.kaggle/`.
