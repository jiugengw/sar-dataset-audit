# RSAR leakage audit

**RSAR** (rotated boxes on SARDet-100K images) has 95,842 images: train 78,837, val 8,467, test 8,538. The README
says only that it "is build on SARDet-100K". In GitHub issue zhasion/RSAR#28 the authors say the splits come from
class-wise stratified sampling and that deduplication was done during cleaning, without saying whether it covered
copies across splits. Data: `RSAR.tar` (5.2 GB) from the Google Drive link in the authors' README, kept packed on the
cluster (about 190k files).

## Main findings (16 Nov 2026)

| | val | test |
|---|---|---|
| RSAR images that are a SARDet-100K image (same pixels) | 8,467 of 8,467 | 8,538 of 8,538 |
| ... in the same SARDet-100K split, under the same file name | all | all |
| Same pixels as an RSAR train image | 0 | 0 |
| **Direct confirmed link to an RSAR train image** | **8,014 (94.6%)** | **8,114 (95.0%)** |
| In a group with RSAR train through chains | 95.9% | 96.1% |

- **RSAR keeps SARDet-100K's own split.** Every RSAR train image is a SARDet train image, every val image a SARDet
  val image and every test image a SARDet test image, unchanged and under the same name (`match.py`,
  `results/match.json`, `results/rsar_to_sardet.csv`). The 20,756 images left out are mostly SAR-AIRcraft, which RSAR
  drops entirely (its aircraft are SADD's).
- **Its deduplication removed exact copies only.** No identical image sits in two RSAR splits: 64 RSAR val / test
  images have an identical file in another SARDet-100K split, and RSAR left that other file out. But the shifted copies, re-processed copies, overlapping tiles and chips that the
  SARDet-100K audit found are all still there.
- **So RSAR leaks even more than SARDet-100K** (95% against 92.1% / 83.3%). It dropped SAR-AIRcraft, the big source
  that leaked least, and kept the ones that leak most:

  | direct leak % (val / test) | RSAR | SARDet-100K |
  |---|---|---|
  | ShipDataset | 96.6 / 97.0 | 96.8 / 97.2 |
  | OGSOD | 99.3 / 99.5 | 99.2 / 99.5 |
  | MSAR | 98.6 / 98.8 | 98.6 / 98.9 |
  | HRSID | 89.9 / 90.1 | 90.2 / 90.2 |
  | SADD | 100 / 100 | 100 / 100 |
  | SIVED | 48.1 / 50.5 | 48.1 / 50.5 |
  | SSDD | 11.2 / 15.5 | 16.4 / 19.0 |
  | AIR-SARShip | 18.4 / 43.6 | 18.4 / 43.6 |
  | SAR-AIRcraft | (dropped) | 79.6 / 48.8 |
  | **All** | **94.6 / 95.0** | **92.1 / 83.3** |

## Method

| Step | Script | What it does |
|---|---|---|
| 1 | `match.py` (`cluster/job_ab`) | Decodes every image of both datasets and hashes its pixels; each RSAR image gets the SARDet-100K image(s) with the same pixels |
| 2 | `leak.py` (`cluster/job_ac`) | Maps every confirmed SARDet-100K link (all split pairs, the [sardet100k](../sardet100k/) rules and link files) onto RSAR's images; an RSAR val / test image leaks when it has a direct link to an RSAR train image (`results/leak.json`) |

Because every RSAR image is a SARDet-100K image, no new search was needed: the SARDet-100K audit already compared
every pair of its images. The leak rate is a lower bound for the same reason as there (the search finds about 80% of
planted copies).

## Score inflation (17 Nov 2026)

**RSAR's official split overstates the score by 5.5 mAP50-95 and 13 mAP50.** A rotated-box YOLO11s scores 55.1 /
85.1 (mAP50-95 / mAP50) on the official test split and 58.1 / 86.2 on val, but 49.6 ± 5.0 / 71.8 ± 5.5 when every
image is tested by a model that never saw its copies (5-fold group cross-validation; each run ~2 h on one GPU).
The official test score sits above every fold but one (folds: 55.8, 52.0, 50.8, 46.1, 43.1).

| Source (own classes) | official val | official test | leak-free CV, mean ± sd | change (test → CV) |
|---|---|---|---|---|
| OGSOD (tank, bridge, harbor) | 40.2 / 72.2 | 38.2 / 70.5 | 24.5 ± 3.3 / 42.0 ± 5.0 | **-13.7 / -28.5** |
| MSAR (ship, tank, bridge) | 60.6 / 88.9 | 69.8 / 94.6 | 52.7 ± 8.1 / 76.8 ± 10.1 | **-17.1 / -17.8** |
| SADD (aircraft) | 77.8 / 98.6 | 64.9 / 98.5 | 62.4 ± 18.9 / 84.4 ± 15.3 | -2.5 / -14.1 |
| AIR-SARShip (ship) | 41.9 / 70.6 | 46.0 / 78.0 | 45.4 ± 6.0 / 76.2 ± 6.4 | -0.6 / -1.8 |
| HRSID (ship) | 58.2 / 91.4 | 57.9 / 89.0 | 59.0 ± 4.7 / 90.0 ± 3.4 | +1.1 / +1.0 |
| ShipDataset (ship) | 68.3 / 97.2 | 68.9 / 97.0 | 67.6 ± 0.3 / 96.4 ± 0.4 | -1.3 / -0.6 |
| SIVED (car) | 73.1 / 98.5 | 72.8 / 98.5 | 72.4 ± 1.7 / 97.1 ± 0.9 | -0.4 / -1.4 |
| SSDD (ship) | 71.4 / 98.3 | 74.2 / 98.4 | 72.9 ± 2.0 / 98.0 ± 0.8 | -1.3 / -0.4 |
| **All (6 classes)** | **58.1 / 86.2** | **55.1 / 85.1** | **49.6 ± 5.0 / 71.8 ± 5.5** | **-5.5 / -13.3** |

- **The loss is in the scene datasets, as on SARDet-100K.** OGSOD and MSAR, whose tiles are cut from a few large
  scenes, lose 14-17 mAP50-95: the model knows the scenes' oil tanks, bridges and harbours, not tanks in general.
  Ships barely move: ShipDataset scores 67-68 in every fold, as it does on the official split.
- **It is smaller than on SARDet-100K (8.7 mAP50-95)** because RSAR drops SAR-AIRcraft, where most of that gap was.
  The two numbers come from different scorers (COCO boxes there, rotated IoU here), so compare the gaps, not the
  scores.
- **SADD is too small to say much** (44 val and 44 test images, a few airports): one fold scores 30.9, the others
  65-82.
- Per source, mAP is the mean over that source's own classes. A few sources hold one or two stray boxes of another
  class (a 'bridge' in ShipDataset, a 'ship' in SIVED and OGSOD); a held-out part that contains one scores that class
  ~0, which halves the plain per-source mean and makes it swing by 30 points between folds (the raw per-class scores
  are in `results/obb_scores_*.json`).

### Method

YOLO11s-OBB (COCO-pretrained YOLO11s backbone, the SARDet-100K recipe: 24 epochs, 512 px, batch 32), trained once on
RSAR's official train split and once per fold of the SARDet-100K 5-fold group cross-validation. Each RSAR image
takes the fold of its SARDet-100K image, so the folds are leak-free for RSAR too (~77k train, ~19k held out per fold).
Scores are ultralytics' OBB mAP50-95 and mAP50 (rotated IoU, conf 0.001, up to 300 boxes); they are comparable with
each other, not with the mmrotate numbers in the RSAR paper.

| Step | Script | What it does |
|---|---|---|
| 3 | `prepare.py` | Lays out RSAR for one run (official split or fold k) and converts its DOTA labels to the OBB format (`results/obb_prepare_*.json`) |
| 4 | `cluster/job_ad.sbatch` | Trains and scores one run (`score_obb.py`; `results/obb_scores_*.json`) |
| 5 | `summary.py` | Official val / test against the CV mean, sd, min and max, overall and per source (`results/obb_summary.json`) |

`cluster/queue_after.sh` submits the six runs and writes the summary when they finish.
