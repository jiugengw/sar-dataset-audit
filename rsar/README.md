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

## Score inflation (queued, 16 Nov)

The same test as for SARDet-100K, with a rotated-box detector: YOLO11s-OBB (COCO-pretrained YOLO11s backbone, the
SARDet-100K recipe: 24 epochs, 512 px, batch 32), trained once on RSAR's official split and once per fold of the
SARDet-100K 5-fold group cross-validation. Each RSAR image takes the fold of its SARDet-100K image, so the folds are
leak-free for RSAR too. Scores are ultralytics' OBB mAP50-95 and mAP50 (rotated IoU), overall and per source; they
are comparable with each other, not with the mmrotate numbers in the RSAR paper.

| Step | Script | What it does |
|---|---|---|
| 3 | `prepare.py` | Lays out RSAR for one run (official split or fold k) and converts its DOTA labels to the OBB format |
| 4 | `cluster/job_ad.sbatch` | Trains and scores one run (`score_obb.py`): `runs/rsar-obb-<fold>-<jobid>/scores.json` |
| 5 | `summary.py` | Official val / test against the CV mean, sd, min and max, overall and per source |

`cluster/queue_after.sh` submits the six runs and writes the summary when they finish. Without SAR-AIRcraft, the sources left are mostly those whose scores barely moved on
SARDet-100K, apart from OGSOD and SADD, so the gap is expected to be smaller than SARDet-100K's 9 points.
