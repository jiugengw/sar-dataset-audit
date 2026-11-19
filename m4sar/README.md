# M4-SAR leakage audit

**M4-SAR** (arXiv 2505.10931, optical + SAR pairs for rotated-box detection) has 112,174 image pairs (Sentinel-2
optical and Sentinel-1 SAR, 512 px, six classes: bridge, harbor, oil tank, playground, airport, wind turbine):
train 56,116, val 22,112, test 33,946. The paper does not say how the split was made or whether duplicates were
removed. Data: `M4-SAR.zip` (20.5 GB, sha256 7ca1ae75…) from the authors' Hugging Face page
(huggingface.co/datasets/wchao0601/m4-sar, CC-BY-4.0), kept packed on the cluster and read straight from the zip.

## Main findings (18 Nov 2026)

| | val | test |
|---|---|---|
| Images per version | 11,056 | 16,973 |
| **Radar copy in train** (what a SAR-only model sees again) | **8,995 (81.4%)** | **13,839 (81.5%)** |
| Optical or radar copy in train (optical or fusion use) | 9,204 (83.2%) | 14,172 (83.5%) |
| Same box list as a train image (label twins, below) | 2,561 (23.2%) | 3,943 (23.2%) |
| Radar copy or label twin | 9,399 (85.0%) | 14,432 (85.0%) |

- **Every tile comes in two versions, kept together.** Image k and image k + 56,087 (10 m optical with VH SAR, and
  60 m optical with VV SAR, per the authors' README) have identical labels for all 56,087 pairs, and every pair sits
  in one split. This is not a leak, but each place counts twice in every score. Every rate above holds for both
  versions.
- **About 81% of val and test repeat a training image on the radar side**, and 83% on either side. These are
  lower bounds: overlaps under 15% of the image are mostly missed, and copies are shortlisted on the optical image
  (see [Limits](#limits)).
- **Most copies are flipped or rotated.** Of the 23,376 leaked val / test tiles, 13,811 have a whole copy in train
  that is mirrored or turned 180°; 5,606 have an exact unflipped whole copy; 7,760 only overlap a train tile
  (shifted crops, 15-99% shared). It looks as if augmented copies were made before the split and spread over all
  three splits: the same pattern as the CS701 course data, which came partly from M4-SAR.
- **Same radar, another optical date.** 4,245 links match on the radar (r 0.98-1.00) but not on the optical image
  (median r 0.94): the radar image is the same acquisition, the optical image another date (different water, haze or
  light; `results/examples_radar_only.png`). For a SAR-only model these are copies.
- **Same optical, radar just under the rule.** 6,508 links match on the optical image only. Their median radar
  correlation is 0.96, just under the 0.97 rule: mostly the same acquisition, resampled. 26 have a blank (all-black)
  SAR image (`results/examples_optical_only.png`).
- **Label twins** (17 Nov): 23.2% of val and test have exactly the same rotated boxes (3 decimals) as a train
  image, in groups of 4, 8 and 16 (two versions times 2, 4 or 8 copies). In a sample of 3,000 they are the same
  pixels: median correlation 0.999 (optical) and 0.998 (SAR), against 0.00 for a random train image with as many
  boxes. The pixel search finds most of them and many more, because flipped and shifted copies have different box
  coordinates.
- The optical and SAR label files are identical for every image.
- **No zoomed copies** (19 Nov). A separate search for tiles that are a magnified part of another (zoom 1.1-2, either
  side in train; `scalesearch.py`, `cluster/job_ah`, `results/scale_search.json`) finds no val / test image whose radar
  matches a train image at another zoom: the best radar correlation is 0.925, against 0.942 for the null and >= 0.97 for
  the planted copies. 27 pairs match on the optical image only (r >= 0.97, at the minimum 30% overlap, radar r about
  0.05), where the optical null reaches 0.954: coincidences. Planted zoomed copies are found 92 / 100 (shrunk copy
  inside a train tile), 197 / 200 (enlarged crop) and 62 / 100 (shrunk copy half outside the tile) at r >= 0.97. The
  leak rates above stand.

## Score inflation (18 Nov 2026)

Rotated-box YOLO11s on the SAR images (both versions), the SARDet-100K / RSAR recipe: 24 epochs, 512 px, batch 32,
COCO-pretrained YOLO11s backbone. Leak-free CV: train on 4 folds, test on the 5th (`results/folds.csv`), mean ± sd
over the 5 folds. mAP50-95 / mAP50 (rotated IoU), `results/obb_summary.json`:

| | official val | official test | leak-free CV (min-max) | drop |
|---|---|---|---|---|
| **All (6 classes)** | 25.0 / 51.6 | **25.6 / 52.0** | **17.4 ± 1.4 / 39.7 ± 2.7** (15.9-19.6) | **-8.3 / -12.3** (-32% / -24%) |
| VH version | 25.4 / 52.2 | 26.2 / 52.6 | 17.8 ± 1.1 / 40.4 ± 2.1 | -8.4 / -12.1 |
| VV version | 24.7 / 51.1 | 25.2 / 51.4 | 17.1 ± 1.7 / 39.1 ± 3.2 | -8.2 / -12.4 |

**A third of the official mAP50-95 comes from test images the model has already seen** (flipped, shifted or exact
copies in train). Every fold scores well below the official test. Per class (mAP50-95, official test vs CV):

| class | official test | leak-free CV | drop |
|---|---|---|---|
| harbor | 44.1 | 23.6 ± 1.2 | -20.5 (-47%) |
| airport | 40.4 | 26.2 ± 6.2 | -14.2 (-35%) |
| oil tank | 8.0 | 2.1 ± 0.6 | -5.9 (-74%) |
| playground | 16.5 | 11.7 ± 1.2 | -4.9 (-29%) |
| wind turbine | 33.2 | 30.7 ± 6.4 | -2.5 (-8%) |
| bridge | 11.5 | 9.9 ± 0.3 | -1.6 (-14%) |

Harbors and airports (large, fixed scenes that repeat across tiles) lose the most; wind turbines and bridges barely
move. The absolute scores are low against RSAR (55.1) because Sentinel-1 pixels are 10 m: an oil tank is a few
pixels across. They are comparable with each other, not with the fusion models in the M4-SAR paper; we train on SAR
only, which is how the audit measures every benchmark.

## Method

| Step | Script | What it does |
|---|---|---|
| 1 | `labeltwins.py` (`cluster/job_ae`) | Groups images by their label file (rotated boxes rounded to 3 decimals); checks the k / k + 56,087 pattern; finds val / test images whose box list is also a train image's; checks a sample of 1,500 per split on the pixels against a control pair (a random train image with the same number of boxes). `results/labeltwins.json`, `label_groups.csv`, `twin_pixels.csv` |
| 2 | `pixsearch.py` (`cluster/job_af`) | Copy search over every pair of the 56,087 version-1 images, with its null and planted controls (below). About 2.2 h on an L40S |
| 3 | `summarise.py` | Link rules from the controls, leak rates, one-sided examples, groups and 5 leak-free folds. `results/search_summary.json`, `links.csv.gz`, `folds.csv`, `examples_*.png` |
| 4 | `prepare.py`, `cluster/job_ag.sbatch`, `cv_summary.py` | Lays out the SAR images for one run (official split or fold k), trains / scores YOLO11s-OBB (`../rsar/score_obb.py`), and collects the runs. `results/obb_scores_*.json`, `obb_summary.json` |
| 5 | `scalesearch.py` (`cluster/job_ah`) | Zoomed copies (one tile a magnified part of another, zoom 1.1-2), val / test against train, with its own null and planted controls (about 7 h on an L40S). `results/scale_search.json` |

**The search** (`pixsearch.py`), on the optical image, which has no speckle:
1. *Shortlist:* whole-image 32 px descriptors under the 8 flips / turns, and votes of slightly blurred 16 px patches of
   64 px thumbnails (finds shifted and partly overlapping tiles): 1,295,627 candidate pairs.
2. *Align:* 128 px, blurred, normalised cross-correlation over every shift with at least 10% shared (FFT), scored on
   the shared region only; 484,718 pairs reach r >= 0.8.
3. *Confirm:* full resolution, the optical correlation within 6 px of that shift (`r_opt`), then the SAR correlation
   at the same shift (`r_sar`), on 269,926 pairs (one of each (a, b) / (b, a)). SAR speckle is random per
   acquisition, so `r_sar` near 1 means the same radar acquisition, not only the same place.

**Link rules**, the same for `r_opt` (optical link) and `r_sar` (radar link):

| shared share of the image | rule |
|---|---|
| >= 25% | r >= 0.97 |
| 10-25% | r >= 0.99 |

**Controls** (`results/search_summary.json`):
- *False links:* the 28,029 version-1 val / test images searched against a fake train pool in which every image is cut
  into 4 x 4 blocks, shuffled, and every block turned or flipped. 10 val / test images (0.04%) get an optical link and
  7 (0.03%) a radar link, all on thin overlaps (10-12% shared), where two neighbouring blocks can survive the shuffle.
  No null pair with 25% or more shared passes (the best reaches r_opt 0.97, r_sar 0.96 at 25-40% shared, and 0.86
  above).
- *Recall:* 500 val / test images planted in the pool as re-saved (JPEG) or shifted copies, the rest of each planted
  image filled from another train image. Found with the right shift and passing the rule (optical and radar alike):

  | shared | 100% | 75% | 50% | 30% | 15% |
  |---|---|---|---|---|---|
  | found | 100 / 100 | 99 / 100 | 100 / 100 | 96 / 100 | 29 / 100 |

**Leak-free folds.** Union-find over every link of either kind, the label twins and the k / k + 56,087 pairs joins
the 112,174 images into 10,497 groups (largest 168 images). Groups are dealt largest first to the emptiest of 5 folds
(22,434-22,436 images each, every fold holding about 20% of each official split); no link crosses two folds
(`results/folds.csv`, md5 29816dc3…).

## Limits

- **Thin overlaps:** under 15% shared, recall is 29% (1 in 67 for thin strips in the trial), so the leak rate is a
  lower bound.
- **Shortlist on the optical image:** a radar copy whose optical image changed completely (clouds, another season)
  may never be shortlisted. The radar-only rate may miss these.
- **Zoom:** copies are searched at zoom 1.1-2 only, and a zoomed copy sticking half outside the other tile is found
  63 times in 100.
