# SAR dataset audit: leakage in public SAR detection benchmarks

This repository checks whether published SAR object detection benchmarks repeat training content in their
evaluation splits. That includes exact copies, flipped copies, shifted copies, overlapping tiles and new
looks at the same place. It also measures how much that inflates reported scores.

It started from a course project on SAR detection, where 26% of the test images turned out to be flipped copies of
labelled images.

## Datasets

| Dataset | Folder | Status |
|---|---|---|
| SARDet-100K (NeurIPS 2024) | [sardet100k/](sardet100k/) | **92.1% of val and 83.3% of test have a confirmed copy in train** (a lower bound: the search finds ~80% of planted copies). YOLO11s drops from **43.5 mAP on official test to 34.7 ± 3.4 leak-free** (5-fold group cross-validation). Airports and OGSOD lose about half their score; ship datasets barely move |
| RSAR (rotated boxes on SARDet-100K images) | [rsar/](rsar/) | **94.6% of val and 95.0% of test have a confirmed copy in train.** RSAR keeps SARDet-100K's own split (same images, same names) without SAR-AIRcraft; its deduplication removed exact copies across splits only. A rotated-box YOLO11s drops from **55.1 mAP50-95 (85.1 mAP50) on official test to 49.6 ± 5.0 (71.8 ± 5.5) leak-free**. OGSOD and MSAR lose 14-17 points; ship datasets barely move |
| M4-SAR (optical and SAR pairs, arXiv 2505.10931) | [m4sar/](m4sar/) | **81.4% of val and 81.5% of test repeat a train image on the radar side** (83% counting optical copies too). Most copies are whole tiles mirrored or turned 180° (13,811 of 23,376 leaked tiles); others are exact copies, shifted crops, or the same radar acquisition with another optical date. Each tile also comes in two versions (10 m optical + VH, 60 m optical + VV), always in the same split. A rotated-box YOLO11s on SAR drops from **25.6 mAP50-95 (52.0 mAP50) on official test to 17.4 ± 1.4 (39.7 ± 2.7) leak-free**, a third of the score. Harbors and airports lose 35-47%; wind turbines and bridges barely move |

## What counts as leakage

| Type | Example | How it is found |
|---|---|---|
| Exact or re-saved copy | The same chip in train and val, maybe flipped | Full-resolution correlation under 8 flips / turns |
| Re-processed copy | The same chip with small pixel changes | Correlation after averaging out speckle |
| Shifted copy | A neighbouring crop, a few pixels up to half a tile away | Shift-proof search (\|FFT\|), then the exact shift |
| Overlapping tile | Neighbouring crops of one scene | The same, scored on the shared region only |
| Chip inside a larger image | A 256 px chip cut from a 512-1000 px image, maybe resized | Every chip-sized window searched at several scales |
| Same tile, other polarization | The same scene and moment, another channel | File-name links, checked on the pixels and boxes |
| Same target, other look | The same ship and place, new speckle or brightness | Tried (speckle-reduced background); not counted, it joins unrelated images |

Every rule is checked against luck: fake train pools that cannot contain copies, random same-source pairs, and
images corrupted on purpose. See each dataset's README.

## Layout

Each dataset folder has:
- a README (findings, method, controls, how to reproduce);
- its scripts;
- `results/` (committed);
- `data/` (download and caches, git-ignored);
- `cluster/` (GPU jobs for the SMU cluster), if it needs any.

## Environment

The first searches ran on a laptop CPU: `python3 -m venv ~/.venvs/sar-audit`, then install `kaggle six faiss-cpu
numpy pillow`. Detector scoring uses a second environment with `ultralytics pycocotools` (`~/.venvs/sar-yolo`). The
GPU searches, checks, training and scoring run on the cluster (`cluster/` in each dataset folder). The datasets have more files than the cluster's 100k-file quota, so jobs unpack a single tar on
the node's local disk.
