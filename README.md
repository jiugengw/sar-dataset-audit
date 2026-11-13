# SAR dataset audit: leakage in public SAR detection benchmarks

This repository checks whether published SAR object detection benchmarks repeat training content in their
evaluation splits. That includes exact copies, flipped copies, wrapped-round copies, overlapping tiles and new
looks at the same place. It also measures how much that inflates reported scores.

It started from a course project on SAR detection, where 26% of the test images turned out to be flipped copies of
labelled images.

## Datasets

| Dataset | Folder | Status |
|---|---|---|
| SARDet-100K (NeurIPS 2024) | [sardet100k/](sardet100k/) | **48.6% of val and 41.5% of test confirmed as repeats of train content** (up to 71.8% / 62.7% counting uncertain matches); within a source, a trained detector scores 8-16 mAP higher on leaked images |
| RSAR (rotated boxes on SARDet-100K images) | `rsar/` (to do) | Not started. The authors say they deduplicated (GitHub issue zhasion/RSAR#28), which this audit can test |
| M4-SAR (CVPR 2025, optical and SAR pairs) | `m4sar/` (to do) | Not started. Plan: match on the optical images (no speckle) |

## What counts as leakage

| Type | Example | How it is found |
|---|---|---|
| Exact or re-saved copy | The same chip in train and val, maybe flipped | Full-resolution correlation under 8 flips / turns |
| Re-processed copy | The same chip with small pixel changes | Correlation after averaging out speckle |
| Wrapped copy | A train chip slid round, edges wrapping | Shift-proof search (\|FFT\|), then the exact shift |
| Overlapping tile | Neighbouring crops of one scene | The same, scored on the shared region only |
| Same target, other look | The same ship and place, new speckle or brightness | Speckle-reduced correlation plus label transfer |

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

The audits run on a laptop CPU: `python3 -m venv ~/.venvs/sar-audit`, then install `kaggle six faiss-cpu numpy
pillow`. Detector scoring uses a second environment with `ultralytics pycocotools` (`~/.venvs/sar-yolo`). Training
runs on the cluster. The datasets have more files than the cluster's 100k-file quota, so jobs unpack a single tar on
the node's local disk.
