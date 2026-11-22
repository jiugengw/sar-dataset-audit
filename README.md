# SAR dataset audit

How often do public SAR object detection benchmarks repeat training images in their evaluation splits, and how much
does that inflate reported scores? Companion to the CS701 project
[sar-benchmark](https://github.com/jiugengw/sar-benchmark).

## Results

| Dataset | Val / test images with a copy in train | YOLO11s mAP<sub>50:95</sub>, official test | Leak-free (5-fold group CV) |
|---|---:|---:|---:|
| [SARDet-100K](sardet100k/) | 92.1% / 83.3% | 43.5 | 34.7 ± 3.4 |
| [RSAR](rsar/) (rotated boxes) | 94.6% / 95.0% | 55.1 | 49.6 ± 5.0 |
| [M4-SAR](m4sar/) (rotated boxes, SAR only) | 81.4% / 81.5% | 25.6 | 17.4 ± 1.4 |

Leak rates are lower bounds. Sources cut from a few large scenes (airports, harbors, OGSOD) lose up to half their
score; ship collections barely move. In M4-SAR most copies are whole tiles mirrored or turned 180°.

## Method

1. **Find copies:** compare every evaluation image with every training image under the 8 flips and quarter turns,
   with shift-proof search for crops and overlapping tiles, chip search for images cut from larger ones, and
   file-name links between polarizations.
2. **Check the rules:** rerun against fake training pools that cannot contain copies (false links) and against
   planted copies (missed links).
3. **Leak-free evaluation:** join linked images into groups and run 5-fold group cross-validation with a fixed
   YOLO11s recipe, so every image is tested by a model that never saw its group.

Details, controls and commands are in each dataset's README.

## Layout

Each dataset folder holds its README, scripts, `results/` and `cluster/` (Slurm jobs). Downloads and caches go in
`data/`, which is git-ignored. Searches need `numpy pillow faiss-cpu`; detector scoring needs
`ultralytics pycocotools`.
