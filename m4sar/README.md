# M4-SAR leakage audit

**M4-SAR** (arXiv 2505.10931, optical + SAR pairs for rotated-box detection) has 112,174 image pairs (Sentinel-2
optical and Sentinel-1 SAR, 512 px, six classes: bridge, harbor, oil tank, playground, airport, wind turbine):
train 56,116, val 22,112, test 33,946. The paper does not say how the split was made or whether duplicates were
removed. Data: `M4-SAR.zip` (20.5 GB, sha256 7ca1ae75…) from the authors' Hugging Face page
(huggingface.co/datasets/wchao0601/m4-sar, CC-BY-4.0), kept packed on the cluster and read straight from the zip.

## Main findings so far (17 Nov 2026)

| | val | test |
|---|---|---|
| Images | 22,112 | 33,946 |
| **Same box list as a train image, and the same pixels** | **5,122 (23.2%)** | **7,886 (23.2%)** |

- **Every tile comes in two versions, kept together.** Image k and image k + 56,087 (10 m optical with VH SAR, and
  60 m optical with VV SAR, per the authors' README) have identical labels for all 56,087 pairs, and every pair sits
  in one split. This is not a leak, but each place counts twice in every score.
- **Beyond that, 23.2% of val and of test repeat a train image.** These images have exactly the same rotated boxes
  (3 decimals) as a train image, in groups of 4, 8 and 16 (the two versions times 2, 4 or 8 copies). On the pixels
  they are the same images: in a sample of 3,000, the median correlation with the train twin is 0.999 (optical) and
  0.998 (SAR), and 2,973 are above 0.95; a random train image with the same number of boxes gives 0.00, and 25 above
  0.95. Same SAR speckle means the same acquisition, not the same place on another date.
- The optical and SAR label files are identical for every image.

This is a lower bound: it finds only copies whose box list is unchanged. Shifted or overlapping tiles and copies
with relabelled boxes need a pixel search (still to do).

## Method

| Step | Script | What it does |
|---|---|---|
| 1 | `labeltwins.py` (`cluster/job_ae`) | Groups images by their label file (rotated boxes rounded to 3 decimals); checks the k / k + 56,087 pattern; finds val / test images whose box list is also a train image's; checks a sample of 1,500 per split on the pixels (grey 128 px correlation, optical and SAR) against a control pair (a random train image with the same number of boxes). Output: `results/labeltwins.json`, `results/label_groups.csv`, `results/twin_pixels.csv` |

## Still to do

- **Pixel search on the optical images** (no speckle), as for SARDet-100K: shifted copies, overlapping tiles,
  re-processed copies, with the same controls. The tiles were cut without overlap from 1,024 px patches, but the
  patches themselves may overlap.
- **Score inflation:** a rotated-box YOLO11s trained on the official split and on leak-free folds.
