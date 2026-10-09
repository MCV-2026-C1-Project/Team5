# Team 5 - Museum painting retrieval (C1)

Query-by-Example retrieval: given a photo of a painting, return the most similar
paintings of the museum collection (BBDD). Images are described with colour
histograms and compared with a histogram distance. From week 2 the background of the
queries can also be removed using only colour.

## Structure

```
src/        shared code: descriptors, distances, retrieval, evaluation, masks
week1/      week 1 scripts (evaluation, submission, error analysis)
week2/      week 2 scripts (descriptor search, background removal)
data/       datasets (not committed)
outputs/    results (not committed)
```

The code of the week 1 submission is also tagged as `w1-submission`.

Run every script **from the root of the repository**: the data and output paths
(`data/`, `outputs/`) are relative to where the command is run.

## Setup

Tested with Python 3.13.7.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download the data from <!-- TODO: add link --> and unzip it so that the repo looks like this:

```
data/
├── BBDD/        bbdd_00000.jpg ... bbdd_00286.jpg (+ .png, .txt, relationships.pkl)
├── qsd1_w1/     00000.jpg ... 00029.jpg, gt_corresps.pkl
├── qst1_w1/     00000.jpg ...            (test set, no ground truth)
├── qsd1_w2/     00000.jpg ... 00029.jpg, gt_corresps.pkl
└── qsd2_w1/     00000.jpg ... 00029.jpg with a mask 00000.png ... (the painting is white),
                 frames.pkl, gt_corresps.pkl
```

Only the `.jpg` files of BBDD are used. In `qsd2_w1` the `.png` files are the
annotated foreground masks.

## Descriptors and measures

**Descriptor.** The image is converted to one colour space (gray, RGB, HSV, Lab or
YCrCb) and a histogram with N bins is computed for each channel over its full range
(OpenCV 8-bit: H in [0, 180), everything else in [0, 256)). Each histogram is divided by
its sum and the concatenation is divided by the number of histograms, so the descriptor
sums to 1 and does not depend on the image size. Three options change the structure:

- *Blocks (`grid=G`).* The image is split into G x G blocks and the histograms of every
  block are concatenated, so the descriptor also knows where each colour is. G=1 is the
  whole image.
- *Spatial pyramid (`grid=(1, 2, 4)`).* The descriptors of several grid levels, for
  example the whole image, 2 x 2 and 4 x 4 blocks, are concatenated. Each level sums to 1
  and counts the same unless `level_weights` is given.
- *Joint histograms (`joint=2` or `3`).* Instead of one 1D histogram per channel, `joint=2`
  computes one 2D histogram for each pair of channels and `joint=3` one 3D histogram of the
  three channels together. They know which values occur together, but have bins^2 and bins^3
  values, so they need few bins.

**Normalisation (`normalize`, Lab and YCrCb only).** The queries are the paintings
with a global change of brightness and colour. Before the histogram, each image is
normalised in the chosen colour space: every channel is shifted to mean 128, and all
three are scaled by the factor that brings the brightness channel (L or Y) to standard
deviation 50. This keeps the shape of the histograms and removes the change of light.

Two details matter for very dark queries (e.g. query 26 of qsd1_w1, about 3 times
darker than its painting):

- *The colour channels are scaled too, not only shifted.* Less light shrinks
  brightness and colour differences by the same amount, so a dark photo looks almost
  grey. Scaling the colour channels by the brightness factor brings the colour back.
- *The colour conversion is done in float.* In a very dark photo the colour channels
  vary by less than one unit; the usual 8-bit conversion rounds them to a single value
  before they can be stretched. Converting the float image keeps those differences.

**Measures.** Euclidean, L1, chi-squared, EMD and Jensen-Shannon are distances; histogram
intersection, cosine, and the Hellinger kernel are similarities. The convention in the code
is "lower = more similar": similarities are negated, so retrieval always sorts in ascending
order (with a stable sort, so ties are reproducible). EMD is only valid for the 1D
per-channel histograms.

**Metric.** mAP@K, the mean over the queries of the average precision at K. With one
correct painting per query, AP@K is 1/rank if the correct painting is in the top K and
0 otherwise. Bootstrap resampling can be used to compute 95% confidence intervals across queries.

## Week 1

### Run

With no arguments, `evaluate.py` prints the table of every colour space and measure
(64 bins, no normalisation) and `make_submission.py` uses the final method.

```bash
# Tasks 1-3: mAP@1 and mAP@5 on qsd1_w1, every colour space and measure
python week1/evaluate.py
python week1/evaluate.py --data data --query-set qsd1_w1 --color-space lab --bins 64 --normalize --measure all

# Block histograms: split the image into a 2 x 2 grid
python week1/evaluate.py --color-space lab --bins 64 --grid 2 --normalize --measure all

# Evaluate with 95% bootstrap confidence intervals (e.g. 1000 resamples)
python week1/evaluate.py --color-space lab --measure all --bootstrap 1000

# Where does the method fail? Lists the missed queries and plots the worst ones
python week1/analyze_errors.py --data data --color-space lab --bins 64 --grid 2 --measure l1

# Task 4: top-10 results for the test set, saved to outputs/result.pkl
python week1/make_submission.py
python week1/make_submission.py --data data --query-set qst1_w1 --color-space lab --bins 64 --normalize --measure l1 --output outputs/result.pkl
```

`outputs/result.pkl` is a pickled list with one entry per query (in filename order);
each entry is the list of the 10 best BBDD IDs as integers (7 means `bbdd_00007.jpg`).

### Final methods

| | Descriptor | mAP@1 / mAP@5 on qsd1_w1 |
|---|---|---|
| Method 1 | Lab, 64 bins, 2 x 2 grid, normalised, L1 distance | 0.9667 / 0.9667 |
| Method 2 | YCrCb, 64 bins, no grid, normalised, chi-squared distance | 0.9000 / 0.9278 |

```bash
# Method 1
python week1/make_submission.py --data data --query-set qst1_w1 --color-space lab --bins 64 --grid 2 --normalize --measure l1 --k 10 --output outputs/Team5/week1/QST1/method1/result.pkl

# Method 2
python week1/make_submission.py --data data --query-set qst1_w1 --color-space ycrcb --bins 64 --grid 1 --normalize --measure chi2 --k 10 --output outputs/Team5/week1/QST1/method2/result.pkl
```

On the blind test set QST1 (results published by the teachers): method 1 0.93 / 0.95,
method 2 0.90 / 0.91.

On qsd1_w1 normalising raised the best single-image result (grid 1) from mAP@5 0.87 (Lab,
64 bins, L1) to 0.93 (Lab, 64 bins, L1; and YCrCb, 64 bins, chi-squared, which also moves
query 26 from rank 57 to rank 3). With `--grid 2` almost every setting already reaches
29/30 queries at rank 1, so the gain there is small (mAP@5 0.967 to 0.975 with YCrCb).

## Week 2

### Task 1-2: do the new histograms improve the Week 1 methods?

`week2/evaluate_descriptors.py` takes the two Week 1 methods in their simplest form (whole
image, 1D histograms, 64 bins: Method 1 is Lab with L1, Method 2 is YCrCb with chi-squared,
both normalised) and changes only the histogram structure: block grids (2x2, 3x3, 4x4),
spatial pyramids, and 2D and 3D joint histograms, at several bin counts. That is 37 settings
per method; each image is read and converted once.

```bash
python week2/evaluate_descriptors.py --data data --query-set qsd1_w2
python week2/evaluate_descriptors.py --data data --query-set qsd1_w2 --all-measures --workers 2
```

`--all-measures` also tries L1, chi-squared, Hellinger and Jensen-Shannon on every setting;
`--workers` is the number of images processed in parallel. The output is
`outputs/week2/descriptor_search.txt` (every setting, sorted by mAP@5, with the change
against the baseline) and `outputs/week2/descriptor_comparison.png`.

Results on qsd1_w2 (the same queries as qsd1_w1), one query is 0.033 of mAP@1:

| | Baseline (1D, whole image, 64 bins) | Best settings |
|---|---|---|
| Method 1 (Lab, L1) | 0.9333 / 0.9333 | 0.9667 / 0.9667 with 2x2, 3x3 or 4x4 blocks, or the pyramids (1,2), (1,2,3), (1,2,4) |
| Method 2 (YCrCb, chi-squared) | 0.9000 / 0.9278 | 0.9667 / 0.9750 with 2x2 blocks and 64 bins; 0.9667 / 0.9667 for the others |

The baselines are the Week 1 numbers. In both methods blocks and pyramids add one or two
correct queries and many settings reach the same score, so the gain comes from keeping the
spatial layout, not from a particular grid.

### Task 3-4: background removal and mask evaluation

`week2/evaluate_masks.py` removes the background of `qsd2_w1` using only colour (no
contour or object detectors) and compares the result with the annotated masks using
precision, recall and F1 (per image, then averaged over the 30 images).

```bash
python week2/evaluate_masks.py --data data --query-set qsd2_w1
python week2/evaluate_masks.py --data data --query-set qsd2_w1 --top 6 --border 0.04
```

Three methods are compared, all followed by the same clean-up (close small gaps, fill
holes, remove specks, keep the largest region):

- `gray_otsu`: Otsu threshold on the grayscale image. Baseline, it ignores colour.
- `border_otsu`: a Gaussian colour model of the wall is estimated in Lab from thin strips
  along the four borders of the image (outlier pixels are discarded). Each pixel gets its
  Mahalanobis distance to that colour, and Otsu is applied to the distance map. It assumes
  the painting does not touch the border of the photo.
- `kmeans`: k-means with 2 clusters in Lab; the background is the cluster that dominates
  the border.

Each one is also run with the convex hull of the largest region (`+hull`), computed from
the mask pixels with `cv2.convexHull`; it is shape post-processing of the colour mask, not
a contour detector.

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| border_otsu+hull | 0.9677 | 0.9153 | 0.9181 |
| border_otsu | 0.9765 | 0.8948 | 0.9084 |
| kmeans+hull | 0.8753 | 0.9203 | 0.8852 |
| gray_otsu+hull | 0.8594 | 0.9230 | 0.8799 |
| kmeans | 0.8993 | 0.8658 | 0.8699 |
| gray_otsu | 0.8666 | 0.8494 | 0.8491 |

With `border_otsu`, 3 images are below 0.8 (13, 17 and 20); the rest are above 0.9. Image 17
fails with every method. A background model that changes smoothly across the image and a
rule that treats darker versions of the wall colour as shadow were also tried and gave no
improvement, so they were removed. The script writes `outputs/week2/mask_results.txt` (F1 of
every image), `mask_metrics.png` and one `masks_<variant>.png` for each of the best variants,
where green is correct foreground, red false positive and blue missed foreground.

### Still to do

Retrieval on the queries with background (Task 5) and the blind submissions for QST1 and QST2
(Task 6).