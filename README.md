# Team 5 - Museum painting retrieval (C1)

Query-by-Example retrieval: given a photo of a painting, return the most similar
paintings of the museum collection (BBDD). Images are described with colour
histograms and compared with a histogram distance. From week 2 the background of the
queries can also be removed using only colour.

## Structure

```
src/        shared code: descriptors, distances, retrieval, evaluation, masks, watershed, denoising, texture
week1/      week 1 scripts (evaluation, submission, error analysis)
week2/      week 2 scripts (descriptor search, background removal, retrieval with masks)
week3/      week 3 scripts (noise detector calibration, noise filtering, texture descriptors, painting
            detection, retrieval, submission)
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
├── qsd2_w1/     00000.jpg ... 00029.jpg with a mask 00000.png ... (the painting is white),
│                frames.pkl, gt_corresps.pkl
├── qsd1_w3/     00000.jpg ... 00029.jpg, gt_corresps.pkl, augmentations.pkl (what was done to
│                each query: impulse noise, hue change or nothing), non_augmented/ (clean versions)
└── qsd2_w3/     same, with background and up to two paintings per image
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

**Mask (`mask=`).** The descriptor functions accept an optional mask (uint8, non-zero = painting):
only the masked pixels are counted in the histograms, and the normalisation uses their mean and
standard deviation. A block without masked pixels gives a histogram of zeros. Without a mask nothing
changes.

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

# Only the best setting of each method (the "Best settings" column below), next to its baseline
python week2/evaluate_descriptors.py --data data --query-set qsd1_w2 --color-space ycrcb --bins 32 --grid 1 2 4 --joint 1 --measure chi2
python week2/evaluate_descriptors.py --data data --query-set qsd1_w2 --color-space lab --bins 64 --grid 1 2 4 --joint 1 --measure l1
```

`--all-measures` also tries L1, chi-squared, Hellinger and Jensen-Shannon on every setting;
`--workers` is the number of images processed in parallel. `--color-space`, `--bins`, `--grid`
(one value for blocks, several for a pyramid), `--joint` and `--measure` restrict the search to the
matching settings; the result is only printed, so the files of the full search are kept. The output is
`outputs/week2/descriptor_search.txt` (every setting, sorted by mAP@5, with the change
against the baseline) and `outputs/week2/descriptor_comparison.png`.

Results on qsd1_w2 (the same queries as qsd1_w1), one query is 0.033 of mAP@1:

| | Baseline (1D, whole image, 64 bins) | Best settings |
|---|---|---|
| Method 1 (Lab, L1) | 0.9333 / 0.9333 | 0.9667 / 0.9667 with the 1D pyramid (1,2,4) and 64 bins |
| Method 2 (YCrCb, chi-squared) | 0.9000 / 0.9278 | 0.9667 / 0.9667 with the 1D pyramid (1,2,4) and 32 bins |

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
python week2/evaluate_masks.py --data data --query-set qsd2_w1 --min-region 0.05
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

With `--min-region F` every variant is also run keeping every region with at least F times
the area of the largest one, instead of only the largest (`+regions`). This keeps paintings
made of several panels, or whose frame has the colour of the wall, which the colour model
splits into separate regions. Without the option the script behaves as before. Results with
`--min-region 0.05` (0.10 and 0.20 give almost the same):

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| border_otsu+regions+hull | 0.9677 | 0.9477 | 0.9431 |
| border_otsu+regions | 0.9765 | 0.9168 | 0.9291 |
| border_otsu+hull | 0.9677 | 0.9153 | 0.9181 |
| border_otsu | 0.9765 | 0.8948 | 0.9084 |
| kmeans+hull | 0.8753 | 0.9203 | 0.8852 |
| gray_otsu+hull | 0.8594 | 0.9230 | 0.8799 |
| kmeans | 0.8993 | 0.8658 | 0.8699 |
| gray_otsu | 0.8666 | 0.8494 | 0.8491 |

With `border_otsu`, 3 images are below 0.8 (13, 17 and 20); the rest are above 0.87. Images 13
and 17 fail because only the largest region is kept: 13 is split into panels by a frame of the
same colour as the wall, and 17 is four panels separated by strips of wall. With
`border_otsu+regions+hull` they go to 0.88 and 0.98 and only image 20 stays below 0.8 (0.14):
the painting is white and grey like the wall, so colour alone cannot separate them. Whether
the convex hull counts as "only colour" should be checked; without it the best variant is
`border_otsu+regions` (F1 0.929). A background model that changes smoothly across the image and a
rule that treats darker versions of the wall colour as shadow were also tried and gave no
improvement, so they were removed. The script writes `outputs/week2/mask_results.txt` (F1 of
every image), `mask_metrics.png` and one `masks_<variant>.png` for each of the best variants,
where green is correct foreground, red false positive and blue missed foreground.

### Task 5: retrieval on the queries with background

`week2/evaluate_retrieval_masks.py` describes each query of `qsd2_w1` using only the pixels of its
mask: the query is cropped to the bounding box of the mask, the normalisation and the histograms use
the masked pixels only, and the museum images are described as usual. It assumes one painting per
query, as in `qsd2_w1`. Three descriptors are compared (all normalised, 1D histograms): M1 is YCrCb,
pyramid (1,2,4), 32 bins, chi-squared; M2 is Lab, pyramid (1,2,4), 64 bins, L1; M3 is Lab, 2x2 blocks,
64 bins, L1 (Week 1 method 1). Two references are included: the whole photo, without removing the
background, and the annotated mask.

```bash
python week2/evaluate_retrieval_masks.py --data data --query-set qsd2_w1
python week2/evaluate_retrieval_masks.py --data data --query-set qsd2_w1 --methods border_otsu
```

`--methods` selects the mask methods to try (default: all) and `--min-region` is the fraction used by
the `+regions` variants. The output is `outputs/week2/retrieval_masks.txt`, with the table and the rank
of the correct painting of every query. mAP@1 / mAP@5 on qsd2_w1 (one query is 0.033 of mAP@1):

| Masks | Mask F1 | M1 | M2 | M3 |
|---|---|---|---|---|
| Whole photo (reference) | 0.7021 | 0.1667 / 0.1667 | 0.2333 / 0.2417 | 0.1667 / 0.1928 |
| Annotated mask (reference) | 1.0000 | 0.9000 / 0.9194 | 0.8333 / 0.8611 | 0.8333 / 0.8567 |
| border_otsu | 0.9084 | 0.7667 / 0.7833 | 0.7333 / 0.7556 | 0.6667 / 0.7000 |
| border_otsu+hull | 0.9181 | 0.8000 / 0.8167 | 0.7667 / 0.7944 | 0.7000 / 0.7483 |
| border_otsu+regions | 0.9291 | 0.8000 / 0.8167 | 0.7667 / 0.7778 | 0.7000 / 0.7333 |
| border_otsu+regions+hull | 0.9431 | 0.8333 / 0.8500 | 0.7667 / 0.7944 | 0.7000 / 0.7483 |

The best `gray_otsu` and `kmeans` variants score lower than the best `border_otsu` variant with the
three descriptors.

### Task 6: blind competition submissions (QST1 and QST2)

`week2/make_submission.py` generates the submission files for the blind competition sets
`qst1_w2` and `qst2_w2` according to the required directory structure and format:

- For each test query in `qst1_w2` and `qst2_w2`, returns a pickled list of lists containing
  the top-10 best matching BBDD IDs (`result.pkl`).
- For `qst2_w2`, saves the predicted binary foreground masks as `.png` files (e.g. `00000.png` ... `00029.png`).
- Output follows the required GDrive layout:
  `outputs/Team5/week2/QST1/method1/result.pkl`
  `outputs/Team5/week2/QST1/method2/result.pkl`
  `outputs/Team5/week2/QST2/method1/result.pkl`
  `outputs/Team5/week2/QST2/method1/*.png`
  `outputs/Team5/week2/QST2/method2/result.pkl`
  `outputs/Team5/week2/QST2/method2/*.png`

```bash
# QST1 (no background)
# Method 1: YCrCb, pyramid (1,2,4), 32 bins, chi-squared
python week2/make_submission.py --data data --query-set qst1_w2 --color-space ycrcb --bins 32 --grid 1 2 4 --normalize --measure chi2 --output outputs/Team5/week2/QST1/method1/result.pkl

# Method 2: Lab, pyramid (1,2,4), 64 bins, L1
python week2/make_submission.py --data data --query-set qst1_w2 --color-space lab --bins 64 --grid 1 2 4 --normalize --measure l1 --output outputs/Team5/week2/QST1/method2/result.pkl

# QST2 (with background - saves result.pkl and the .png masks)
# Method 1: YCrCb, pyramid (1,2,4), 32 bins, chi-squared; border_otsu + regions + hull
python week2/make_submission.py --data data --query-set qst2_w2 --color-space ycrcb --bins 32 --grid 1 2 4 --normalize --measure chi2 --mask border_otsu --min-region 0.05 --hull --output outputs/Team5/week2/QST2/method1/result.pkl

# Method 2: Lab, pyramid (1,2,4), 64 bins, L1; border_otsu + regions + hull
python week2/make_submission.py --data data --query-set qst2_w2 --color-space lab --bins 64 --grid 1 2 4 --normalize --measure l1 --mask border_otsu --min-region 0.05 --hull --output outputs/Team5/week2/QST2/method2/result.pkl
```
## Week 3

### Task 1: noise filtering

`src/denoise.py` has the filters and a noise detector; `week3/evaluate_denoising.py` evaluates them on
`qsd1_w3`, using the clean version of every query (`non_augmented/`) and the annotation of what was done
to it (`augmentations.pkl`).

```bash
# Choose the thresholds of the noise detector on the clean museum images, test them on the queries
python week3/calibrate_noise_detector.py --data data
# Compare the filters
python week3/evaluate_denoising.py --data data --query-set qsd1_w3
```

**The noise.** 7 of the 30 queries have impulse noise: about 1.5-2.7% of the pixels are replaced by
random colours (14 others have a hue change, 9 are untouched). The noise model is not given, so it was
measured against the clean versions.

**Detecting it.** Only some images are noisy, and filtering a clean image only blurs it, so a filter is
applied only when noise is detected. An impulse differs from all of its 8 neighbours, while a pixel on an
edge or in a texture is always close to some of them. A pixel is an impulse if its smallest difference
to its neighbours is above `DETECT_DIFF` in some channel, and the image is noisy if more than
`NOISY_RATIO` of its pixels are impulses. A plain "far from the local median" test does not work: the
edges and textures of clean photos score as high as the noise.

**Choosing the thresholds without the query annotations.** `week3/calibrate_noise_detector.py` uses only
the 287 museum images, which are clean (`outputs/week3/noise_calibration.txt`):

- `DETECT_DIFF` is the smallest of 30, 40, ..., 80 at which 99% of the clean images have fewer than 1
  impulse per 10,000 pixels, so an "impulse" is something that almost never happens in a clean painting: **70**.
- `NOISY_RATIO` is twice the largest impulse ratio of any clean image: **0.00018** (0.018%).

The query sets were not used to choose anything, so they are a real test: the detector finds exactly the
noisy images in both, with no false alarms and none missed.

| Query set | Correct | Impulse ratio, noisy images | Impulse ratio, the others |
|---|---|---|---|
| qsd1_w3 (7 noisy) | 30/30 | 0.34-1.11% | 0.00% |
| qsd2_w3 (9 noisy) | 30/30 | 0.27-0.76% | 0.00-0.01% |

**The filters**. 

| Filter | Noisy queries vs clean version: PSNR / SSIM | Other queries vs themselves, if filtered anyway: PSNR / SSIM |
|---|---|---|
| no filter | 23.73 / 0.488 | - / 1.000 |
| gaussian 3x3 (linear) | 29.12 / 0.740 | 35.05 / 0.955 |
| **median 3x3** | **30.78 / 0.797** | **36.09** / 0.946 |
| median 5x5 | 30.27 / 0.771 | 31.47 / 0.864 |
| bilateral | 26.37 / 0.653 | 34.22 / 0.897 |
| non-local means | 27.16 / 0.710 | 32.97 / 0.882 |

The first column is how well the noise is removed; the second is how much a clean query would be damaged
if it were filtered by mistake.

**Size of the median filter** (also in `denoise_results.txt`):

| Size | 3x3 | 5x5 | 7x7 | 9x9 | 11x11 | 13x13 | 15x15 |
|---|---|---|---|---|---|---|---|
| Noisy queries vs clean: PSNR | **30.78** | 30.27 | 29.42 | 28.72 | 28.11 | 27.54 | 27.03 |
| Noisy queries vs clean: SSIM | **0.797** | 0.771 | 0.732 | 0.705 | 0.685 | 0.671 | 0.659 |
| Other queries vs themselves: PSNR | **36.09** | 31.47 | 28.99 | 27.36 | 26.22 | 25.49 | 24.94 |
| Other queries vs themselves: SSIM | **0.946** | 0.864 | 0.796 | 0.748 | 0.716 | 0.697 | 0.684 |

3x3 already removes the impulses, which are isolated pixels, so a larger window removes no more noise and
only blurs the painting: both metrics fall steadily with the size, on the noisy queries and on the clean
ones. From 7x7 on, a filtered clean query is further from itself than a filtered noisy query is from its
clean version.

**Also tried: a switching median**, which replaces only the pixels detected as noise and leaves the rest
untouched, to keep the texture of the painting. It is worse than the plain median whatever the rule
(pixels far from all their neighbours, or far from their local median, with or without a 1-pixel ring
around them): the quality keeps rising the more pixels are replaced, up to replacing all of them, which
is the plain median (30.78 dB / 0.797). The JPEG compression has spread the noise over the whole image,
not only over the impulse pixels, so it was not kept.

### Task 2: texture descriptors

`src/texture.py` has the descriptors; `week3/evaluate_texture.py` retrieves on `qsd1_w3` with texture
descriptors only, on the queries as given, filtered with each filter of Task 1 (only the ones detected as
noisy: the whole Task 1 + 2 pipeline) and on the clean originals (the upper bound).

```bash
python week3/evaluate_texture.py --data data --query-set qsd1_w3
```

All of them work on the grey-level image resized to 256 x 256, so a photo and the museum image are
described at the same scale, and they ignore colour, so the hue changes of `qsd1_w3` do not affect them.
As with the colour histograms, the image is split into blocks and the descriptors of the blocks are
concatenated.

- **LBP** (radius 1, 4x4 blocks): each of the 8 neighbours on a circle gives a bit (1 if at least as bright
  as the centre), the 8 bits a code from 0 to 255; histogram of the codes per block (4,096 values).
- **Multiscale LBP** (radii 1, 2 and 3, 4x4 blocks): the same on three circles, with bilinear
  interpolation for the neighbours that do not fall on a pixel (12,288 values).
- **DCT** (8x8 blocks, 10 coefficients): the first 10 coefficients in zigzag order (lowest frequencies) of
  every block, without the DC coefficient (the mean brightness); unit length (640 values).
- **Haar wavelet** (3 levels, 4x4 blocks): mean absolute value of the horizontal, vertical and diagonal
  details at each level, per block (144 values).

Best distance and filter of each descriptor (highest mAP@5, then mAP@1, on the 30 queries; every
combination is in `outputs/week3/texture_results.txt`). "Noisy @1" is how many of the 7 noisy queries
have the correct painting at rank 1:

| Descriptor | Best distance | Best filter (on the queries detected as noisy) | mAP@1 | mAP@5 | Noisy @1 |
|---|---|---|---|---|---|
| LBP | L1 | bilateral | 0.867 | 0.919 | 7/7 |
| Multiscale LBP | L1 | bilateral | 1.000 | 1.000 | 7/7 |
| DCT | cosine (L1 equal) | Every filter gives the same | 0.967 | 0.983 | 7/7 |
| Haar wavelet | L1 | median 3x3 | 0.967 | 0.967 | 6/7 |

### Task 3: up to two paintings per image

`segment_paintings` in `src/masks.py` returns one mask per painting (at most two), in reading order
(left to right, top to bottom if one is above the other). `week3/evaluate_masks.py` filters the noisy
images first (Task 1, so the noise does not spoil the wall model), segments them and compares the union
of the paintings with the annotated mask (precision, recall, F1), and the number of paintings found with
the ground truth.

```bash
python week3/evaluate_masks.py --data data/qsd2_w3 --query-set qsd2_w3
python week3/evaluate_masks.py --data data/qsd2_w3 --query-set qsd2_w3 --save-masks   # also writes the masks as .png
```

It starts from the Week 2 method (wall colour learnt from the borders, Mahalanobis distance, Otsu,
closing, fill holes, opening) and changes four things:

- **Otsu on log(1 + distance)** (`border_otsu_log`). With two paintings, Otsu on the distance put the
  threshold between the wall and the most different painting, and a dark or greyish second painting fell
  below it. The log compresses the large distances, so the threshold falls in the valley between the wall
  and everything that is not wall.
- **Opening of 5% of the shorter side** before splitting, to cut thin bridges (a rail or a line on the
  wall) that joined two paintings into one region.
- **Grouping instead of the largest region.** Regions smaller than 1% of the image or 5% of the largest
  one are dropped (labels, specks); regions whose bounding boxes are closer than 3% of the longer side are
  one painting (the panels of one work); the two largest groups are kept. The convex hull is taken per
  painting, so the wall between two paintings is never filled.

- **A pixel further than 4 from the wall is always painting** (`WALL_LIMIT`; 99.9% of a 3D Gaussian is
  closer). Otsu only splits two groups: in image 9 of `qsd2_w3` the wall, the black of the painting and
  its red are three, and Otsu cut between the black and the red, so the painting was lost and split in
  two. Any limit between 3 and 4 gives the same.

Results (the noisy images are filtered first; without the filter the masks are the same, 0.9654):

| Set | Method | Precision | Recall | F1 | Worst F1 | Paintings counted right |
|---|---|---|---|---|---|---|
| **qsd2_w3** (39 paintings, 9 images with two) | Week 2: border_otsu+hull | 0.9790 | 0.9043 | 0.9292 | 0.56 | 28/30 |
| | border_otsu_log, no hull | 0.9757 | 0.9487 | 0.9579 | 0.76 | 30/30 |
| | **border_otsu_log + hull** | 0.9660 | 0.9690 | **0.9652** | 0.82 | **30/30** |
| qsd2_w1 (one painting each) | Week 2: border_otsu+regions+hull | 0.9677 | 0.9477 | 0.9431 | 0.14 | 30/30 |
| | **border_otsu_log + hull** | 0.9396 | 0.9885 | **0.9617** | 0.81 | 30/30 |

The two paintings are always returned in the order of the ground truth (left to right). The worst images
of `qsd2_w3` are 21 (0.82), a white canvas on a white wall, and 3 (0.84), whose silver frame has the colour
of the wall and is lost.

Before having `qsd2_w3`, the splitting was also checked on a synthetic set (the painting of another
`qsd2_w1` query pasted on an extension of the wall of each photo): 29/30 images with both paintings found,
all in the right order; the miss is the white painting of query 20 pasted small on a white wall.

**Watershed from the Otsu masks** (`watershed_masks` in `src/watershed.py`). The Otsu masks above decide how
many paintings there are and roughly where, but their boundary is wherever the distance to the wall crosses
the threshold. Watershed keeps the count and the position and moves the boundary: the core of each Otsu
painting (its mask shrunk by 15% of its bounding box) is marked as sure painting, a strip of 3% along the
image border as sure wall, and OpenCV's watershed grows them over the image (600 px on the longer side)
until they meet, which happens where the image changes most, usually the outer edge of the frame. Each
painting is then filled and replaced by its convex hull, as above. `week3/evaluate_masks.py` runs it as the
last variant, so its per-image table and `--save-masks` are about it.

| Set | Method | Precision | Recall | F1 | Worst F1 | Paintings counted right |
|---|---|---|---|---|---|---|
| qsd2_w3 | border_otsu_log + hull | 0.9660 | 0.9690 | 0.9652 | 0.82 | 30/30 |
| | **watershed (Otsu seeds)** | **0.9805** | 0.9620 | **0.9698** | **0.87** | **30/30** |
| qsd2_w1 | border_otsu_log + hull | 0.9396 | 0.9885 | 0.9617 | 0.81 | 30/30 |
| | watershed (Otsu seeds) | 0.9403 | 0.9843 | 0.9600 | 0.82 | 30/30 |

On `qsd2_w3` it mostly removes wall from the masks (precision 0.966 to 0.981) and fixes image 21, the white
canvas on the white wall (0.82 to 0.996); the worst image is now 3 (0.87). On `qsd2_w1` it is equal.



### Task 4: retrieval with up to two paintings per image

`src/painting_retrieval.py` is the whole pipeline for one photo: filter it if it is noisy (Task 1), find
the paintings (Task 3), crop each one to the bounding box of its mask, describe it and rank the museum
images. The result has one ranked list per painting, in reading order (`[[7, 25, ...], [44, 81, ...]]`).
`week3/evaluate_retrieval.py` evaluates it per painting (`mapk_paintings` in `src/evaluation.py`: every
painting of the ground truth is one query; a painting not found counts 0, an extra one is ignored).

```bash
python week3/evaluate_retrieval.py --query-dir data/qsd2_w3/qsd2_w3
python week3/evaluate_retrieval.py --query-dir data/qsd1_w3/qsd1_w3 --no-segment   # cropped queries
```

Two things make it work:

- **A 10% margin** is cut from every side of the crops and of the museum images. The edge of a crop is
  where the mask errors are (a strip of wall, part of the frame), and the block descriptors shift with it:
  DCT goes from 0.67 with our masks (0.95 with the annotated frames) to 0.79 with the margin.
- **Several descriptors together.** Each one fails on different paintings, so their distances are put on
  the same scale (shifted to 0 and divided by their median over the museum) and averaged: multiscale LBP,
  DCT, Haar wavelet (Task 2) and the Week 2 colour histogram (YCrCb pyramid 1-2-4, 32 bins, chi-squared, on
  the masked pixels). Colour alone suffers from the hue changes, texture alone from the crop, together they
  find every painting.

mAP@1 / mAP@5 per painting, margin 10% (`outputs/week3/retrieval_qsd2_w3.txt`, `retrieval_qsd1_w3.txt`):

| Descriptors | qsd2_w3, our masks (39 paintings) | qsd2_w3, annotated frames | qsd1_w3 (30, cropped) |
|---|---|---|---|
| multiscale LBP | 0.846 / 0.859 | 0.872 / 0.877 | 1.000 / 1.000 |
| DCT | 0.795 / 0.842 | 0.974 / 0.974 | 1.000 / 1.000 |
| Haar wavelet | 0.949 / 0.962 | 0.949 / 0.957 | 1.000 / 1.000 |
| colour (Week 2) | 0.821 / 0.843 | 0.897 / 0.910 | 1.000 / 1.000 |
| multiscale LBP + DCT + colour | 0.974 / 0.987 | 0.974 / 0.987 | 1.000 / 1.000 |
| **multiscale LBP + DCT + wavelet + colour** | **1.000 / 1.000** | 1.000 / 1.000 | **1.000 / 1.000** |

Without the margin the same combination gives 0.923 / 0.928 on `qsd2_w3`, and with 5% or 15% 0.923 and
0.949: 10% was chosen on these two sets, so the test sets may score lower. At every margin the
combinations of three or four descriptors beat any descriptor alone (at 5%, DCT + wavelet + colour is the
best with 0.974).

**With watershed** (`--extractor watershed` in `evaluate_retrieval.py` and in the submission script; the
default is still the Otsu masks):

```bash
python week3/evaluate_retrieval.py --query-dir data/qsd2_w3/qsd2_w3 --extractor watershed --output-dir outputs/week3/watershed
python week3/evaluate_retrieval.py --query-dir data/qsd2_w3/qsd2_w3 --extractor watershed --margin 0.05 --output-dir outputs/week3/watershed_005
```

mAP@1 on `qsd2_w3` (39 paintings) at three margins:

| Paintings found with | Descriptors | 5% | 10% | 15% |
|---|---|---|---|---|
| Otsu masks | multiscale LBP + DCT + wavelet + colour | 0.923 | 1.000 | 0.949 |
| | DCT + wavelet + colour | 0.974 | 1.000 | 0.949 |
| watershed | multiscale LBP + DCT + wavelet + colour | 0.949 | 1.000 | 0.949 |
| | **DCT + wavelet + colour** | **1.000** | **1.000** | **0.974** |

At 10% both fusions find every painting with both extractors. Watershed is never worse, and at 5% and 15%
its crops lose fewer paintings. DCT alone, the descriptor most sensitive to the crop, goes from 0.795 to
0.821 at 10% (0.769 to 0.872 at 5%, 0.821 to 0.897 at 15%). On `qsd1_w3` (cropped, no segmentation) both
fusions give 1.000 at 5%, 10% and 15%.

### Task 5: blind competition submissions (QST1 and QST2)

`week3/make_submission.py` generates the submission files for `qst1_w3` and `qst2_w3` following the required format and folder layout:

- For each test query, returns top-10 matching BBDD integer IDs with an extra list level for multi-painting support:
  `[[[p1_id0, ...], [p2_id0, ...]], [[p1_id0, ...]], ...]`.
- For `qst2_w3`, saves predicted binary foreground masks as `00000.png` ... `00029.png` in the method folder.
- Follows the required delivery layout:
  `outputs/Team5/week3/QST1/method1/result.pkl`
  `outputs/Team5/week3/QST2/method1/result.pkl`
  `outputs/Team5/week3/QST2/method1/*.png`

```bash
# QST1 (cropped, single painting per image)
python week3/make_submission.py --data data --query-set qst1_w3

# QST2 (with background, up to two paintings per image - saves result.pkl and *.png masks)
python week3/make_submission.py --data data --query-set qst2_w3
```

**The delivered files.** This week only one method per test set is accepted, in the `method1` folder. The
delivered files were made with `week3/make_submission_final.py`, which runs three fusions (method1:
multiscale LBP + DCT + wavelet + colour; method2: DCT + wavelet + colour; method3: multiscale LBP + DCT +
colour), writes each one to its own `methodK` folder in the same format, and an analysis of every image
(`report.html`, `cases.txt`, `summary.txt`) to `--analysis-dir`. Its method2 output was copied into the
delivered `method1` folders.

| | QST1-W3 (50 cropped images) | QST2-W3 (30 images with background) |
|---|---|---|
| Descriptors | DCT + Haar wavelet + colour | DCT + Haar wavelet + colour |
| Paintings | the whole image (no segmentation) | watershed from the Otsu masks |
| Crop margin | 10% | 10% |
| Noise | median 3x3 on the images detected as noisy | median 3x3 on the images detected as noisy |

Why, from the development sets (tables of Tasks 3 and 4):

- **Margin 10%**: on `qsd2_w3` it is the only margin at which both fusions find every painting, with the
  Otsu masks and with watershed; on `qsd1_w3` every margin from 5% to 15% gives 1.000.
- **DCT + wavelet + colour** instead of the four descriptors: equal at 10% on both sets, and more stable
  when the crop changes (with watershed on `qsd2_w3`, 1.000 / 1.000 / 0.974 at 5% / 10% / 15%, against
  0.949 / 1.000 / 0.949).
- **Watershed** for QST2: better masks on `qsd2_w3` and fewer paintings lost when the margin changes.

```bash
# QST1 (cropped): no segmentation
python week3/make_submission_final.py --data data --query-dir data/qst1_w3 --no-segment --output-dir outputs/final_run --analysis-dir outputs/week3/final_analysis
# QST2 (with background): watershed
python week3/make_submission_final.py --data data --query-dir data/qst2_w3 --extractor watershed --output-dir outputs/final_run --analysis-dir outputs/week3/final_analysis
```

The delivered files are `outputs/final_run/QST1/method2/result.pkl` and `outputs/final_run/QST2/method2/`
(`result.pkl` and the 30 masks), copied to `Team5/week3/QST1/method1/` and `Team5/week3/QST2/method1/`.
## Week 4

### Tasks 1-2: keypoints, local descriptors, matching and unknown paintings

`src/keypoints.py` describes a painting with keypoints and local descriptors and scores it against a museum image;
`week4/compute_scores.py` scores every query painting against every museum image for a grid of settings, and
`week4/analyze_keypoints.py` makes the choices, the tables (`outputs/week4/keypoint_results.txt`,
`outputs/week4/results.json`) and the figures. `week4/plot_examples.py` adds the noise study and the pictures
(`outputs/week4/examples.json`).

```bash
python week4/compute_scores.py --contrast 0.04                         # SIFT, RootSIFT, ORB, AKAZE on the 3 query sets
python week4/compute_scores.py --methods RootSIFT --contrast 0.02      # DoG contrast thresholds (0.02 is the default)
python week4/compute_scores.py --methods RootSIFT --contrast 0.01
python week4/analyze_keypoints.py
python week4/plot_examples.py --example 2 --failure 8 0
```

**Pipeline for one painting.** The paintings of the photo are found as in Week 3 (median filter on the noisy
images, watershed extraction) and cropped to their bounding box. Each crop and each museum image is converted to grey
(the hue changes do not reach gradients of the intensity) and resized to 512 px on the longer side, and its 1,000
strongest keypoints are described. For every museum image: the two nearest museum descriptors of every query
descriptor (tentative matches), Lowe's ratio test (d1 < 0.75 d2), and a RANSAC homography (a painting is planar);
the score is the number of inliers. The museum is ranked by score, and if the best score is below 44 inliers the
painting is unknown (`[-1]`).

**How the choices were made.** Everything is chosen on `qsd1_w4` (39 paintings, 10 of them not in the museum).
`qsd2_w3` and `qsd1_w3` are kept to check the choices: they have no unknown paintings, so unknowns are simulated by
removing the correct museum image (the best score left is that of the best wrong image).

| Decision | Options (qsd1_w4) | Chosen | Why | Check on qsd2_w3 / qsd1_w3 |
|---|---|---|---|---|
| Local feature and distance | SIFT (L2), RootSIFT (Hellinger), ORB (Hamming), AKAZE (Hamming): all mAP@1 0.862 on the known paintings | RootSIFT | best mAP@5 (0.879) and widest gap: correct image 197 inliers, best wrong 7 (median) | best of the four: mAP@1 0.949 / 0.967 |
| DoG contrast threshold | 0.04, 0.02, 0.01: same mAP@1 | 0.02 | flat paintings keep keypoints (painting 11: 42 -> 108); interval of best unknown F1 15-30 -> 24-65 | unknown F1 0.897 -> 0.938 / 0.897 -> 0.984 |
| Ratio test | 0.6 to 0.9, and none | 0.75 | without it every museum image gets ~1,000 matches (mAP 0); 0.75 has the best mAP@5 | |
| RANSAC | with / without | with | best wrong image 18 -> 8 matches, best unknown 56 -> 23, correct 250 -> 220 | |
| Unknown threshold | 1 to 100 inliers | 44 | middle of the interval with the best F1 (24-65) | unknown F1 0.938 / 0.984 |

**Noise.** On the 7 known noisy paintings of `qsd1_w4` (cut with the annotated frames so only the filter changes),
the 3x3 median raises the inliers with the correct image from 134 to 184 (median; 277 on the clean originals). One
of them goes from 36, below the unknown threshold, to 69.

**Results on qsd1_w4** (39 paintings): mAP@1 0.897, mAP@5 0.906 (0.641 / 0.658 without the unknown test); every
unknown painting is rejected and 2 known ones are wrongly rejected (unknown F1 0.909). The 4 paintings that are not
right: 128, a painting behind a plastic sheet whose reflections change with the view (7 inliers); 11, a uniform
grey canvas (0 inliers); and the two paintings of image 28, which the Week 3 extraction merges into one crop.

### Task 3: comparison with Week 3

`week4/compare_w3_vs_w4.py` compares the delivered Week 3 method (DCT + Haar wavelet + colour fusion, 10% margin)
with the Week 4 method (RootSIFT, ratio 0.75, RANSAC, unknown below 44 inliers) on `qsd1_w4`. Both get the same
paintings (median filter on noisy images, watershed), so only the description and ranking change. The output is
`outputs/week4/comparison_w3_vs_w4.txt`.

```bash
python week4/compare_w3_vs_w4.py
```

| | All paintings (39) mAP@1 / @5 | Known paintings (29) mAP@1 / @5 | Unknown F1 | Time per painting |
|---|---|---|---|---|
| Week 3 (DCT + wavelet + colour) | 0.692 / 0.692 | 0.931 / 0.931 | - | 0.12 s |
| Week 4 (RootSIFT, threshold 44) | 0.897 / 0.906 | 0.862 / 0.874 | 0.909 | 1.12 s |

Week 3 cannot reject a painting that is not in the museum, so it loses the 10 unknowns; on the known paintings it is
slightly better and about 10 times faster. Week 4 rejects all 10 unknowns and wrongly rejects 2 known ones
(painting 128 behind a plastic sheet and painting 11, a flat grey canvas, which Week 3 gets right). Image 28 fails in both
systems because the extraction merges its two paintings into one crop.

### Task 4: blind competition submission (QST1-W4)

`week4/make_submission.py` generates the submission files for `qst1_w4` following the required format and folder layout:

- Uses the best pipeline identified in Tasks 1-3: watershed segmentation, impulse noise filtering (3x3 median on detected noisy images), RootSIFT local features (contrast threshold 0.02, 1000 features, 512 px max side), Lowe's ratio test (0.75), and RANSAC homography geometric verification.
- For each test query, returns top-10 matching BBDD integer IDs per painting in reading order, or `[-1]` for unknown paintings (best inliers < 44):
  `[[[p1_id0, ...], [p2_id0, ...]], [[-1]], ...]`.
- Saves predicted binary foreground masks as `00000.png` ... `00029.png` in the method folder.
- Follows the required delivery layout:
  `Team5/week4/QST1/method1/result.pkl`
  `outputs/Team5/week4/QST1/method1/result.pkl`
  `Team5/week4/QST1/method1/*.png`
- Generates diagnostic reports in `outputs/week4/analysis/qst1_w4/` (`report.html`, `cases.txt`, `summary.txt`, and per-image visual inspection plots in `images/`).

```bash
# Run Week 4 submission on QST1-W4
python week4/make_submission.py --data data --query-dir data/w4/qst1_w4
```
