# Team 5 - Museum painting retrieval (C1)

Query-by-Example retrieval: given a photo of a painting, return the most similar
paintings of the museum collection (BBDD). Images are described with 1D colour
histograms and compared with a histogram distance.

## Structure

```
src/        shared code: descriptors, distances, retrieval, evaluation
week1/      week 1 scripts (evaluation, submission, error analysis)
week2/      week 2 scripts
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
└── qst1_w1/     00000.jpg ...            (test set, no ground truth)
```

Only the `.jpg` files are used in week 1.

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

### Method

**Descriptor.** The image is converted to one colour space (gray, RGB, HSV, Lab or
YCrCb) and a histogram with N bins is computed for each channel over its full range
(OpenCV 8-bit: H in [0, 180), everything else in [0, 256)). With `--grid G` the image
is first split into G x G blocks and the histograms of every block are concatenated, so
the descriptor also knows where each colour is (each histogram is still 1D; G=1 is the
whole image). Each histogram is divided by its sum and the concatenation is divided by
the number of histograms, so the descriptor sums to 1 and does not depend on the image size.

**Normalisation (`--normalize`, Lab and YCrCb only).** The queries are the paintings
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

On qsd1_w1 this raised the best single-image result (grid 1) from mAP@5 0.87 (Lab, 64
bins, L1) to 0.93 (Lab, 64 bins, L1; and YCrCb, 64 bins, chi-squared, which also moves
query 26 from rank 57 to rank 3). With `--grid 2` almost every setting already reaches
29/30 queries at rank 1, so the gain there is small (mAP@5 0.967 to 0.975 with YCrCb).

**Measures.** Euclidean, L1, chi-squared, EMD and Jensen-Shannon are distances; histogram
intersection, cosine, and the Hellinger kernel are similarities. The convention in the code
is "lower = more similar": similarities are negated, so retrieval always sorts in ascending
order (with a stable sort, so ties are reproducible).

**Metric.** mAP@K, the mean over the queries of the average precision at K. With one
correct painting per query, AP@K is 1/rank if the correct painting is in the top K and
0 otherwise. Bootstrap resampling can be used to compute 95% confidence intervals across queries.

## Week 2

In progress. The scripts will go in `week2/`; shared code stays in `src/`.