# Team 5 - Museum painting retrieval (C1, week 1)

Query-by-Example retrieval: given a photo of a painting, return the most similar
paintings of the museum collection (BBDD). Images are described with 1D colour
histograms and compared with a histogram distance.

## Setup

Tested with Python 3.13.7.

```bash
python -m venv .venv
source .venv/bin/activate
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

Only the `.jpg` files are used.

## Run

With no arguments, `evaluate.py` prints the table of every colour space and measure
(64 bins, no normalisation) and `make_submission.py` uses the final method.

```bash
# Tasks 1-3: mAP@1 and mAP@5 on qsd1_w1, every colour space and measure
python evaluate.py
python evaluate.py --data data --query-set qsd1_w1 --color-space lab --bins 64 --normalize --measure all

# Evaluate with 95% bootstrap confidence intervals (e.g. 1000 resamples)
python evaluate.py --color-space lab --measure all --bootstrap 1000

# Task 4: top-10 results for the test set, saved to outputs/result.pkl
python make_submission.py
python make_submission.py --data data --query-set qst1_w1 --color-space lab --bins 64 --normalize --measure l1 --output outputs/result.pkl
```

`outputs/result.pkl` is a pickled list with one entry per query (in filename order);
each entry is the list of the 10 best BBDD IDs as integers (7 means `bbdd_00007.jpg`).

## Method

**Descriptor.** The image is converted to one colour space (gray, RGB, HSV, Lab or
YCrCb) and a histogram with N bins is computed for each channel over its full range
(OpenCV 8-bit: H in [0, 180), everything else in [0, 256)). Each channel histogram is
divided by its sum and the concatenation is divided by the number of channels, so the
descriptor sums to 1 and does not depend on the image size.

**Normalisation (`--normalize`, Lab and YCrCb only).** The queries are the paintings
with a global change of brightness and colour. Before the histogram, each image is
normalised in the chosen colour space: the brightness channel (L or Y) is shifted to
mean 128 and scaled to standard deviation 50; the two colour channels are shifted to
mean 128. This keeps the shape of the histograms and removes the shift. 

**Measures.** Euclidean, L1, chi-squared, and EMD are distances; histogram intersection,
cosine, and the Hellinger kernel are similarities. The convention in the code is "lower = more
similar": similarities are negated, so retrieval always sorts in ascending order (with
a stable sort, so ties are reproducible).

**Metric.** mAP@K, the mean over the queries of the average precision at K. With one
correct painting per query, AP@K is 1/rank if the correct painting is in the top K and
0 otherwise. Bootstrap resampling can be used to compute 95% confidence intervals across queries.
