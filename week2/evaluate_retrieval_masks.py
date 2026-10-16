"""Task 5: retrieval on the queries with background (qsd2): the masks decide which pixels are described.

For every mask variant the query is cropped to the bounding box of its mask and described using
only the masked pixels (histograms and normalisation). The museum images are described as usual.
Two references are added: no mask (the whole photo, wall included) and the annotated mask.
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.descriptors import compute_descriptors, convert_image, histogram_descriptor, image_id, list_images, read_image
from src.evaluation import mapk
from src.masks import METHODS, mask_scores, segment
from src.retrieval import retrieve

try:
    from tqdm import tqdm
except ImportError:   # pip install tqdm for a progress bar
    def tqdm(iterable, **kwargs):
        return iterable

# The descriptors under consideration (name, colour space, bins, grid or pyramid, distance); always normalised, 1D.
MODELS = [
    ("YCrCb, pyramid (1,2,4), 32 bins, chi2", "ycrcb", 32, (1, 2, 4), "chi2"),
    ("Lab, pyramid (1,2,4), 64 bins, L1", "lab", 64, (1, 2, 4), "l1"),
    ("Lab, 2x2 blocks, 64 bins, L1 (week 1)", "lab", 64, 2, "l1"),
]


def crop_to_mask(img, mask):
    """Crop image and mask to the bounding box of the foreground, so the blocks line up with the painting."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:   # nothing found: use the whole image
        return img, np.ones(mask.shape, bool)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def describe_query(img, mask, color_space, bins, grid):
    """Descriptor of the painting of a query: crop, normalise and count only the masked pixels."""
    crop, crop_mask = crop_to_mask(img, mask)
    crop_mask = crop_mask.astype(np.uint8)
    converted = convert_image(crop, color_space, normalize=True, mask=crop_mask)
    return histogram_descriptor(converted, color_space, bins, grid, mask=crop_mask)


def evaluate(images, masks, db_desc, db_ids, model, gt):
    """mAP@1, mAP@5 and the rank of the correct painting of every query."""
    _, color_space, bins, grid, measure = model
    queries = np.stack([describe_query(img, mask, color_space, bins, grid) for img, mask in zip(images, masks)])
    ranking = retrieve(queries, db_desc, db_ids, measure, k=len(db_ids))
    top5 = [r[:5] for r in ranking]
    return mapk(gt, top5, 1), mapk(gt, top5, 5), [r.index(g[0]) + 1 for r, g in zip(ranking, gt)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing BBDD and the query set")
    parser.add_argument("--query-set", default="qsd2_w1", help="folder with NNNNN.jpg, the masks NNNNN.png and gt_corresps.pkl")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS), help="mask methods to try")
    parser.add_argument("--border", type=float, default=0.04, help="width of the border strips, as a fraction of the shorter side")
    parser.add_argument("--min-region", type=float, default=0.05, help="fraction used by the '+regions' variants")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week2"))
    args = parser.parse_args()

    query_dir = os.path.join(args.data, args.query_set)
    with open(os.path.join(query_dir, "gt_corresps.pkl"), "rb") as f:
        gt = pickle.load(f)
    paths = list_images(query_dir)
    images = [read_image(p) for p in paths]
    annotated = [cv2.imread(os.path.splitext(p)[0] + ".png", cv2.IMREAD_GRAYSCALE) > 127 for p in paths]
    os.makedirs(args.output_dir, exist_ok=True)

    # Museum descriptors, one set per model (the museum images are whole paintings: no mask).
    bbdd_dir = os.path.join(args.data, "BBDD")
    db = []
    for name, color_space, bins, grid, _ in MODELS:
        print(f"Describing BBDD: {name}", flush=True)
        db.append(compute_descriptors(bbdd_dir, color_space, bins, normalize=True, grid=grid))

    # Every set of masks to compare.
    variants = {"no mask (whole photo)": [np.ones(a.shape, bool) for a in annotated], "annotated mask": annotated}
    for method in args.methods:
        for min_region in (None, args.min_region):
            for hull in (False, True):
                name = method + ("+regions" if min_region else "") + ("+hull" if hull else "")
                variants[name] = [segment(img, method, args.border, hull, min_region) for img in tqdm(images, desc=f"  masks {name}", leave=False)]
    mask_f1 = {name: np.mean([mask_scores(m, a)[2] for m, a in zip(masks, annotated)]) for name, masks in variants.items()}

    results = {}   # variant -> one (mAP@1, mAP@5, ranks) per model
    for name, masks in variants.items():
        print(f"Evaluating {name}", flush=True)
        results[name] = [evaluate(images, masks, d_db, ids, model, gt) for model, (d_db, ids) in zip(MODELS, db)]

    n = len(gt)
    fixed = ["no mask (whole photo)", "annotated mask"]
    rest = sorted((v for v in results if v not in fixed), key=lambda v: -np.mean([r[1] for r in results[v]]))
    lines = [f"{args.query_set}: {n} queries (1 query = {1 / n:.4f} mAP@1); masks use border strips of {args.border:.0%}, regions >= {args.min_region} of the largest",
             *[f"M{i + 1} = {model[0]}" for i, model in enumerate(MODELS)], "",
             f"{'masks':<28} {'mask F1':>7} " + " ".join(f"{'M%d mAP@1' % (i + 1):>10} {'mAP@5':>7}" for i in range(len(MODELS)))]
    for name in fixed + rest:
        lines.append(f"{name:<28} {mask_f1[name]:>7.4f} " + " ".join(f"{r[0]:>10.4f} {r[1]:>7.4f}" for r in results[name]))
    summary = "\n".join(lines)
    print("\n" + summary)

    # Rank of the correct painting of every query (1 = first), to see which queries fail.
    detail = []
    for i, model in enumerate(MODELS):
        detail += ["", f"Rank of the correct painting, M{i + 1} = {model[0]}", f"{'query':<6}" + "".join(f"{v:>26}" for v in results)]
        detail += [f"{q:<6}" + "".join(f"{results[v][i][2][q]:>26}" for v in results) for q in range(n)]
    path = os.path.join(args.output_dir, "retrieval_masks.txt")
    with open(path, "w") as f:
        f.write(summary + "\n" + "\n".join(detail) + "\n")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
