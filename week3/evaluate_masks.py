"""Task 3: up to two paintings per image (qsd2_w3); remove the background and evaluate the masks.

Every image is filtered first if it is noisy (Task 1), so the noise does not spoil the wall model.
The predicted mask of an image is the union of its paintings; it is compared with the annotated mask
(precision, recall, F1 per image, then averaged), and the number of paintings found is compared with
the number in the ground truth.
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.denoise import FILTERS, denoise
from src.descriptors import list_images, read_image
from src.watershed import watershed_masks
from src.masks import mask_scores, segment_paintings
from week2.evaluate_masks import plot_visual_results

# Variants compared: (name, keyword arguments of segment_paintings, or a function of the image returning the
# masks). The first one is the Week 2 method, the last one is shown per image.
VARIANTS = [
    ("W2: border_otsu+hull", dict(method="border_otsu", bridge=0)),
    ("border_otsu_log", dict(hull=False)),
    ("border_otsu_log+hull", dict()),
    ("watershed (otsu seeds)", watershed_masks),   # a function instead of keyword arguments
]


def paintings_in_gt(entry, gt_mask):
    """Number of paintings of one image: from gt_corresps.pkl if given, else regions of the annotated mask."""
    if entry is not None:
        return len(entry) if isinstance(entry, (list, tuple)) else 1
    n, _, stats, _ = cv2.connectedComponentsWithStats(gt_mask.astype(np.uint8), connectivity=8)
    return int(np.sum(stats[1:, cv2.CC_STAT_AREA] > 0.01 * gt_mask.size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing the query sets")
    parser.add_argument("--query-set", default="qsd2_w3", help="folder with NNNNN.jpg images and NNNNN.png masks")
    parser.add_argument("--filter", default="median 3x3", choices=[*FILTERS, "none"],
                        help="filter applied to the images detected as noisy before segmenting")
    parser.add_argument("--save-masks", action="store_true", help="save the masks of the last variant as NNNNN.png")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week3"))
    args = parser.parse_args()

    query_dir = os.path.join(args.data, args.query_set)
    paths = list_images(query_dir)
    images = [read_image(p) for p in paths]
    if args.filter != "none":
        images = [denoise(img, args.filter) for img in images]
    gts = [cv2.imread(os.path.splitext(p)[0] + ".png", cv2.IMREAD_GRAYSCALE) > 127 for p in paths]
    corresps = os.path.join(query_dir, "gt_corresps.pkl")
    entries = pickle.load(open(corresps, "rb")) if os.path.exists(corresps) else [None] * len(paths)
    n_true = [paintings_in_gt(e, g) for e, g in zip(entries, gts)]
    os.makedirs(args.output_dir, exist_ok=True)

    lines = [f"{args.query_set}: {len(paths)} images, {sum(n_true)} paintings "
             f"({sum(n == 2 for n in n_true)} images with two); filter on noisy images: {args.filter}",
             f"{'variant':<24} {'precision':>9} {'recall':>7} {'F1':>7} {'worst F1':>9}  {'count right':>11}"]
    per_image = {}
    for name, kwargs in VARIANTS:
        preds = [kwargs(img) if callable(kwargs) else segment_paintings(img, **kwargs) for img in images]
        unions = [np.any(p, axis=0) if p else np.zeros(g.shape, bool) for p, g in zip(preds, gts)]
        s = np.array([mask_scores(u, g) for u, g in zip(unions, gts)])
        right = sum(len(p) == n for p, n in zip(preds, n_true))
        per_image[name] = (s, [len(p) for p in preds], unions)
        lines.append(f"{name:<24} {s[:, 0].mean():>9.4f} {s[:, 1].mean():>7.4f} {s[:, 2].mean():>7.4f} "
                     f"{s[:, 2].min():>9.4f}  {right:>8}/{len(paths)}")
        print(f"done {name}", flush=True)

    name = VARIANTS[-1][0]
    s, found, unions = per_image[name]
    lines += ["", f"Per image ({name}):", f"{'image':<6} {'GT':>3} {'found':>6} {'F1':>7}"]
    lines += [f"{i:<6} {n_true[i]:>3} {found[i]:>6} {s[i, 2]:>7.4f}" for i in range(len(paths))]
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "mask_results.txt"), "w") as f:
        f.write(text + "\n")
    plot_visual_results(name, images, gts, unions, s[:, 2], os.path.join(args.output_dir, "masks.png"))

    if args.save_masks:
        mask_dir = os.path.join(args.output_dir, "masks", args.query_set)
        os.makedirs(mask_dir, exist_ok=True)
        for p, u in zip(paths, unions):
            cv2.imwrite(os.path.join(mask_dir, os.path.splitext(os.path.basename(p))[0] + ".png"), u.astype(np.uint8) * 255)
        print(f"Masks saved to {mask_dir}")
    print(f"Saved {os.path.join(args.output_dir, 'mask_results.txt')} and masks.png")


if __name__ == "__main__":
    main()