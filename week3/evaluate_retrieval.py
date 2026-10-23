"""Task 4: retrieval on qsd2_w3 (background, up to two paintings, noise and hue changes).

For every image: filter it if it is noisy (Task 1), find the paintings (Task 3), crop each one and rank
the museum images (src/painting_retrieval.py). Each descriptor is evaluated alone and in combinations.
With frames.pkl, the same is done on the crops of the annotated frames: the upper bound if the
segmentation were perfect. mAP is computed per painting (src/evaluation.mapk_paintings).
With --no-segment (cropped query sets such as qsd1_w3) the whole image is the painting.
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.descriptors import list_images, read_image
from src.evaluation import flatten_paintings, mapk_paintings
from src.painting_retrieval import (FUSION, MARGIN, TEXTURES, crop_painting, describe_database, find_paintings,
                                    fuse, painting_distances, rankings)

try:
    from tqdm import tqdm
except ImportError:   # pip install tqdm for a progress bar
    def tqdm(iterable, **kwargs):
        return iterable

DESCRIPTORS = [*TEXTURES, "colour"]
COMBINATIONS = [("multiscale LBP", "colour"), ("multiscale LBP", "DCT"), ("multiscale LBP", "DCT", "colour"),
                ("DCT", "wavelet", "colour"), FUSION]


def frame_masks(shape, frames):
    """One mask per annotated painting (bounding box of its frame), in the order of the ground truth."""
    masks = []
    for _, corners in frames:
        corners = np.array(corners)
        (x0, y0), (x1, y1) = np.maximum(corners.min(0), 0).astype(int), corners.max(0).astype(int)
        m = np.zeros(shape, bool)
        m[y0:y1 + 1, x0:x1 + 1] = True
        masks.append(m)
    return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing BBDD")
    parser.add_argument("--query-dir", default=os.path.join("data", "qsd2_w3", "qsd2_w3"),
                        help="folder with NNNNN.jpg and gt_corresps.pkl (and frames.pkl for the upper bound)")
    parser.add_argument("--no-segment", action="store_true", help="the queries are already cropped (e.g. qsd1_w3)")
    parser.add_argument("--margin", type=float, default=MARGIN, help="fraction cut from every side of crops and museum images")
    parser.add_argument("--filter", default="median 3x3", help="filter for the images detected as noisy ('none' to skip)")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week3"))
    args = parser.parse_args()

    paths = list_images(args.query_dir)
    found = [find_paintings(read_image(p), not args.no_segment, args.filter) for p in tqdm(paths, desc="paintings", unit="img")]
    gt = pickle.load(open(os.path.join(args.query_dir, "gt_corresps.pkl"), "rb"))
    ours = "whole image" if args.no_segment else "our masks"
    sources = {ours: found}
    frames_path = os.path.join(args.query_dir, "frames.pkl")
    if not args.no_segment and os.path.exists(frames_path):
        frames = pickle.load(open(frames_path, "rb"))
        sources["annotated frames"] = [(img, frame_masks(img.shape[:2], f)) for (img, _), f in zip(found, frames)]

    db, db_ids = describe_database(os.path.join(args.data, "BBDD"), DESCRIPTORS, args.margin)
    results, predictions = {}, {}
    for source, per_image in sources.items():
        crops = [crop_painting(img, m, args.margin) for img, masks in per_image for m in masks]
        dist = painting_distances(crops, db)
        options = {name: (name,) for name in DESCRIPTORS}
        options.update({" + ".join(c): c for c in COMBINATIONS})
        counts = np.cumsum([0] + [len(masks) for _, masks in per_image])
        for name, parts in options.items():
            ranked = rankings(fuse(dist, parts), db_ids)
            predicted = [ranked[counts[i]:counts[i + 1]] for i in range(len(per_image))]
            results[(source, name)] = (mapk_paintings(gt, predicted, 1), mapk_paintings(gt, predicted, 5))
            predictions[(source, name)] = predicted

    n = sum(len(g) for g in gt)
    final = " + ".join(FUSION)
    lines = [f"{os.path.basename(os.path.normpath(args.query_dir))}: {len(paths)} images, {n} paintings "
             f"(1 painting = {1 / n:.3f} mAP@1); margin {args.margin:.0%}; noisy images filtered with {args.filter}",
             f"{'descriptors':<44}" + "".join(f"{s + ': mAP@1':>26} {'mAP@5':>6}" for s in sources)]
    for name in dict.fromkeys(name for _, name in results):
        lines.append(f"{name:<44}" + "".join(f"{results[(s, name)][0]:>26.4f} {results[(s, name)][1]:>6.4f}" for s in sources))

    flat_gt, flat_pred = flatten_paintings(gt, predictions[(ours, final)])
    index = [(i, j) for i, g in enumerate(gt) for j in range(len(g))]
    misses = [f"  image {i:>2}, painting {j}: correct {a[0]}, rank {p.index(a[0]) + 1 if a[0] in p else '>10'}"
              for (i, j), a, p in zip(index, flat_gt, flat_pred) if not p or p[0] != a[0]]
    lines += ["", f"Final method ({final}), {ours}: {len(misses)} of {n} paintings not at rank 1"] + misses
    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"retrieval_{os.path.basename(os.path.normpath(args.query_dir))}.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
