import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

"""Task 6: save the top-K BBDD IDs of every query as a pickled list of lists of int."""

import argparse
import pickle

import cv2
import numpy as np

from src.descriptors import (
    COLOR_SPACES,
    NORMALIZABLE,
    compute_descriptors,
    convert_image,
    histogram_descriptor,
    list_images,
    read_image,
)
from src.distances import MEASURES
from src.masks import METHODS, segment
from src.retrieval import retrieve


def crop_to_mask(img, mask):
    """Crop image and mask to the bounding box of the foreground, so the blocks line up with the painting."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return img, np.ones(mask.shape, bool)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def describe_query(img, mask, color_space, bins, grid, normalize=True):
    """Descriptor of the painting of a query: crop, normalise and count only the masked pixels."""
    crop, crop_mask = crop_to_mask(img, mask)
    crop_mask = crop_mask.astype(np.uint8)
    converted = convert_image(crop, color_space, normalize=normalize, mask=crop_mask)
    return histogram_descriptor(converted, color_space, bins, grid, mask=crop_mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing BBDD and the query sets")
    parser.add_argument("--query-set", default="qst1_w2")
    parser.add_argument("--color-space", default="lab", choices=list(COLOR_SPACES))
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--grid", type=int, nargs="+", default=[1, 2, 4],
                        help="split the image into grid x grid blocks or a spatial pyramid (e.g. 2 or 1 2 4)")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="remove the global brightness/colour shift first (lab and ycrcb only)")
    parser.add_argument("--measure", default="l1", choices=list(MEASURES))
    parser.add_argument("--mask", default=None, choices=list(METHODS),
                        help="mask method for background removal (e.g. border_otsu)")
    parser.add_argument("--border", type=float, default=0.04)
    parser.add_argument("--min-region", type=float, default=0.05)
    parser.add_argument("--hull", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default=os.path.join("outputs", "result.pkl"))
    args = parser.parse_args()

    grid = args.grid[0] if len(args.grid) == 1 else tuple(args.grid)
    if args.normalize and args.color_space not in NORMALIZABLE:
        parser.error(f"--normalize is only defined for {', '.join(NORMALIZABLE)}: add --no-normalize")

    bbdd_dir = os.path.join(args.data, "BBDD")
    if not os.path.isdir(bbdd_dir) and os.path.isdir(os.path.join(args.data, "w1", "BBDD")):
        bbdd_dir = os.path.join(args.data, "w1", "BBDD")

    query_dir = os.path.join(args.data, args.query_set)
    if not os.path.isdir(query_dir) and os.path.isdir(os.path.join(args.data, "w2", args.query_set)):
        query_dir = os.path.join(args.data, "w2", args.query_set)

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    db_desc, db_ids = compute_descriptors(bbdd_dir, args.color_space, args.bins, args.normalize, grid)

    if args.mask:
        paths = list_images(query_dir)
        query_descs = []
        for p in paths:
            img = read_image(p)
            mask = segment(img, args.mask, border=args.border, hull=args.hull, min_region=args.min_region)
            mask_name = os.path.splitext(os.path.basename(p))[0] + ".png"
            cv2.imwrite(os.path.join(out_dir, mask_name), mask.astype(np.uint8) * 255)
            query_descs.append(describe_query(img, mask, args.color_space, args.bins, grid, args.normalize))
        query_desc = np.stack(query_descs)
    else:
        query_desc, _ = compute_descriptors(query_dir, args.color_space, args.bins, args.normalize, grid)

    results = retrieve(query_desc, db_desc, db_ids, args.measure, args.k)
    results = [[int(x) for x in row] for row in results]

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved top-{args.k} results for {len(results)} queries to {args.output}")


if __name__ == "__main__":
    main()
