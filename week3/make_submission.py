"""Week 3, Task 5: generate blind competition submissions for QST1-W3 and QST2-W3.

Outputs:
- result.pkl: pickled list of lists of lists of ints:
    [[[p1_id0, p1_id1, ...], [p2_id0, p2_id1, ...]], ...]
  with K=10 best matching BBDD IDs per detected painting, in reading order.
  Even for images with a single painting, an extra list level is required.
- For queries with background (e.g. QST2), saves predicted binary masks as NNNNN.png
  in the same folder as result.pkl (0 = background, 255 = foreground painting).

Submission directory structure (GDrive / m1-results):
  Team5/week3/QST1/method1/result.pkl
  Team5/week3/QST2/method1/result.pkl
  Team5/week3/QST2/method1/*.png
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.descriptors import list_images, read_image
from src.painting_retrieval import (
    FUSION,
    MARGIN,
    TEXTURES,
    crop_painting,
    describe_database,
    find_paintings,
    fuse,
    painting_distances,
    rankings,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def resolve_dir(base_dir, sub_path):
    """Find a directory looking in base_dir, or subfolders (e.g. data/w3/qst1_w3)."""
    candidates = [
        os.path.join(base_dir, sub_path),
        os.path.join(base_dir, "w3", sub_path),
        os.path.join(base_dir, sub_path, sub_path),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return os.path.join(base_dir, sub_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing BBDD and query sets")
    parser.add_argument("--query-set", default="qst1_w3", help="query set folder (e.g. qst1_w3 or qst2_w3)")
    parser.add_argument("--query-dir", default=None, help="explicit path to query folder (overrides --query-set)")
    parser.add_argument("--no-segment", action="store_true",
                        help="queries are already cropped (e.g. QST1); default is auto-detected from query set name")
    parser.add_argument("--segment", action="store_true",
                        help="force background removal even if query set name contains qst1/qsd1")
    parser.add_argument("--filter", default="median 3x3", help="noise filter applied if image is noisy ('none' to skip)")
    parser.add_argument("--margin", type=float, default=MARGIN, help="fraction cut from crop borders and BBDD images")
    parser.add_argument("--k", type=int, default=10, help="number of top retrieved BBDD IDs per painting")
    parser.add_argument("--output", default=None,
                        help="path to result.pkl (defaults to outputs/Team5/week3/<SET>/method1/result.pkl)")
    args = parser.parse_args()

    # Resolve BBDD path
    bbdd_dir = os.path.join(args.data, "BBDD")
    if not os.path.isdir(bbdd_dir):
        alt = os.path.join(args.data, "w1", "BBDD")
        if os.path.isdir(alt):
            bbdd_dir = alt

    # Resolve query directory
    query_dir = args.query_dir or resolve_dir(args.data, args.query_set)
    if not os.path.isdir(query_dir):
        parser.error(f"Query directory not found: {query_dir}")

    # Determine segmentation
    set_name = os.path.basename(os.path.normpath(query_dir)).lower()
    if args.segment:
        should_segment = True
    elif args.no_segment:
        should_segment = False
    else:
        # Auto-detect: qst1/qsd1 are cropped single-painting sets; qst2/qsd2 have background
        should_segment = "2" in set_name

    # Determine output path following course submission guidelines
    if args.output:
        output_file = args.output
    else:
        sub_folder = "QST2" if "2" in set_name else "QST1"
        output_file = os.path.join("outputs", "Team5", "week3", sub_folder, "method1", "result.pkl")

    out_dir = os.path.dirname(output_file) or "."
    os.makedirs(out_dir, exist_ok=True)

    paths = list_images(query_dir)
    if not paths:
        parser.error(f"No images found in {query_dir}")

    print(f"Processing {len(paths)} queries from {query_dir} (segmentation: {should_segment})...")

    # 1. Find paintings (adaptive noise filter + segmentation up to 2 paintings)
    found = [find_paintings(read_image(p), segment=should_segment, noise_filter=args.filter)
             for p in tqdm(paths, desc="finding paintings", unit="img")]

    # 2. Save binary masks if segmentation is performed (required for QST2)
    if should_segment:
        for p, (_, masks) in zip(paths, found):
            union_mask = np.any(masks, axis=0).astype(np.uint8) * 255 if masks else np.zeros((100, 100), np.uint8)
            mask_path = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".png")
            cv2.imwrite(mask_path, union_mask)
        print(f"Saved {len(paths)} segmentation masks (*.png) to {out_dir}")

    # 3. Describe museum collection (BBDD)
    print("Computing BBDD descriptors (texture + colour)...")
    descriptors_to_use = [*TEXTURES, "colour"]
    db, db_ids = describe_database(bbdd_dir, descriptors_to_use, args.margin)

    # 4. Crop each painting, compute distance, fuse, and rank
    print("Retrieving best matches for each detected painting...")
    crops = [crop_painting(img, m, args.margin) for img, masks in found for m in masks]
    dist = painting_distances(crops, db)
    fused_dist = fuse(dist, FUSION)
    ranked = rankings(fused_dist, db_ids, k=args.k)

    # Group rankings per query image: [[[p1_ids], [p2_ids]], ...]
    counts = np.cumsum([0] + [len(masks) for _, masks in found])
    results = [ranked[counts[i]:counts[i + 1]] for i in range(len(found))]

    # Verify formatting: list of lists of lists of int
    assert len(results) == len(paths)
    for row in results:
        assert isinstance(row, list) and len(row) >= 1
        for painting_matches in row:
            assert isinstance(painting_matches, list) and len(painting_matches) == args.k
            assert all(isinstance(x, int) for x in painting_matches)

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    total_paintings = sum(len(r) for r in results)
    print(f"Saved top-{args.k} results for {len(results)} images ({total_paintings} paintings) to {output_file}")


if __name__ == "__main__":
    main()
