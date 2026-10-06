import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

"""Tasks 1-3: list the queries whose correct painting is not ranked first and plot the worst ones."""

import argparse
import pickle

import cv2
import matplotlib.pyplot as plt

from src.descriptors import COLOR_SPACES, NORMALIZABLE, compute_descriptors, image_id, list_images, read_image
from src.distances import MEASURES
from src.retrieval import retrieve




def show(ax, path, title):
    """Draw one image (OpenCV reads BGR, matplotlib wants RGB)."""
    ax.imshow(cv2.cvtColor(read_image(path), cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing BBDD and the query sets")
    parser.add_argument("--query-set", default="qsd1_w1")
    parser.add_argument("--color-space", default="lab", choices=list(COLOR_SPACES))
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--grid", type=int, default=1,
                        help="split the image into grid x grid blocks and concatenate their histograms")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--measure", default="l1", choices=list(MEASURES))
    parser.add_argument("--n", type=int, default=6, help="how many of the worst queries to plot")
    parser.add_argument("--output", default=os.path.join("outputs", "errors.png"))
    args = parser.parse_args()
    if args.grid < 1:
        parser.error("--grid must be at least 1")
    if args.normalize and args.color_space not in NORMALIZABLE:
        parser.error(f"--normalize is only defined for {', '.join(NORMALIZABLE)}: add --no-normalize")

    bbdd_dir = os.path.join(args.data, "BBDD")
    query_dir = os.path.join(args.data, args.query_set)
    with open(os.path.join(query_dir, "gt_corresps.pkl"), "rb") as f:
        gt = pickle.load(f)

    db_desc, db_ids = compute_descriptors(bbdd_dir, args.color_space, args.bins, args.normalize, args.grid)
    query_desc, _ = compute_descriptors(query_dir, args.color_space, args.bins, args.normalize, args.grid)
    # Rank the whole database so we can see where the correct painting ended up.
    ranking = retrieve(query_desc, db_desc, db_ids, args.measure, k=len(db_ids))

    # (rank of the correct painting, query number); rank 1 means the query was correct.
    ranks = [(ranking[i].index(gt[i][0]) + 1, i) for i in range(len(gt))]
    errors = sorted((r for r in ranks if r[0] > 1), reverse=True)

    print(f"{args.color_space}, {args.measure}, {args.bins} bins, grid {args.grid}, normalize={args.normalize}: "
          f"{len(gt) - len(errors)}/{len(gt)} queries correct at rank 1")
    if not errors:
        return
    print(f"{'query':>5} {'correct':>8} {'top-1':>6} {'rank of correct':>16}")
    for rank, i in errors:
        print(f"{i:>5} {gt[i][0]:>8} {ranking[i][0]:>6} {rank:>16}")

    # One row per error: query | what we returned | what was correct.
    query_paths = list_images(query_dir)
    bbdd_paths = {image_id(p): p for p in list_images(bbdd_dir)}
    worst = errors[:args.n]
    fig, axes = plt.subplots(len(worst), 3, figsize=(9, 3 * len(worst)), squeeze=False)
    for row, (rank, i) in zip(axes, worst):
        show(row[0], query_paths[i], f"query {i}")
        show(row[1], bbdd_paths[ranking[i][0]], f"returned (top-1): {ranking[i][0]}")
        show(row[2], bbdd_paths[gt[i][0]], f"correct: {gt[i][0]} (rank {rank})")
    fig.suptitle(f"Worst errors: {args.color_space}, {args.measure}, {args.bins} bins", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(f"Saved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()