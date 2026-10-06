import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

"""Tasks 1-3: retrieve on a query set with ground truth and print mAP@1 and mAP@5."""

import argparse
import pickle

from src.descriptors import COLOR_SPACES, NORMALIZABLE, compute_descriptors
from src.distances import MEASURES
from src.evaluation import bootstrap_mapk, mapk
from src.retrieval import retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing BBDD and the query sets")
    parser.add_argument("--query-set", default="qsd1_w1")
    parser.add_argument("--color-space", default="all", choices=[*COLOR_SPACES, "all"])
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--grid", type=int, default=1,
                        help="split the image into grid x grid blocks and concatenate their histograms")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                        help="remove the global brightness/colour shift first (lab and ycrcb only)")
    parser.add_argument("--measure", default="all", choices=[*MEASURES, "all"])
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="number of bootstrap resamples for 95%% confidence intervals (e.g. 1000; default 0 to disable)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for bootstrap resampling")
    args = parser.parse_args()
    if args.grid < 1:
        parser.error("--grid must be at least 1")

    color_spaces = list(COLOR_SPACES) if args.color_space == "all" else [args.color_space]
    if args.normalize:
        if args.color_space == "all":
            color_spaces = list(NORMALIZABLE)
        elif args.color_space not in NORMALIZABLE:
            parser.error(f"--normalize is only defined for {', '.join(NORMALIZABLE)}")
    measures = list(MEASURES) if args.measure == "all" else [args.measure]

    query_dir = os.path.join(args.data, args.query_set)
    with open(os.path.join(query_dir, "gt_corresps.pkl"), "rb") as f:
        gt = pickle.load(f)

    print(f"query set: {args.query_set}   bins: {args.bins}   grid: {args.grid}   normalize: {args.normalize}"
          + (f"   bootstrap: {args.bootstrap}" if args.bootstrap else ""))
    if args.bootstrap > 0:
        print(f"{'color space':<12} {'measure':<13} {'mAP@1 (95% CI)':<25} {'mAP@5 (95% CI)':<25}")
    else:
        print(f"{'color space':<12} {'measure':<13} {'mAP@1':>7} {'mAP@5':>7}")

    for color_space in color_spaces:
        # Descriptors are computed once per colour space and reused for every measure.
        db_desc, db_ids = compute_descriptors(os.path.join(args.data, "BBDD"), color_space, args.bins, args.normalize, args.grid)
        query_desc, _ = compute_descriptors(query_dir, color_space, args.bins, args.normalize, args.grid)
        for measure in measures:
            predicted = retrieve(query_desc, db_desc, db_ids, measure, k=5)
            if args.bootstrap > 0:
                b1 = bootstrap_mapk(gt, predicted, 1, n_bootstraps=args.bootstrap, seed=args.seed)
                b5 = bootstrap_mapk(gt, predicted, 5, n_bootstraps=args.bootstrap, seed=args.seed)
                s1 = f"{b1.mean:.4f} [{b1.ci_lower:.4f}, {b1.ci_upper:.4f}]"
                s5 = f"{b5.mean:.4f} [{b5.ci_lower:.4f}, {b5.ci_upper:.4f}]"
                print(f"{color_space:<12} {measure:<13} {s1:<25} {s5:<25}")
            else:
                print(f"{color_space:<12} {measure:<13} {mapk(gt, predicted, 1):>7.4f} {mapk(gt, predicted, 5):>7.4f}")


if __name__ == "__main__":
    main()
