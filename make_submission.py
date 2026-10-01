"""Task 4: save the top-K BBDD IDs of every query as a pickled list of lists of int."""

import argparse
import os
import pickle

from src.descriptors import COLOR_SPACES, NORMALIZABLE, compute_descriptors
from src.distances import MEASURES
from src.retrieval import retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing BBDD and the query sets")
    parser.add_argument("--query-set", default="qst1_w1")
    parser.add_argument("--color-space", default="lab", choices=list(COLOR_SPACES))
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--grid", type=int, default=1,
                        help="split the image into grid x grid blocks and concatenate their histograms")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="remove the global brightness/colour shift first (lab and ycrcb only)")
    parser.add_argument("--measure", default="l1", choices=list(MEASURES))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default=os.path.join("outputs", "result.pkl"))
    args = parser.parse_args()
    if args.grid < 1:
        parser.error("--grid must be at least 1")
    if args.normalize and args.color_space not in NORMALIZABLE:
        parser.error(f"--normalize is only defined for {', '.join(NORMALIZABLE)}: add --no-normalize")

    db_desc, db_ids = compute_descriptors(os.path.join(args.data, "BBDD"), args.color_space, args.bins, args.normalize, args.grid)
    # Queries are read in sorted filename order, so entry i of the result is query i.
    query_desc, _ = compute_descriptors(os.path.join(args.data, args.query_set), args.color_space, args.bins, args.normalize, args.grid)
    results = retrieve(query_desc, db_desc, db_ids, args.measure, args.k)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved top-{args.k} results for {len(results)} queries to {args.output}")


if __name__ == "__main__":
    main()
