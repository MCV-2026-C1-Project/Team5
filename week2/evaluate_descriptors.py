"""Task 1-2: do 2D/3D, block and pyramid histograms improve the best Week 1 methods on the dev set?

Each Week 1 method keeps its colour space, normalisation and distance; only the histogram
structure changes. Every image is read and converted once and reused for all the settings.
"""

import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.descriptors import convert_image, histogram_descriptor, image_id, list_images, read_image
from src.evaluation import mapk
from src.retrieval import retrieve

try:
    from tqdm import tqdm
except ImportError:   # pip install tqdm for a progress bar
    def tqdm(iterable, **kwargs):
        return iterable

# The Week 1 methods, with the simple histogram (whole image, 1D, 64 bins) as the baseline.
METHODS = [("Method 1", "lab", "l1"), ("Method 2", "ycrcb", "chi2")]
BASELINE = ("1D whole image", 64)
EXTRA_MEASURES = ["l1", "chi2", "hellinger", "js"]   # with --all-measures (EMD does not apply to 2D/3D histograms)


def settings():
    """Every histogram structure to try: (name, bins, joint, grid, level_weights)."""
    out = [("1D whole image", b, 1, 1, None) for b in (16, 32, 64)]
    for g in (2, 3, 4):
        out += [(f"1D blocks {g}x{g}", b, 1, g, None) for b in (16, 32, 64)]
    for levels, weights in [((1, 2), None), ((1, 2, 3), None), ((1, 2, 4), None), ((1, 2, 4), (1, 2, 4))]:
        name = f"1D pyramid {levels}" + (f" weights {weights}" if weights else "")
        out += [(name, b, 1, levels, weights) for b in (16, 32, 64)]
    for g in (1, 2):
        out += [(f"2D joint, grid {g}", b, 2, g, None) for b in (16, 32, 64)]
        out += [(f"3D joint, grid {g}", b, 3, g, None) for b in (4, 8, 16)]
    out.append(("3D joint, grid 1", 32, 3, 1, None))
    return out


def describe_image(task):
    """All descriptors (one per setting) of one image; it is read and converted only once."""
    path, color_space, cfgs = task
    img = convert_image(read_image(path), color_space, normalize=True)
    cache = {}   # levels shared by several settings are computed once
    return [histogram_descriptor(img, color_space, bins, grid, joint, weights, cache) for _, bins, joint, grid, weights in cfgs]


def describe_folder(paths, color_space, cfgs, workers):
    """One descriptor matrix (N, D) per setting."""
    tasks = [(path, color_space, cfgs) for path in paths]
    if workers > 1:
        with Pool(workers) as pool:
            per_image = list(tqdm(pool.imap(describe_image, tasks), total=len(tasks), desc=f"  {color_space}", unit="img"))
    else:
        per_image = [describe_image(t) for t in tqdm(tasks, desc=f"  {color_space}", unit="img")]
    return [np.stack(column) for column in zip(*per_image)]


def plot_comparison(results, n_queries, path):
    """Best mAP@1 / mAP@5 of each histogram structure, against the Week 1 baseline (dashed)."""
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(14, 4.8 * len(METHODS)), squeeze=False)
    for ax, (method, color_space, measure) in zip(axes.ravel(), METHODS):
        rows = [r for r in results if r["method"] == method]
        base = next(r for r in rows if (r["name"], r["bins"]) == BASELINE and r["measure"] == measure)
        names = list(dict.fromkeys(r["name"] for r in rows))
        best = [max((r for r in rows if r["name"] == n), key=lambda r: (r["m5"], r["m1"])) for n in names]
        x = np.arange(len(names))
        ax.bar(x - 0.2, [b["m1"] for b in best], 0.4, label="mAP@1")
        ax.bar(x + 0.2, [b["m5"] for b in best], 0.4, label="mAP@5")
        for xi, b in zip(x, best):
            ax.text(xi, 0.02, f"{b['bins']} bins", ha="center", fontsize=7, color="white", rotation=90, va="bottom", transform=ax.get_xaxis_transform())
        ax.axhline(base["m1"], color="C0", linestyle="--", linewidth=1)
        ax.axhline(base["m5"], color="C1", linestyle="--", linewidth=1)
        ax.set_xticks(x, names, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(max(0, min(min(b["m1"] for b in best), base["m1"], base["m5"]) - 0.1), 1.02)
        ax.set_title(f"{method} ({color_space}, {measure}): best setting of each structure; dashed = Week 1 baseline (1 query = {1 / n_queries:.3f})")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing BBDD and the query set")
    parser.add_argument("--query-set", default="qsd1_w2")
    parser.add_argument("--all-measures", action="store_true", help="also try l1, chi2, hellinger and js (default: only each method's own distance)")
    parser.add_argument("--workers", type=int, default=4, help="images processed in parallel (1 = no parallelism)")
    parser.add_argument("--top", type=int, default=15, help="rows of the table to print per method")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week2"))
    args = parser.parse_args()

    query_dir = os.path.join(args.data, args.query_set)
    with open(os.path.join(query_dir, "gt_corresps.pkl"), "rb") as f:
        gt = pickle.load(f)
    db_paths, query_paths = list_images(os.path.join(args.data, "BBDD")), list_images(query_dir)
    db_ids = [image_id(p) for p in db_paths]
    cfgs = settings()
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for method, color_space, own_measure in METHODS:
        print(f"{method}: {color_space}, normalised, {len(cfgs)} histogram settings", flush=True)
        db_desc = describe_folder(db_paths, color_space, cfgs, args.workers)
        query_desc = describe_folder(query_paths, color_space, cfgs, args.workers)
        for (name, bins, joint, grid, weights), d_db, d_q in zip(cfgs, db_desc, query_desc):
            for measure in (EXTRA_MEASURES if args.all_measures else [own_measure]):
                predicted = retrieve(d_q, d_db, db_ids, measure, k=5)
                results.append(dict(method=method, name=name, bins=bins, measure=measure, dims=d_db.shape[1],
                                    m1=mapk(gt, predicted, 1), m5=mapk(gt, predicted, 5)))

    n = len(gt)
    header = [f"{args.query_set}: {n} queries (1 query = {1 / n:.4f} mAP@1). Baseline = Week 1 simple histogram, {BASELINE[1]} bins.",
              "'vs base' = change in correct answers at rank 1 (in queries) and in mAP@5, relative to the baseline."]
    full, short = list(header), list(header)
    for method, color_space, own_measure in METHODS:
        rows = [r for r in results if r["method"] == method]
        base = next(r for r in rows if (r["name"], r["bins"]) == BASELINE and r["measure"] == own_measure)
        better = sum(r["m1"] >= base["m1"] and r["m5"] >= base["m5"] and (r["m1"], r["m5"]) != (base["m1"], base["m5"]) for r in rows)
        title = (f"=== {method}: {color_space}, normalised. Baseline ({own_measure}) mAP@1 / mAP@5 = {base['m1']:.4f} / {base['m5']:.4f}; "
                 f"{better} of {len(rows)} settings are at least as good on both metrics and better on one ===")
        columns = f"{'structure':<36} {'bins':>4} {'measure':<10} {'dims':>7} {'mAP@1':>7} {'mAP@5':>7} {'vs base: queries':>17} {'mAP@5':>8}"
        table = [f"{r['name']:<36} {r['bins']:>4} {r['measure']:<10} {r['dims']:>7} {r['m1']:>7.4f} {r['m5']:>7.4f} "
                 f"{(r['m1'] - base['m1']) * n:>+17.0f} {r['m5'] - base['m5']:>+8.4f}"
                 for r in sorted(rows, key=lambda r: (-r["m5"], -r["m1"]))]
        full += ["", title, columns] + table
        short += ["", title, columns] + table[:args.top]

    with open(os.path.join(args.output_dir, "descriptor_search.txt"), "w") as f:
        f.write("\n".join(full) + "\n")
    plot_comparison(results, n, os.path.join(args.output_dir, "descriptor_comparison.png"))
    print("\n" + "\n".join(short))
    print(f"\nFull table: {os.path.join(args.output_dir, 'descriptor_search.txt')}")
    print(f"Plot: {os.path.join(args.output_dir, 'descriptor_comparison.png')}")


if __name__ == "__main__":
    main()