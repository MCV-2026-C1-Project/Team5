"""Task 1: noise filtering on the queries of qsd1_w3.

The query set has the clean version of every query (non_augmented/) and the annotation of what
was done to it (augmentations.pkl), so each filter is evaluated in two ways:
- PSNR and SSIM against the clean image, on the noisy queries (higher = closer to the original);
- PSNR and SSIM against the input, on the queries that are not noisy, if the filter were applied
  anyway (higher = less damage), which is why filters are only applied when noise is detected.
It also reports PSNR and SSIM of the median filter as its size grows. The effect of filtering on
retrieval is evaluated with the texture descriptors (evaluate_texture.py).
"""

import argparse
import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.denoise import FILTERS, NOISY_RATIO, impulse_ratio, median
from src.descriptors import list_images, read_image
from src.evaluation import ssim

MEDIAN_SIZES = (3, 5, 7, 9, 11, 13, 15)
CROP = 120   # side of the zoomed crops in the examples figure


def quality(filtered, references):
    """Mean PSNR and SSIM of filtered images against their references."""
    return (np.mean([cv2.PSNR(f, r) for f, r in zip(filtered, references)]),
            np.mean([ssim(f, r) for f, r in zip(filtered, references)]))


def plot_examples(images, cleans, indices, path):
    """Zoomed crop of some noisy queries: input, every filter, clean original."""
    columns = ["input"] + list(FILTERS) + ["clean"]
    fig, axes = plt.subplots(len(indices), len(columns), figsize=(2.2 * len(columns), 2.4 * len(indices)), squeeze=False)
    for row, i in zip(axes, indices):
        h, w = images[i].shape[:2]
        y, x = (h - CROP) // 2, (w - CROP) // 2
        crops = [images[i]] + [FILTERS[name](images[i]) for name in FILTERS] + [cleans[i]]
        for ax, name, img in zip(row, columns, crops):
            ax.imshow(cv2.cvtColor(img[y:y + CROP, x:x + CROP], cv2.COLOR_BGR2RGB), interpolation="nearest")
            ax.set_title(f"#{i} {name}" if name == "input" else name, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing BBDD and the query set")
    parser.add_argument("--query-set", default="qsd1_w3")
    parser.add_argument("--examples", type=int, default=3, help="noisy queries shown in the examples figure")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week3"))
    args = parser.parse_args()

    query_dir = os.path.join(args.data, args.query_set)
    paths = list_images(query_dir)
    images = [read_image(p) for p in paths]
    cleans = [read_image(os.path.join(query_dir, "non_augmented", os.path.basename(p))) for p in paths]
    with open(os.path.join(query_dir, "augmentations.pkl"), "rb") as f:
        noisy = ["Noise" in str(a) for a in pickle.load(f)]
    os.makedirs(args.output_dir, exist_ok=True)
    noisy_images = [x for x, n in zip(images, noisy) if n]
    noisy_cleans = [c for c, n in zip(cleans, noisy) if n]
    other_images = [x for x, n in zip(images, noisy) if not n]

    # 1. Noise detection against the annotation.
    ratios = [impulse_ratio(img) for img in images]
    detected = [r > NOISY_RATIO for r in ratios]
    lines = [f"{args.query_set}: {len(images)} queries, {sum(noisy)} annotated as noisy",
             f"Detection (impulse ratio > {NOISY_RATIO}): {sum(d == n for d, n in zip(detected, noisy))}/{len(images)} correct; "
             f"impulse ratio of noisy queries {min(r for r, n in zip(ratios, noisy) if n):.4f}-{max(r for r, n in zip(ratios, noisy) if n):.4f}, "
             f"of the others {min(r for r, n in zip(ratios, noisy) if not n):.4f}-{max(r for r, n in zip(ratios, noisy) if not n):.4f}"]

    # 2. Quality of every filter.
    lines += ["", f"{'filter':<18} {'noisy vs clean: PSNR':>21} {'SSIM':>6} {'others vs input: PSNR':>22} {'SSIM':>6}"]
    rows = [("no filter", *quality(noisy_images, noisy_cleans), float("inf"), 1.0)]
    for name, fn in FILTERS.items():
        rows.append((name, *quality([fn(x) for x in noisy_images], noisy_cleans), *quality([fn(x) for x in other_images], other_images)))
    lines += [f"{name:<18} {pn:>21.2f} {sn:>6.3f} {po:>22.2f} {so:>6.3f}" for name, pn, sn, po, so in rows]

    # 3. Median filter size: larger windows remove no more impulses, they only blur more.
    lines += ["", "Median filter size: PSNR / SSIM", f"{'size':<6} {'noisy vs clean':>16} {'others vs input':>17}"]
    for size in MEDIAN_SIZES:
        n = quality([median(x, size) for x in noisy_images], noisy_cleans)
        o = quality([median(x, size) for x in other_images], other_images)
        lines.append(f"{size}x{size:<4} {n[0]:>9.2f} / {n[1]:.3f} {o[0]:>10.2f} / {o[1]:.3f}")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "denoise_results.txt"), "w") as f:
        f.write(text + "\n")
    plot_examples(images, cleans, [i for i, n in enumerate(noisy) if n][:args.examples], os.path.join(args.output_dir, "denoise_examples.png"))
    print(f"Saved denoise_results.txt and denoise_examples.png in {args.output_dir}")


if __name__ == "__main__":
    main()
