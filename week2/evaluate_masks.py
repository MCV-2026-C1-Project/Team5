"""Tasks 3-4: background removal with colour only; metrics table, metrics plot and visual results."""

import argparse
import os
import sys                                                  # <- add

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # <- add

from src.descriptors import list_images, read_image
from src.masks import METHODS, mask_scores, segment

THUMB = 250   # longest side of the thumbnails in the visual results
COLS = 6


def overlay(img_bgr, pred, gt):
    """Thumbnail with correct foreground in green, false positives in red and missed foreground in blue."""
    scale = THUMB / max(img_bgr.shape[:2])
    size = (int(img_bgr.shape[1] * scale), int(img_bgr.shape[0] * scale))
    thumb = cv2.cvtColor(cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    pred, gt = (cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool) for m in (pred, gt))
    for region, colour in ((pred & gt, (0, 200, 0)), (pred & ~gt, (230, 0, 0)), (~pred & gt, (0, 0, 230))):
        thumb[region] = 0.55 * thumb[region] + 0.45 * np.array(colour, np.float32)
    return thumb.astype(np.uint8)


def plot_visual_results(name, images, gts, preds, f1s, path):
    """All images of the query set, worst F1 first."""
    order = np.argsort(f1s)
    rows = int(np.ceil(len(order) / COLS))
    fig, axes = plt.subplots(rows, COLS, figsize=(3 * COLS, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), order):
        ax.imshow(overlay(images[i], preds[i], gts[i]))
        ax.set_title(f"#{i}  F1={f1s[i]:.3f}", fontsize=9)
    fig.suptitle(f"{name}   (green: correct, red: false positive, blue: missed)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_metrics(names, scores, path):
    """Mean precision/recall/F1 per variant, and the spread of the per-image F1."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(names))
    for j, label in enumerate(("precision", "recall", "F1")):
        left.bar(x + (j - 1) * 0.27, [scores[n][:, j].mean() for n in names], 0.27, label=label)
    left.set_xticks(x, names, rotation=45, ha="right")
    left.set_ylim(0, 1)
    left.set_title("Mean over images")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=3)
    right.boxplot([scores[n][:, 2] for n in names], tick_labels=names)
    right.set_xticklabels(names, rotation=45, ha="right")
    right.set_ylim(max(0, min(s[:, 2].min() for s in scores.values()) - 0.05), 1.01)   # zoom to where the images are
    right.set_title("F1 per image")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing the query sets")
    parser.add_argument("--query-set", default="qsd2_w1", help="folder with NNNNN.jpg images and NNNNN.png masks")
    parser.add_argument("--border", type=float, default=0.04, help="width of the border strips, as a fraction of the shorter side")
    parser.add_argument("--top", type=int, default=3, help="visual results for the best N variants")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week2"))
    args = parser.parse_args()

    query_dir = os.path.join(args.data, args.query_set)
    paths = list_images(query_dir)
    images = [read_image(p) for p in paths]
    gts = []
    for p, img in zip(paths, images):
        gt = cv2.imread(os.path.splitext(p)[0] + ".png", cv2.IMREAD_GRAYSCALE) > 127
        gts.append(gt if gt.shape == img.shape[:2] else cv2.resize(gt.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool))
    os.makedirs(args.output_dir, exist_ok=True)

    # Every method, without and with the convex hull of its largest region.
    preds, scores = {}, {}
    for method in METHODS:
        for hull in (False, True):
            name = method + ("+hull" if hull else "")
            preds[name] = [segment(img, method, args.border, hull) for img in images]
            scores[name] = np.array([mask_scores(p, g) for p, g in zip(preds[name], gts)])
            print(f"done {name}", flush=True)
    scores = {n: scores[n] for n in sorted(scores, key=lambda n: -scores[n][:, 2].mean())}   # best F1 first

    lines = [f"{args.query_set}: {len(paths)} images, border strips {args.border:.0%}",
             f"{'variant':<22} {'precision':>9} {'recall':>7} {'F1':>7} {'worst F1':>9}  worst image"]
    for name, s in scores.items():
        lines.append(f"{name:<22} {s[:, 0].mean():>9.4f} {s[:, 1].mean():>7.4f} {s[:, 2].mean():>7.4f} {s[:, 2].min():>9.4f}  #{s[:, 2].argmin()}")
    lines += ["", "F1 per image:", f"{'image':<6}" + "".join(f"{n:>22}" for n in scores)]
    lines += [f"{i:<6}" + "".join(f"{s[i, 2]:>22.4f}" for s in scores.values()) for i in range(len(paths))]
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "mask_results.txt"), "w") as f:
        f.write(text + "\n")

    plot_metrics(list(scores), scores, os.path.join(args.output_dir, "mask_metrics.png"))
    for name in list(scores)[:args.top]:
        path = os.path.join(args.output_dir, f"masks_{name.replace('+', '_')}.png")
        plot_visual_results(name, images, gts, preds[name], scores[name][:, 2], path)
        print(f"Saved {path}")
    print(f"Saved {os.path.join(args.output_dir, 'mask_metrics.png')} and mask_results.txt")


if __name__ == "__main__":
    main()