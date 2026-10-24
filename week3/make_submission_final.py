"""Task 5: blind submission of week 3 for the best pipelines, with a case-by-case analysis.

This is the script that produced the delivered week 3 files (method2, watershed for QST2, 10% margin).

For every image: filter it if it is noisy, find the paintings, crop each one and rank the museum images with
each fusion of descriptors in METHODS (the first one is the one to submit as method1; the others are written
too, as method2 and method3, to compare them).

Submission files (the layout of the GDrive folder), in --output-dir:
  <QST1|QST2>/methodK/result.pkl   per image a list with one list of 10 IDs per painting (the extra level)
  <QST2>/methodK/NNNNN.png         binary masks, for the sets with background

Analysis, in --analysis-dir/<set>/: report.html (open it in a browser: one picture per image, the cases
that need a look first), cases.txt (one line per painting), summary.txt and images/NNNNN.png.
If the set has a ground truth (gt_corresps.pkl, e.g. qsd2_w3) every case is marked correct or wrong and the
mAP of each method is printed: that is how to check this script. The test sets have none, so the cases are
marked by whether the methods agree on the first result, and the museum images are shown to judge by eye.
"""

import argparse
import html
import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.denoise import is_noisy
from src.descriptors import image_id, list_images, read_image
from src.evaluation import mapk_paintings
from src.painting_retrieval import (EXTRACTORS, MARGIN, crop_painting, describe_database, find_paintings, fuse,
                                    painting_distances, rankings)

try:
    from tqdm import tqdm
except ImportError:   # pip install tqdm for a progress bar
    def tqdm(iterable, **kwargs):
        return iterable

# The approaches to run, best first (mAP of week3/evaluate_retrieval.py on qsd2_w3 with our masks).
METHODS = [
    ("multiscale LBP", "DCT", "wavelet", "colour"),   # method1: 1.0000 / 1.0000
    ("DCT", "wavelet", "colour"),                      # method2: 1.0000 / 1.0000
    ("multiscale LBP", "DCT", "colour"),               # method3: 0.9744 / 0.9872
]
K = 10        # results per painting in result.pkl
SHOWN = 5     # museum images shown per painting and method in the pictures
THUMB = 220   # longest side of the thumbnails
SEVERITY = {0: "#2e7d32", 1: "#ef6c00", 2: "#c62828", 3: "#6a1b9a"}   # colour of each case in the report


def thumbnail(img_bgr):
    """RGB thumbnail."""
    scale = THUMB / max(img_bgr.shape[:2])
    return cv2.cvtColor(cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def photo_with_masks(img_bgr, masks):
    """Thumbnail of the photo with the mask of every painting tinted in its own colour."""
    thumb = thumbnail(img_bgr).astype(np.float32)
    for mask, colour in zip(masks, ((0, 160, 0), (0, 90, 230))):
        small = cv2.resize(mask.astype(np.uint8), (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        thumb[small] = 0.6 * thumb[small] + 0.4 * np.array(colour, np.float32)
    return thumb.astype(np.uint8)


def show(ax, rgb, title, colour=None):
    """Draw an image; a coloured frame marks it."""
    ax.axis("on")
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(colour is not None)
        if colour is not None:
            spine.set_edgecolor(colour)
            spine.set_linewidth(3)


def judge(found, correct, tops, fallback):
    """(severity, text) of an image. 0 fine, 1 methods disagree, 2 a wrong first result, 3 the segmentation looks wrong."""
    problems = []
    if fallback:
        problems.append((3, "no painting found: the whole image was used"))
    if correct is not None and found != len(correct):
        problems.append((3, f"{found} paintings found, {len(correct)} in the ground truth"))
    for j in range(found):
        firsts = [t[j] for t in tops]
        if correct is not None and j < len(correct):
            wrong = [f"method{k + 1}" for k, first in enumerate(firsts) if first != correct[j]]
            if wrong:
                problems.append((2, f"painting {j}: wrong with {', '.join(wrong)}"))
        elif correct is None and len(set(firsts)) > 1:
            problems.append((1, f"painting {j}: the methods disagree on the first result"))
    if not problems:
        return 0, "all correct" if correct is not None else "the methods agree"
    return max(p[0] for p in problems), "; ".join(p[1] for p in problems)


def plot_case(stem, photo, masks, crops, full_ranks, correct, status, museum_thumb, path):
    """One picture per image: the photo with its masks, then for every painting and method the crop and the museum images."""
    n_methods = len(METHODS)
    rows = 1 + len(crops) * n_methods
    fig, axes = plt.subplots(rows, 1 + SHOWN, figsize=(1.9 * (1 + SHOWN), 2.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    show(axes[0][0], photo_with_masks(photo, masks), f"image {stem}")
    for j, (crop, _) in enumerate(crops):
        known = correct is not None and j < len(correct)
        for k in range(n_methods):
            ranking = full_ranks[k][j]
            note = f"\ncorrect bbdd_{correct[j]:05d}: rank {ranking.index(correct[j]) + 1}" if known else ""
            row = 1 + j * n_methods + k
            show(axes[row][0], thumbnail(crop), f"painting {j}, method{k + 1}{note}")
            for r in range(SHOWN):
                ident = ranking[r]
                colour = ("green" if ident == correct[j] else None) if known else ("#1565c0" if r == 0 else None)
                show(axes[row][1 + r], museum_thumb(ident), f"#{r + 1}  bbdd_{ident:05d}", colour)
    fig.suptitle(f"image {stem}: {status}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="folder containing BBDD")
    parser.add_argument("--query-dir", default=os.path.join("data", "qst2_w3"), help="folder with NNNNN.jpg (and gt_corresps.pkl if there is a ground truth)")
    parser.add_argument("--no-segment", action="store_true", help="the queries are already cropped (QST1, qsd1_w3): the whole image is the painting")
    parser.add_argument("--margin", type=float, default=MARGIN, help="fraction cut from every side of crops and museum images")
    parser.add_argument("--filter", default="median 3x3", help="filter for the images detected as noisy ('none' to skip)")
    parser.add_argument("--extractor", default="otsu", choices=list(EXTRACTORS), help="how the paintings are found")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "Team5", "week3"), help="submission files")
    parser.add_argument("--analysis-dir", default=os.path.join("outputs", "week3", "analysis"))
    args = parser.parse_args()

    name = os.path.basename(os.path.normpath(args.query_dir))   # qst2_w3
    label = name.split("_")[0].upper()                           # QST2
    paths = list_images(args.query_dir)
    print(f"{len(paths)} images in {args.query_dir}", flush=True)
    gt = augmentations = None
    if os.path.exists(os.path.join(args.query_dir, "gt_corresps.pkl")):
        with open(os.path.join(args.query_dir, "gt_corresps.pkl"), "rb") as f:
            gt = pickle.load(f)
    if os.path.exists(os.path.join(args.query_dir, "augmentations.pkl")):
        with open(os.path.join(args.query_dir, "augmentations.pkl"), "rb") as f:
            augmentations = pickle.load(f)

    bbdd_dir = os.path.join(args.data, "BBDD")
    names = list(dict.fromkeys(n for method in METHODS for n in method))
    db, db_ids = describe_database(bbdd_dir, names, args.margin)
    bbdd_paths = {image_id(p): p for p in list_images(bbdd_dir)}
    museum = {}

    def museum_thumb(ident):
        if ident not in museum:
            museum[ident] = thumbnail(read_image(bbdd_paths[ident]))
        return museum[ident]

    analysis_dir = os.path.join(args.analysis_dir, name)
    os.makedirs(os.path.join(analysis_dir, "images"), exist_ok=True)
    method_dirs = [os.path.join(args.output_dir, label, f"method{k + 1}") for k in range(len(METHODS))]
    for d in method_dirs:
        os.makedirs(d, exist_ok=True)

    predictions = [[] for _ in METHODS]   # per method, per image, per painting: the K best IDs
    records = []
    for i, path in enumerate(tqdm(paths, desc=name, unit="img")):
        stem = os.path.splitext(os.path.basename(path))[0]
        original = read_image(path)
        photo, masks = find_paintings(original, not args.no_segment, args.filter, args.extractor)
        fallback = (not args.no_segment) and len(masks) == 1 and bool(masks[0].all())
        crops = [crop_painting(photo, m, args.margin) for m in masks]
        dist = painting_distances(crops, db)
        full = [rankings(fuse(dist, method), db_ids, len(db_ids)) for method in METHODS]   # method -> painting -> all IDs
        for k in range(len(METHODS)):
            predictions[k].append([ranking[:K] for ranking in full[k]])
        if not args.no_segment:   # the mask of an image: its paintings together
            union = (np.any(masks, axis=0).astype(np.uint8)) * 255
            for d in method_dirs:
                cv2.imwrite(os.path.join(d, stem + ".png"), union)

        correct = [int(c) for c in gt[i]] if gt is not None else None
        tops = [[ranking[0] for ranking in full[k]] for k in range(len(METHODS))]
        severity, status = judge(len(masks), correct, tops, fallback)
        noisy = is_noisy(original)
        extra = f" | noise detected: {'yes' if noisy else 'no'}" + (f" | augmentation: {str(augmentations[i])[:50]}" if augmentations is not None and len(augmentations) == len(paths) else "")
        plot_case(stem, photo, masks, crops, full, correct, status + extra, museum_thumb, os.path.join(analysis_dir, "images", stem + ".png"))
        ranks = [[ranking.index(correct[j]) + 1 for j, ranking in enumerate(full[k]) if j < len(correct)] for k in range(len(METHODS))] if correct else None
        records.append(dict(stem=stem, noisy=noisy, found=len(masks), correct=correct, tops=tops, ranks=ranks,
                            severity=severity, status=status + extra))

    for k, d in enumerate(method_dirs):   # the format asked for: a list (images) of lists (paintings) of lists of int
        assert all(type(x) is int and len(p) == K for image in predictions[k] for p in image for x in p), "unexpected result format"
        with open(os.path.join(d, "result.pkl"), "wb") as f:
            pickle.dump(predictions[k], f)

    # cases.txt: one line per painting
    has_gt = gt is not None
    header = f"{'image':<7}{'paint':<6}{'noisy':<6}" + (f"{'GT':>7}" if has_gt else "") + "".join(f"{'m%d top1' % (k + 1):>9}" for k in range(len(METHODS)))
    header += ("".join(f"{'m%d rank' % (k + 1):>9}" for k in range(len(METHODS))) if has_gt else f"{'agree':>9}") + "  result"
    lines = [header]
    for r in records:
        for j in range(max(r["found"], len(r["correct"]) if has_gt else 0)):
            found = j < r["found"]
            row = f"{r['stem']:<7}{j:<6}{'yes' if r['noisy'] else 'no':<6}"
            correct = r["correct"][j] if has_gt and j < len(r["correct"]) else None
            if has_gt:
                row += f"{correct if correct is not None else '-':>7}"
            row += "".join(f"{t[j] if found else '-':>9}" for t in r["tops"])
            if has_gt:
                row += "".join(f"{r['ranks'][k][j] if found and correct is not None else '-':>9}" for k in range(len(METHODS)))
                result = "NOT FOUND" if not found else "EXTRA (not in the ground truth)" if correct is None else \
                    "ok" if all(t[j] == correct for t in r["tops"]) else "WRONG with " + ", ".join(f"method{k + 1}" for k, t in enumerate(r["tops"]) if t[j] != correct)
            else:
                same = len({t[j] for t in r["tops"]}) == 1
                row += f"{'yes' if same else 'NO':>9}"
                result = "agree" if same else "DISAGREE"
            lines.append(row + "  " + result)
    with open(os.path.join(analysis_dir, "cases.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # summary.txt
    n_paintings = sum(r["found"] for r in records)
    summary = [f"{name}: {len(records)} images, {n_paintings} paintings found ({sum(r['found'] == 2 for r in records)} images with two); "
               f"noise detected in {sum(r['noisy'] for r in records)} images; paintings found with: {args.extractor}",
               *[f"method{k + 1} = {' + '.join(method)}" for k, method in enumerate(METHODS)], ""]
    if has_gt:
        summary.append("mAP per painting (1 painting = %.3f mAP@1):" % (1 / sum(len(g) for g in gt)))
        for k in range(len(METHODS)):
            summary.append(f"  method{k + 1}: mAP@1 {mapk_paintings(gt, predictions[k], 1):.4f}  mAP@5 {mapk_paintings(gt, predictions[k], 5):.4f}")
        summary.append(f"images with a wrong number of paintings: {sum(r['found'] != len(r['correct']) for r in records)}")
    else:
        for k in range(1, len(METHODS)):
            same = sum(a == b for r in records for a, b in zip(r["tops"][0], r["tops"][k]))
            summary.append(f"method{k + 1} gives the same first result as method1 for {same} of {n_paintings} paintings")
        summary.append("(no ground truth: the cases marked 'disagree' are the ones to look at)")
    summary.append(f"images flagged for a look: {sum(r['severity'] > 0 for r in records)} of {len(records)}")
    with open(os.path.join(analysis_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")

    # report.html: the cases that need a look first
    body = [f"<h1>{html.escape(name)}</h1>", "<pre>" + html.escape("\n".join(summary)) + "</pre>"]
    for r in sorted(records, key=lambda r: (-r["severity"], r["stem"])):
        body.append(f"<h3 style='color:{SEVERITY[r['severity']]}'>image {r['stem']}: {html.escape(r['status'])}</h3>"
                    f"<img src='images/{r['stem']}.png' style='max-width:1000px'>")
    with open(os.path.join(analysis_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write("<html><body style='font-family:sans-serif'>" + "\n".join(body) + "</body></html>")

    print("\n" + "\n".join(summary))
    print(f"\nSubmission files: {os.path.join(args.output_dir, label)}")
    print(f"Analysis (open report.html): {analysis_dir}")


if __name__ == "__main__":
    main()