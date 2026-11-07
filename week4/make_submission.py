"""Task 4: blind submission of Week 4 for keypoint-based painting retrieval.

For every image:
1. Filter impulse noise if detected (3x3 median).
2. Segment and find up to two paintings (watershed from Otsu masks).
3. Crop each painting to its bounding box (margin 0 for local features).
4. Describe with local keypoint features (RootSIFT, contrast 0.02, 1000 features, 512px max side).
5. Match against BBDD using Lowe's ratio test (0.75) and RANSAC homography geometric verification.
6. If best inliers < 44 -> unknown painting ([-1]); else return top-K BBDD IDs.
7. Save binary foreground masks as NNNNN.png for background evaluation.
8. Generate detailed case analysis (report.html, cases.txt, summary.txt) and figures.

Submission layout (GDrive format):
  outputs/Team5/week4/QST1/method1/result.pkl
  outputs/Team5/week4/QST1/method1/NNNNN.png
  (Mirrored to Team5/week4/QST1/method1/ for direct GDrive upload)
"""

import argparse
import html
import os
import pickle
import shutil
import sys
import time
from multiprocessing import Pool

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import keypoints as K
from src.denoise import is_noisy
from src.descriptors import image_id, list_images, read_image
from src.evaluation import apk, mapk
from src.painting_retrieval import EXTRACTORS, crop_painting, find_paintings

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

K_TOP = 10       # results per painting in result.pkl
SHOWN = 5        # museum images shown in diagnostic plots
THUMB = 220      # longest side of thumbnails
SEVERITY = {0: "#2e7d32", 1: "#ef6c00", 2: "#c62828", 3: "#6a1b9a"}

# Worker globals for multiprocessing
_DB = None
_CONTRAST = 0.02
_METHOD = "RootSIFT"
_RATIO = 0.75


def _init_worker(db, contrast, method, ratio):
    global _DB, _CONTRAST, _METHOD, _RATIO
    _DB = db
    _CONTRAST = contrast
    _METHOD = method
    _RATIO = ratio
    K.CONTRAST = contrast


def _score_painting(gray):
    """Number of verified inliers of one query painting against every BBDD museum image."""
    pts, desc = K.describe(gray, _METHOD)
    scores = np.zeros(len(_DB), np.int32)
    for j, (pts_db, desc_db) in enumerate(_DB):
        match = K.knn_matches(desc, desc_db, _METHOD)
        scores[j] = K.score(match, pts, pts_db, ratio=_RATIO, ransac=True)
    return scores, len(pts)


def thumbnail(img_bgr):
    """RGB thumbnail with max side THUMB."""
    scale = THUMB / max(img_bgr.shape[:2])
    return cv2.cvtColor(cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def photo_with_masks(img_bgr, masks):
    """Thumbnail of the photo with masks tinted."""
    thumb = thumbnail(img_bgr).astype(np.float32)
    colours = ((0, 160, 0), (0, 90, 230))
    for mask, colour in zip(masks, colours):
        small = cv2.resize(mask.astype(np.uint8), (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        thumb[small] = 0.6 * thumb[small] + 0.4 * np.array(colour, np.float32)
    return thumb.astype(np.uint8)


def show(ax, rgb, title, colour=None, linestyle="-"):
    """Draw image on matplotlib axis."""
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
            spine.set_linestyle(linestyle)


def judge(found, correct, predictions, best_scores, threshold, fallback):
    """Determine (severity, status) for a query image."""
    problems = []
    if fallback:
        problems.append((3, "no painting found: fallback to whole image"))
    if correct is not None and found != len(correct):
        problems.append((3, f"{found} paintings found, {len(correct)} in ground truth"))

    for j in range(found):
        pred = predictions[j]
        score = best_scores[j]
        is_unknown = (pred == [-1])

        if correct is not None and j < len(correct):
            c = correct[j]
            if c == -1:
                if not is_unknown:
                    problems.append((2, f"painting {j}: expected UNKNOWN, got bbdd_{pred[0]:05d} ({score} inliers)"))
            else:
                if is_unknown:
                    problems.append((2, f"painting {j}: wrongly rejected as UNKNOWN (best was bbdd_{c:05d}? score {score})"))
                elif pred[0] != c:
                    problems.append((2, f"painting {j}: wrong match bbdd_{pred[0]:05d} instead of bbdd_{c:05d}"))
        else:
            # Blind test heuristics
            if abs(score - threshold) <= 5:
                problems.append((1, f"painting {j}: borderline score ({score} inliers, threshold {threshold})"))

    if not problems:
        if correct is not None:
            return 0, "all correct"
        status_strs = []
        for j in range(found):
            if predictions[j] == [-1]:
                status_strs.append(f"p{j}: UNKNOWN ({best_scores[j]} inl)")
            else:
                status_strs.append(f"p{j}: #{predictions[j][0]} ({best_scores[j]} inl)")
        return 0, "; ".join(status_strs)

    return max(p[0] for p in problems), "; ".join(p[1] for p in problems)


def plot_case(stem, photo, masks, crops, full_ranks, best_scores, predictions, correct, threshold, status, museum_thumb, path):
    """Diagnostic plot for one query image."""
    rows = 1 + len(crops)
    fig, axes = plt.subplots(rows, 1 + SHOWN, figsize=(2.0 * (1 + SHOWN), 2.3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    show(axes[0][0], photo_with_masks(photo, masks), f"image {stem}")

    for j, crop in enumerate(crops):
        row = 1 + j
        ranking = full_ranks[j]
        pred = predictions[j]
        best_s = best_scores[j]
        is_unk = (pred == [-1])
        c = correct[j] if correct is not None and j < len(correct) else None

        crop_color = "#c62828" if is_unk else "#2e7d32"
        crop_title = f"p{j}: {'UNKNOWN' if is_unk else f'#{pred[0]}'}\n({best_s} inliers)"
        if c is not None:
            crop_title += f"\nGT: {'UNK' if c == -1 else f'#{c}'}"
        show(axes[row][0], thumbnail(crop), crop_title, colour=crop_color)

        if is_unk:
            # Show best non-match
            best_id = ranking[0]
            show(axes[row][1], museum_thumb(best_id), f"Best non-match\n#{best_id} ({best_s} < {threshold})",
                 colour="#c62828", linestyle="--")
            for r in range(1, SHOWN):
                axes[row][1 + r].text(0.5, 0.5, "UNKNOWN\n(rejected)", ha="center", va="center",
                                      fontsize=9, color="#718096", transform=axes[row][1 + r].transAxes)
        else:
            for r in range(SHOWN):
                ident = ranking[r]
                is_hit = (ident == c) if c is not None and c != -1 else (r == 0)
                col = "#2e7d32" if is_hit else None
                lbl = f"#{r + 1}  bbdd_{ident:05d}"
                if r == 0:
                    lbl += f" ({best_s} inl)"
                show(axes[row][1 + r], museum_thumb(ident), lbl, colour=col)

    fig.suptitle(f"image {stem}: {status}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="folder containing BBDD and query sets")
    parser.add_argument("--query-dir", default=None, help="direct path to query folder (e.g. data/w4/qst1_w4)")
    parser.add_argument("--query-set", default="qst1_w4", help="query set name if query-dir not specified")
    parser.add_argument("--no-segment", action="store_true", help="queries are cropped, whole image is painting")
    parser.add_argument("--extractor", default="watershed", choices=list(EXTRACTORS), help="mask extractor")
    parser.add_argument("--filter", default="median 3x3", help="noise filter for noisy images ('none' to skip)")
    parser.add_argument("--methods", nargs="+", default=["RootSIFT"], choices=list(K.FEATURES),
                        help="local feature methods to evaluate; method1 is the first and submitted")
    parser.add_argument("--contrast", type=float, default=K.CONTRAST, help="SIFT contrast threshold")
    parser.add_argument("--ratio", type=float, default=K.RATIO, help="Lowe's ratio threshold")
    parser.add_argument("--threshold", type=int, default=K.UNKNOWN_THRESHOLD, help="unknown painting threshold in inliers")
    parser.add_argument("--max-side", type=int, default=K.MAX_SIDE, help="max side for keypoint images")
    parser.add_argument("--k", type=int, default=K_TOP, help="top-K matching museum IDs")
    parser.add_argument("--workers", type=int, default=4, help="number of worker processes for matching")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "Team5", "week4"), help="submission output root")
    parser.add_argument("--delivery-dir", default=os.path.join("Team5", "week4"), help="mirror folder matching GDrive layout")
    parser.add_argument("--analysis-dir", default=os.path.join("outputs", "week4", "analysis"))
    parser.add_argument("--save-masks", action=argparse.BooleanOptionalAction, default=True, help="save binary masks")
    parser.add_argument("--reuse-matches", action="store_true", help="reuse existing matches.pkl if available")
    args = parser.parse_args()

    # Resolve BBDD folder
    bbdd_dir = os.path.join(args.data, "BBDD")
    if not os.path.isdir(bbdd_dir) and os.path.isdir(os.path.join(args.data, "w1", "BBDD")):
        bbdd_dir = os.path.join(args.data, "w1", "BBDD")
    if not os.path.isdir(bbdd_dir):
        sys.exit(f"Error: BBDD directory not found in {args.data}")

    # Resolve query folder
    query_dir = args.query_dir
    if query_dir is None:
        candidates = [
            os.path.join(args.data, "w4", args.query_set),
            os.path.join(args.data, args.query_set),
            os.path.join("data", "w4", args.query_set),
            os.path.join("data", args.query_set),
        ]
        for c in candidates:
            if os.path.isdir(c):
                query_dir = c
                break
    if query_dir is None or not os.path.isdir(query_dir):
        sys.exit(f"Error: Query directory not found for set '{args.query_set}'. Looked in {candidates}")

    name = os.path.basename(os.path.normpath(query_dir))
    label = name.split("_")[0].upper()
    query_paths = list_images(query_dir)
    print(f"Found {len(query_paths)} query images in {query_dir}", flush=True)

    gt = augmentations = None
    gt_file = os.path.join(query_dir, "gt_corresps.pkl")
    if os.path.exists(gt_file):
        with open(gt_file, "rb") as f:
            gt = pickle.load(f)
        print(f"Loaded ground truth from {gt_file}", flush=True)
    aug_file = os.path.join(query_dir, "augmentations.pkl")
    if os.path.exists(aug_file):
        with open(aug_file, "rb") as f:
            augmentations = pickle.load(f)

    bbdd_paths = list_images(bbdd_dir)
    db_ids = [image_id(p) for p in bbdd_paths]
    print(f"Found {len(bbdd_paths)} BBDD images in {bbdd_dir}", flush=True)

    museum = {}
    bbdd_map = {image_id(p): p for p in bbdd_paths}

    def museum_thumb(ident):
        if ident not in museum:
            museum[ident] = thumbnail(read_image(bbdd_map[ident]))
        return museum[ident]

    # Pre-segment query images and find painting crops
    print("Segmenting paintings and detecting noise...", flush=True)
    processed_queries = []
    total_paintings = 0
    t0_seg = time.time()
    for p in tqdm(query_paths, desc="segmentation", unit="img"):
        stem = os.path.splitext(os.path.basename(p))[0]
        original = read_image(p)
        noisy = is_noisy(original)
        photo, masks = find_paintings(original, not args.no_segment, args.filter, args.extractor)
        fallback = (not args.no_segment) and len(masks) == 1 and bool(masks[0].all())
        crops = [crop_painting(photo, m, margin=0)[0] for m in masks]
        grays = [K.prepare(c, args.max_side) for c in crops]
        processed_queries.append({
            "stem": stem,
            "original": original,
            "photo": photo,
            "masks": masks,
            "crops": crops,
            "grays": grays,
            "noisy": noisy,
            "fallback": fallback,
        })
        total_paintings += len(crops)
    t_seg = time.time() - t0_seg
    print(f"Extracted {total_paintings} paintings across {len(query_paths)} images in {t_seg:.1f}s", flush=True)

    # Prepare output directories
    method_dirs = [os.path.join(args.output_dir, label, f"method{k + 1}") for k in range(len(args.methods))]
    delivery_dirs = [os.path.join(args.delivery_dir, label, f"method{k + 1}") for k in range(len(args.methods))] if args.delivery_dir else []
    for d in method_dirs:
        os.makedirs(d, exist_ok=True)
    for d in delivery_dirs:
        os.makedirs(d, exist_ok=True)

    analysis_dir = os.path.join(args.analysis_dir, name)
    os.makedirs(os.path.join(analysis_dir, "images"), exist_ok=True)

    # Save binary foreground masks
    if args.save_masks and not args.no_segment:
        print("Saving binary foreground masks...", flush=True)
        for item in processed_queries:
            stem = item["stem"]
            union = (np.any(item["masks"], axis=0).astype(np.uint8)) * 255
            for d in method_dirs:
                cv2.imwrite(os.path.join(d, stem + ".png"), union)
            for d in delivery_dirs:
                cv2.imwrite(os.path.join(d, stem + ".png"), union)

    # Run each method
    all_predictions = []  # per method: list of image results
    all_full_ranks = []   # per method: per image: list of full ranking lists
    all_best_scores = []  # per method: per image: list of best inlier counts

    for m_idx, method in enumerate(args.methods):
        print(f"\n--- Running method{m_idx + 1}: {method} ---", flush=True)
        cache_file = os.path.join(method_dirs[m_idx], "matches.pkl")
        if args.reuse_matches and os.path.exists(cache_file):
            print(f"Reusing cached matches from {cache_file}...", flush=True)
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            cur_predictions = cached["predictions"]
            cur_full_ranks = cached["full_ranks"]
            cur_best_scores = cached["best_scores"]
        else:
            K.CONTRAST = args.contrast
            t0_db = time.time()
            print(f"Computing BBDD {method} descriptors...", flush=True)
            db_desc = [K.describe(K.prepare(read_image(p), args.max_side), method)
                       for p in tqdm(bbdd_paths, desc=f"BBDD {method}", unit="img")]
            print(f"BBDD described in {time.time() - t0_db:.1f}s", flush=True)

            all_grays = [g for item in processed_queries for g in item["grays"]]
            print(f"Matching {len(all_grays)} paintings with {args.workers} workers...", flush=True)
            t0_match = time.time()
            with Pool(args.workers, initializer=_init_worker, initargs=(db_desc, args.contrast, method, args.ratio)) as pool:
                match_results = list(tqdm(pool.imap(_score_painting, all_grays), total=len(all_grays), desc="matching", unit="p"))
            print(f"Matching finished in {time.time() - t0_match:.1f}s", flush=True)

            # Assemble predictions
            cur_predictions = []
            cur_full_ranks = []
            cur_best_scores = []
            ptr = 0
            for item in processed_queries:
                n_paint = len(item["grays"])
                img_preds = []
                img_ranks = []
                img_scores = []
                for _ in range(n_paint):
                    scores, n_pts = match_results[ptr]
                    ptr += 1
                    order = np.argsort(-scores, kind="stable")
                    best_score = int(scores[order[0]]) if len(scores) > 0 else 0
                    full_ranked_ids = [int(db_ids[idx]) for idx in order]
                    img_ranks.append(full_ranked_ids)
                    img_scores.append(best_score)

                    if best_score < args.threshold:
                        img_preds.append([-1])
                    else:
                        img_preds.append(full_ranked_ids[:args.k])

                cur_predictions.append(img_preds)
                cur_full_ranks.append(img_ranks)
                cur_best_scores.append(img_scores)

            with open(cache_file, "wb") as f:
                pickle.dump({"predictions": cur_predictions, "full_ranks": cur_full_ranks, "best_scores": cur_best_scores}, f)

        all_predictions.append(cur_predictions)
        all_full_ranks.append(cur_full_ranks)
        all_best_scores.append(cur_best_scores)

        # Save result.pkl
        out_pkl = os.path.join(method_dirs[m_idx], "result.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(cur_predictions, f)
        print(f"Saved result.pkl to {out_pkl}", flush=True)

        if delivery_dirs:
            deliv_pkl = os.path.join(delivery_dirs[m_idx], "result.pkl")
            shutil.copy2(out_pkl, deliv_pkl)
            print(f"Mirrored result.pkl to {deliv_pkl}", flush=True)

        # Verify format
        for img_res in cur_predictions:
            for p_res in img_res:
                if p_res == [-1]:
                    continue
                assert len(p_res) == args.k, f"Expected top-{args.k} results, got {len(p_res)}"
                assert all(isinstance(x, int) for x in p_res), "Expected integer IDs"

    # Generate diagnostic plots and records using method1 (primary)
    print("\nGenerating visual report and diagnostic plots...", flush=True)
    records = []
    primary_preds = all_predictions[0]
    primary_ranks = all_full_ranks[0]
    primary_scores = all_best_scores[0]

    for i, item in enumerate(tqdm(processed_queries, desc="analysis", unit="img")):
        stem = item["stem"]
        correct = [int(c) for c in gt[i]] if gt is not None else None
        severity, status = judge(len(item["crops"]), correct, primary_preds[i], primary_scores[i], args.threshold, item["fallback"])
        extra = f" | noise: {'yes' if item['noisy'] else 'no'}"
        if augmentations is not None and i < len(augmentations):
            extra += f" | aug: {str(augmentations[i])[:40]}"

        img_plot_path = os.path.join(analysis_dir, "images", stem + ".png")
        plot_case(stem, item["photo"], item["masks"], item["crops"], primary_ranks[i],
                  primary_scores[i], primary_preds[i], correct, args.threshold,
                  status + extra, museum_thumb, img_plot_path)

        records.append({
            "stem": stem,
            "noisy": item["noisy"],
            "found": len(item["crops"]),
            "correct": correct,
            "preds": [all_predictions[m][i] for m in range(len(args.methods))],
            "scores": [all_best_scores[m][i] for m in range(len(args.methods))],
            "severity": severity,
            "status": status + extra,
        })

    # Write cases.txt
    has_gt = (gt is not None)
    header = f"{'image':<7}{'paint':<6}{'noisy':<6}"
    if has_gt:
        header += f"{'GT':>7}"
    header += f"{'score':>7}{'top1':>8}  result"
    lines = [header]

    for r in records:
        for j in range(r["found"]):
            pred = r["preds"][0][j]
            score = r["scores"][0][j]
            row = f"{r['stem']:<7}{j:<6}{'yes' if r['noisy'] else 'no':<6}"
            if has_gt:
                c = r["correct"][j] if j < len(r["correct"]) else None
                row += f"{c if c is not None else '-':>7}"
            row += f"{score:>7}"
            top1_str = "[-1]" if pred == [-1] else str(pred[0])
            row += f"{top1_str:>8}  "

            if has_gt:
                c = r["correct"][j] if j < len(r["correct"]) else None
                if c is None:
                    res = "EXTRA"
                elif c == -1:
                    res = "OK (unknown)" if pred == [-1] else "FALSE POSITIVE (called known)"
                else:
                    res = "OK" if pred[0] == c else "WRONG"
            else:
                res = "UNKNOWN" if pred == [-1] else f"MATCH (bbdd_{pred[0]:05d})"
            lines.append(row + res)

    with open(os.path.join(analysis_dir, "cases.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Write summary.txt
    n_paintings = sum(r["found"] for r in records)
    n_unknowns = sum(p == [-1] for r in records for p in r["preds"][0])
    n_knowns = n_paintings - n_unknowns
    n_two = sum(r["found"] == 2 for r in records)
    n_noisy = sum(r["noisy"] for r in records)

    summary = [
        f"Week 4 Submission Summary: {name}",
        f"Images: {len(records)}, Paintings: {n_paintings} ({n_two} images with two paintings)",
        f"Noise detected: {n_noisy} images (filtered with {args.filter})",
        f"Extractor: {args.extractor}, Feature: {args.methods[0]} (contrast={args.contrast}, ratio={args.ratio})",
        f"Unknown threshold: {args.threshold} inliers",
        f"Predictions: {n_knowns} known paintings, {n_unknowns} unknown paintings ([-1])",
        "",
    ]

    if has_gt:
        flat_gt = [c for g in gt for c in g]
        flat_pred = [p for r in records for p in r["preds"][0]]
        actual = [[c] for c in flat_gt]
        pred_k1 = [p[:1] if p != [-1] else [-1] for p in flat_pred]
        pred_k5 = [p[:5] if p != [-1] else [-1] for p in flat_pred]
        m1 = mapk(actual, pred_k1, 1)
        m5 = mapk(actual, pred_k5, 5)

        tp = sum(p == [-1] and a == [-1] for p, a in zip(pred_k1, [a[0] for a in actual]))
        fp = sum(p == [-1] and a != [-1] for p, a in zip(pred_k1, [a[0] for a in actual]))
        fn = sum(p != [-1] and a == [-1] for p, a in zip(pred_k1, [a[0] for a in actual]))
        f1_unk = 2 * tp / max(2 * tp + fp + fn, 1)

        summary.append(f"Ground Truth Evaluation:")
        summary.append(f"  mAP@1: {m1:.4f}")
        summary.append(f"  mAP@5: {m5:.4f}")
        summary.append(f"  Unknown F1: {f1_unk:.4f} (TP={tp}, FP={fp}, FN={fn})")
        summary.append(f"  Segmentation errors: {sum(r['found'] != len(r['correct']) for r in records)} images")
    else:
        summary.append(f"Blind Test Statistics:")
        scores_arr = [s for r in records for s in r["scores"][0]]
        summary.append(f"  Median inliers: {np.median(scores_arr):.1f}")
        summary.append(f"  Min inliers: {np.min(scores_arr)}, Max inliers: {np.max(scores_arr)}")
        summary.append(f"  Borderline paintings (+/-5 of threshold): {sum(abs(s - args.threshold) <= 5 for s in scores_arr)}")

    with open(os.path.join(analysis_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")

    # Write report.html
    body = [f"<h1>{html.escape(name)} - Week 4 Submission Analysis</h1>",
            "<pre style='background:#f7fafc;padding:12px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;'>" +
            html.escape("\n".join(summary)) + "</pre>"]

    for r in sorted(records, key=lambda r: (-r["severity"], r["stem"])):
        col = SEVERITY.get(r["severity"], "#000")
        body.append(f"<div style='margin-bottom:24px;border-bottom:1px solid #edf2f7;padding-bottom:16px;'>")
        body.append(f"<h3 style='color:{col};margin:8px 0;'>Image {r['stem']}: {html.escape(r['status'])}</h3>")
        body.append(f"<img src='images/{r['stem']}.png' style='max-width:1100px;border-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,0.1);'>")
        body.append(f"</div>")

    with open(os.path.join(analysis_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Week 4 Submission Report</title></head>"
                "<body style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;max-width:1200px;margin:20px auto;padding:0 20px;color:#2d3748;'>" +
                "\n".join(body) + "</body></html>")

    print("\n" + "\n".join(summary))
    print(f"\nSaved submission to: {method_dirs[0]}")
    if delivery_dirs:
        print(f"Mirrored for delivery to: {delivery_dirs[0]}")
    print(f"Detailed visual analysis in: {analysis_dir}/report.html")


if __name__ == "__main__":
    main()
