"""Task 3: the delivered Week 3 method against the Week 4 keypoint method, on qsd1_w4.

Both systems get exactly the same paintings: every photo is denoised if it is noisy and its paintings are found
with watershed (as in Week 3 and in week4/compute_scores.py), so the only thing that changes is how a painting
is described and ranked.

- Week 3 (delivered, README "The delivered files"): crop to the mask, cut a 10% margin from the crop and from the
  museum images, and average the median-scaled distances of DCT (cosine) + Haar wavelet (L1) + colour (YCrCb
  pyramid 1-2-4, 32 bins, chi-squared). It always returns 10 museum IDs: it cannot say "unknown".
- Week 4: RootSIFT (contrast 0.02), ratio test 0.75, RANSAC homography, score = inliers; fewer inliers than the
  threshold in the best match -> unknown ([-1]).

qsd1_w4 has paintings that are not in the museum (ground truth -1), so the two systems are compared in three ways:
all paintings (Week 3 scores 0 on every unknown, it has no way to reject), only the known paintings (the fair
retrieval comparison) and, for Week 4, the unknown-detection F1. Week 4 is also shown without the unknown test.

Run from the root of the repository:
    python week4/compare_w3_vs_w4.py
    python week4/compare_w3_vs_w4.py --workers 2 --threshold 44
"""

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import keypoints as K
from src.descriptors import image_id, list_images, read_image
from src.evaluation import mapk
from src.painting_retrieval import (MARGIN, crop_painting, describe_database, find_paintings, fuse,
                                    painting_distances, rankings)

try:
    from tqdm import tqdm
except ImportError:   # pip install tqdm for a progress bar
    def tqdm(iterable, **kwargs):
        return iterable

W3_NAMES = ("DCT", "wavelet", "colour")   # the delivered Week 3 fusion (the 4-stream one also has "multiscale LBP")
METHOD = "RootSIFT"

_DB = None   # (points, descriptors) of every museum image, per worker


def _init(db, contrast):
    global _DB
    _DB = db
    K.CONTRAST = contrast   # spawned workers re-import the module, so the setting is passed again


def _inliers(gray):
    """Number of verified matches of one painting (grey, already resized) against every museum image."""
    pts, desc = K.describe(gray, METHOD)
    scores = np.zeros(len(_DB), np.int32)
    for j, (pts_db, desc_db) in enumerate(_DB):
        scores[j] = K.score(K.knn_matches(desc, desc_db, METHOD), pts, pts_db)
    return scores


def rank_of(ranking, correct):
    """1-based position of the correct ID in a full ranking, or None."""
    return ranking.index(correct) + 1 if correct in ranking else None


def keypoint_ranking(scores, db_ids, k, threshold):
    """IDs of the k best museum images by inliers, or [-1] if the best score is below the threshold."""
    return K.rank(scores, db_ids, k, threshold)


def summarise(actual, predicted, rows_known):
    """mAP@1, mAP@5 over all paintings and over the known ones only."""
    out = {"all": (mapk(actual, predicted, 1), mapk(actual, predicted, 5))}
    out["known"] = (mapk([actual[i] for i in rows_known], [predicted[i] for i in rows_known], 1),
                    mapk([actual[i] for i in rows_known], [predicted[i] for i in rows_known], 5))
    return out


def unknown_f1(actual, predicted):
    """F1 of 'unknown' detection (positive = painting not in the museum) and the three counts."""
    tp = sum(p[:1] == [-1] and a == [-1] for p, a in zip(predicted, actual))
    fp = sum(p[:1] == [-1] and a != [-1] for p, a in zip(predicted, actual))
    fn = sum(p[:1] != [-1] and a == [-1] for p, a in zip(predicted, actual))
    return 2 * tp / max(2 * tp + fp + fn, 1), tp, fp, fn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="folder containing BBDD")
    parser.add_argument("--query-dir", default=os.path.join("data", "qsd1_w4", "qsd1_w4"),
                        help="folder with NNNNN.jpg and gt_corresps.pkl")
    parser.add_argument("--margin", type=float, default=MARGIN, help="Week 3 crop margin")
    parser.add_argument("--threshold", type=int, default=K.UNKNOWN_THRESHOLD, help="Week 4 unknown threshold (inliers)")
    parser.add_argument("--contrast", type=float, default=K.CONTRAST, help="SIFT contrast threshold")
    parser.add_argument("--max-side", type=int, default=K.MAX_SIDE, help="longer side of the images for keypoints")
    parser.add_argument("--filter", default="median 3x3", help="noise filter for the noisy images ('none' to skip)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week4"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    K.CONTRAST = args.contrast

    with open(os.path.join(args.query_dir, "gt_corresps.pkl"), "rb") as f:
        gt = pickle.load(f)   # per image: the correct ID of each painting (-1 = not in the museum)
    paths = list_images(args.query_dir)
    if len(gt) != len(paths):
        raise ValueError(f"{len(gt)} ground-truth entries but {len(paths)} images in {args.query_dir}")

    bbdd_dir = os.path.join(args.data, "BBDD")
    bbdd_paths = list_images(bbdd_dir)
    db_ids = [image_id(p) for p in bbdd_paths]

    # Same paintings for both systems: denoise + watershed, then one crop per system.
    photos = []   # per image: (denoised photo, masks)
    for p in tqdm(paths, desc="paintings", unit="img"):
        photos.append(find_paintings(read_image(p), True, args.filter, extractor="watershed"))
    counts = [len(masks) for _, masks in photos]

    # Week 3: texture + colour fusion.
    t0 = time.time()
    w3_db, w3_ids = describe_database(bbdd_dir, W3_NAMES, args.margin)
    assert list(w3_ids) == db_ids
    t_w3_db = time.time() - t0
    t0 = time.time()
    w3_ranks = []   # per image, per painting: the full ranking
    for img, masks in photos:
        dist = painting_distances([crop_painting(img, m, args.margin) for m in masks], w3_db)
        w3_ranks.append(rankings(fuse(dist, W3_NAMES), db_ids, len(db_ids)))
    t_w3 = (time.time() - t0) / max(sum(counts), 1)

    # Week 4: RootSIFT + ratio + RANSAC inliers against every museum image.
    t0 = time.time()
    db = [K.describe(K.prepare(read_image(p), args.max_side), METHOD) for p in tqdm(bbdd_paths, desc="BBDD keypoints", unit="img")]
    t_w4_db = time.time() - t0
    grays = [K.prepare(crop_painting(img, m, margin=0)[0], args.max_side) for img, masks in photos for m in masks]
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(db, args.contrast)) as pool:
        w4_scores = np.stack(list(tqdm(pool.imap(_inliers, grays), total=len(grays), desc="matching", unit="painting")))
    t_w4 = (time.time() - t0) / max(len(grays), 1)

    # Line up the paintings of the ground truth with the paintings found (reading order), like mapk_paintings.
    cases = []   # (image, painting index, correct ID, row of w4_scores or None, W3 ranking or [])
    start = 0
    for i, (ids, n) in enumerate(zip(gt, counts)):
        for j, c in enumerate(ids):
            c = int(c)
            cases.append((i, j, c, start + j if j < n else None, w3_ranks[i][j] if j < n else []))
        start += n
    actual = [[c] for _, _, c, _, _ in cases]
    known = [idx for idx, (_, _, c, _, _) in enumerate(cases) if c != -1]

    w3_pred = [r[:args.k] for _, _, _, _, r in cases]
    w4_pred = [keypoint_ranking(w4_scores[row], db_ids, args.k, args.threshold) if row is not None else []
               for _, _, _, row, _ in cases]
    w4_raw = [keypoint_ranking(w4_scores[row], db_ids, args.k, None) if row is not None else []
              for _, _, _, row, _ in cases]

    systems = [("Week 3: DCT + wavelet + colour", w3_pred),
               (f"Week 4: RootSIFT, unknown < {args.threshold} inliers", w4_pred),
               ("Week 4: RootSIFT, no unknown test", w4_raw)]
    n_unknown = len(cases) - len(known)
    lines = [f"qsd1_w4: {len(paths)} images, {len(cases)} paintings in the ground truth ({n_unknown} not in the museum), "
             f"{sum(counts)} paintings found by watershed ({sum(n != len(g) for n, g in zip(counts, gt))} images with a wrong count)",
             f"one painting = {1 / len(cases):.3f} of mAP@1 over all, {1 / max(len(known), 1):.3f} over the known ones", "",
             f"{'system':<42}{'all mAP@1/@5':>18}{'known mAP@1/@5':>20}{'unknown F1':>13}"]
    for name, pred in systems:
        s = summarise(actual, pred, known)
        f1 = f"{unknown_f1(actual, pred)[0]:.3f}" if "unknown <" in name else "-"
        lines.append(f"{name:<42}{s['all'][0]:>9.3f} /{s['all'][1]:.3f}{s['known'][0]:>11.3f} /{s['known'][1]:.3f}{f1:>13}")
    f1, tp, fp, fn = unknown_f1(actual, w4_pred)
    lines += ["", f"Week 4 unknown detection: {tp} unknown rejected, {fn} unknown accepted (missed), {fp} known wrongly rejected",
              "Week 3 cannot reject: its 'all paintings' score counts every unknown as a miss, so compare the 'known' columns.",
              "", f"time per painting: Week 3 {t_w3:.2f} s (museum described once in {t_w3_db:.0f} s), "
                  f"Week 4 {t_w4:.2f} s with {args.workers} workers (museum described once in {t_w4_db:.0f} s)",
              "", f"{'img':<5}{'paint':<6}{'GT':>5}  {'W3 top1':>8}{'W3 rank':>8}  {'W4 top1':>8}{'W4 rank':>8}{'inliers':>8}  result"]
    for idx, (i, j, c, row, w3r) in enumerate(cases):
        w3_top = w3r[0] if w3r else None
        w4_top = w4_pred[idx][0] if w4_pred[idx] else None
        full4 = None
        inl = None
        if row is not None:
            s = w4_scores[row]
            order = [int(db_ids[x]) for x in np.argsort(-s, kind="stable")]
            full4 = rank_of(order, c)
            inl = int(s.max())
        w3_ok, w4_ok = w3_top == c, w4_top == c
        verdict = ("both right" if w3_ok and w4_ok else "only Week 3 right" if w3_ok else
                   "only Week 4 right" if w4_ok else "both wrong")
        if row is None:
            verdict += " (painting not found)"
        fmt = lambda v: "-" if v is None else str(v)
        lines.append(f"{i:<5}{j:<6}{c:>5}  {fmt(w3_top):>8}{fmt(rank_of(w3r, c)):>8}  {fmt(w4_top):>8}{fmt(full4):>8}{fmt(inl):>8}  {verdict}")

    text = "\n".join(lines)
    print("\n" + text)
    out = os.path.join(args.output_dir, "comparison_w3_vs_w4.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()