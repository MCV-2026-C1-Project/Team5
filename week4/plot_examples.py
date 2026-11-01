"""Tasks 1-2: the studies and pictures behind the slides (numbers saved to outputs/week4/examples.json).

- noise: the noisy paintings of qsd1_w4, cut with the annotated frames so only the filter changes, described as
  given, after the 3x3 median of Task 1 and from the clean original: keypoints, inliers with the correct museum
  image and its rank.
- matches: one query painting against its museum image and against the best wrong one, before and after RANSAC.
- keypoints: the keypoints of a query painting and of its museum image.
- failure: a painting that is not found, with the museum image it is confused with.
"""

import argparse
import json
import os
import pickle
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import keypoints as K
from src.denoise import denoise, impulse_ratio
from src.descriptors import image_id, list_images, read_image
from src.painting_retrieval import crop_painting, find_paintings

QUERY_DIR = os.path.join("data", "qsd1_w4", "qsd1_w4")
MAX_LINES = 60   # inlier lines drawn per picture
FLAT_PAINTINGS = (11, 164)   # museum paintings with the least texture (uniform canvas, soft gradient)


def frame_crop(img, corners):
    corners = np.array(corners)
    (x0, y0), (x1, y1) = np.maximum(corners.min(0), 0).astype(int), corners.max(0).astype(int)
    return img[y0:y1 + 1, x0:x1 + 1]


def scores_against(gray, db, method):
    pts, desc = K.describe(gray, method)
    return np.array([K.score(K.knn_matches(desc, d, method), pts, p) for p, d in db]), len(pts)


def draw_matches(gray_q, gray_db, method, ratio, ransac):
    """Side-by-side picture with the matches kept (green) and the tentative ones dropped by RANSAC (red)."""
    pq, dq = K.describe(gray_q, method)
    pd, dd = K.describe(gray_db, method)
    match = K.knn_matches(dq, dd, method)
    tentative = K.verified_matches(match, pq, pd, ratio=ratio, ransac=False)
    kept = K.verified_matches(match, pq, pd, ratio=ratio, ransac=True) if ransac else tentative
    h = max(gray_q.shape[0], gray_db.shape[0])
    canvas = np.full((h, gray_q.shape[1] + gray_db.shape[1] + 20, 3), 255, np.uint8)
    canvas[:gray_q.shape[0], :gray_q.shape[1]] = cv2.cvtColor(gray_q, cv2.COLOR_GRAY2BGR)
    off = gray_q.shape[1] + 20
    canvas[:gray_db.shape[0], off:] = cv2.cvtColor(gray_db, cv2.COLOR_GRAY2BGR)
    q, t = match[0], match[1]
    shown = np.random.default_rng(0).permutation(kept)[:MAX_LINES]   # a sample, or the lines hide the paintings
    for idx, colour in ((np.setdiff1d(tentative, kept), (40, 40, 220)), (shown, (40, 170, 40))):
        for i in idx:
            a = tuple(int(v) for v in pq[q[i]]); b = (int(pd[t[i]][0]) + off, int(pd[t[i]][1]))
            cv2.line(canvas, a, b, colour, 2, cv2.LINE_AA)
    return canvas, len(tentative), len(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--method", default="RootSIFT")
    parser.add_argument("--ratio", type=float, default=K.RATIO)
    parser.add_argument("--example", type=int, default=2, help="qsd1_w4 image for the matches picture")
    parser.add_argument("--failure", type=int, nargs=2, default=None, metavar=("IMAGE", "PAINTING"),
                        help="qsd1_w4 image and painting index of a failure to draw")
    parser.add_argument("--skip-noise", action="store_true", help="keep the noise study of a previous run")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week4"))
    parser.add_argument("--figures-dir", default=os.path.join("slides", "figures"))
    args = parser.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    gt = pickle.load(open(os.path.join(QUERY_DIR, "gt_corresps.pkl"), "rb"))
    aug = pickle.load(open(os.path.join(QUERY_DIR, "augmentations.pkl"), "rb"))
    frames = pickle.load(open(os.path.join(QUERY_DIR, "frames.pkl"), "rb"))
    paths = list_images(QUERY_DIR)
    db_paths = list_images(os.path.join(args.data, "BBDD"))
    db_ids = [image_id(p) for p in db_paths]
    db_grays = [K.prepare(read_image(p)) for p in db_paths]
    db = [K.describe(g, args.method) for g in db_grays]
    col = {i: j for j, i in enumerate(db_ids)}
    out_path = os.path.join(args.output_dir, "examples.json")
    out = json.load(open(out_path)) if args.skip_noise and os.path.exists(out_path) else {}

    # Noise study.
    rows = out.get("noise", []) if args.skip_noise else []
    for i, (p, a) in enumerate(zip(paths, aug)):
        if args.skip_noise or "Noise" not in str(a):
            continue
        noisy = read_image(p)
        clean = read_image(os.path.join(QUERY_DIR, "non_augmented", os.path.basename(p)))
        versions = {"noisy": noisy, "median 3x3": denoise(noisy, "median 3x3"), "clean": clean}
        for j, (c, (_, corners)) in enumerate(zip(gt[i], frames[i])):
            if c == -1:
                continue
            row = dict(image=i, painting=j, correct=int(c), impulse_ratio=impulse_ratio(noisy))
            for name, img in versions.items():
                s, n = scores_against(K.prepare(frame_crop(img, corners)), db, args.method)
                rank = 1 + int(np.sum(s > s[col[c]]))
                row[name] = dict(keypoints=n, inliers=int(s[col[c]]), best_wrong=int(np.delete(s, col[c]).max()), rank=rank)
            rows.append(row)
            print(f"noise: image {i} painting {j}: " + "  ".join(
                f"{k} kp {row[k]['keypoints']} inl {row[k]['inliers']} wrong {row[k]['best_wrong']} rank {row[k]['rank']}"
                for k in versions), flush=True)
    out["noise"] = rows
    if rows:
        names = ["noisy", "median 3x3", "clean"]
        med = {k: float(np.median([r[k]["inliers"] for r in rows])) for k in names}
        out["noise_summary"] = out.get("noise_summary") if args.skip_noise else None
        out["noise_summary"] = out["noise_summary"] or dict(n=len(rows), inliers_median=med,
                                    keypoints_median={k: float(np.median([r[k]["keypoints"] for r in rows])) for k in names},
                                    rank1={k: int(sum(r[k]["rank"] == 1 for r in rows)) for k in names})
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for r in rows:
            ax.plot(range(3), [max(r[k]["inliers"], 0.8) for k in names], color="#a0aec0", lw=1, marker="o", ms=3)
        ax.plot(range(3), [max(med[k], 0.8) for k in names], color="black", lw=2.5, marker="o", label="median")
        ax.axhline(K.UNKNOWN_THRESHOLD, color="#c53030", ls="--", lw=1)
        ax.text(2.05, K.UNKNOWN_THRESHOLD, f"unknown threshold {K.UNKNOWN_THRESHOLD}", color="#c53030", fontsize=8, va="bottom", ha="right")
        ax.set_xticks(range(3), names); ax.set_yscale("log"); ax.set_ylabel("inliers with the correct image")
        ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(args.figures_dir, "noise.png"), dpi=150); plt.close(fig)

    # Keypoints of the flattest museum paintings for each contrast threshold of the DoG detector.
    out["contrast"] = {}
    for i in FLAT_PAINTINGS:
        g = db_grays[col[i]]
        out["contrast"][str(i)] = {str(c): len(cv2.SIFT_create(nfeatures=K.N_FEATURES, contrastThreshold=c).detect(g, None))
                                   for c in (0.04, 0.02, 0.01)}
    print("contrast:", out["contrast"])

    # Keypoints and matches of one example (crops from our own segmentation, as in the pipeline).
    img, masks = find_paintings(read_image(paths[args.example]), extractor="watershed")
    gray_q = K.prepare(crop_painting(img, masks[0], margin=0)[0])
    c = gt[args.example][0]
    pq, dq = K.describe(gray_q, args.method)
    s = np.array([K.score(K.knn_matches(dq, d, args.method), pq, p, ratio=args.ratio) for p, d in db])
    wrong = int(np.argmax(np.where(np.arange(len(s)) == col[c], -1, s)))
    pic_ok, tent_ok, kept_ok = draw_matches(gray_q, db_grays[col[c]], args.method, args.ratio, True)
    pic_bad, tent_bad, kept_bad = draw_matches(gray_q, db_grays[wrong], args.method, args.ratio, True)
    cv2.imwrite(os.path.join(args.figures_dir, "matches_correct.jpg"), pic_ok)
    cv2.imwrite(os.path.join(args.figures_dir, "matches_wrong.jpg"), pic_bad)
    kp_pic = cv2.drawKeypoints(cv2.cvtColor(gray_q, cv2.COLOR_GRAY2BGR),
                               [cv2.KeyPoint(float(x), float(y), 8) for x, y in pq], None, (0, 160, 255))
    cv2.imwrite(os.path.join(args.figures_dir, "keypoints.jpg"), kp_pic)
    out["example"] = dict(image=args.example, correct=int(c), wrong=int(db_ids[wrong]), keypoints=len(pq),
                          correct_tentative=tent_ok, correct_inliers=kept_ok, wrong_tentative=tent_bad, wrong_inliers=kept_bad)
    print("example:", out["example"])

    # Failure case.
    if args.failure:
        i, j = args.failure
        img, masks = find_paintings(read_image(paths[i]), extractor="watershed")
        gray_f = K.prepare(crop_painting(img, masks[min(j, len(masks) - 1)], margin=0)[0])
        pf, df = K.describe(gray_f, args.method)
        s = np.array([K.score(K.knn_matches(df, d, args.method), pf, p, ratio=args.ratio) for p, d in db])
        top = int(np.argmax(s))
        c = gt[i][j]
        canvas = [cv2.resize(cv2.cvtColor(gray_f, cv2.COLOR_GRAY2BGR), None, fx=0.6, fy=0.6),
                  cv2.resize(cv2.cvtColor(db_grays[top], cv2.COLOR_GRAY2BGR), None, fx=0.6, fy=0.6)]
        if c != -1:
            canvas.append(cv2.resize(cv2.cvtColor(db_grays[col[c]], cv2.COLOR_GRAY2BGR), None, fx=0.6, fy=0.6))
        h = max(x.shape[0] for x in canvas)
        canvas = np.hstack([cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in canvas])
        cv2.imwrite(os.path.join(args.figures_dir, "failure.jpg"), canvas)
        out["failure"] = dict(image=i, painting=j, correct=int(c), top=int(db_ids[top]), top_score=int(s[top]),
                              correct_score=int(s[col[c]]) if c != -1 else None, keypoints=len(pf),
                              rank=1 + int(np.sum(s > s[col[c]])) if c != -1 else None)
        print("failure:", out["failure"])

    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved {os.path.join(args.output_dir, 'examples.json')}")


if __name__ == "__main__":
    main()
