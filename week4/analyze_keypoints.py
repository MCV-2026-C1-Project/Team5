"""Tasks 1-2: from the score matrices of compute_scores.py to the decisions, tables, figures and results.json.

Choices are made on qsd1_w4 only; qsd2_w3 and qsd1_w3 are kept to check them. These have no unknown paintings, so
unknowns are simulated: a painting whose correct museum image is removed from the database is a query that is not
in it, and its best score is the best score among the wrong museum images.

1. Local feature: mAP of the known paintings of qsd1_w4 with each feature (ratio 0.75 + RANSAC for all).
2. Ratio test and RANSAC: for every ratio, with and without RANSAC, the mAP and the margin between the score of the
   correct museum image and the best wrong one.
3. Unknown threshold: the minimum number of inliers of the best match, chosen by F1 of the unknown detection on
   qsd1_w4 (the middle of the best interval), and tested on the two other sets.
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.evaluation import mapk

GT_DIRS = {"qsd1_w4": os.path.join("data", "qsd1_w4", "qsd1_w4"),
           "qsd2_w3": os.path.join("data", "qsd2_w3", "qsd2_w3"),
           "qsd1_w3": os.path.join("data", "qsd1_w3", "qsd1_w3")}
METHODS = ["SIFT", "RootSIFT", "ORB", "AKAZE"]
DEFAULT = "r0.75_ransac"
CONTRASTS = {0.04: "_c0.04", 0.02: "_c0.02", 0.01: "_c0.01"}   # SIFT contrast threshold -> suffix of the score files


def feature_tag(method, contrast=0.04):
    """Suffix of the score files: the contrast threshold, for the SIFT family only (the features are compared at
    OpenCV's default 0.04)."""
    return CONTRASTS[contrast] if "SIFT" in method else ""
COLORS = {"SIFT": "#2b6cb0", "RootSIFT": "#2f855a", "ORB": "#c05621", "AKAZE": "#6b46c1"}


def load(scores_dir, method, query_set, tag=""):
    path = os.path.join(scores_dir, f"{method}_{query_set}{tag}.npz")
    return dict(np.load(path)) if os.path.exists(path) else None


def painting_rows(gt, counts):
    """(row of the score matrix or None, correct ID) for every painting of the ground truth, in order."""
    out, start = [], 0
    for ids, n in zip(gt, counts):
        out += [(start + j if j < n else None, c) for j, c in enumerate(ids)]
        start += n
    return out


def evaluate(S, db_ids, rows, threshold=None, k=10):
    """Rankings of every GT painting (with [-1] below the threshold), mAP@1/@5 and the unknown-detection F1."""
    predicted = []
    for r, _ in rows:
        if r is None:
            predicted.append([])
            continue
        s = S[r]
        if threshold is not None and s.max() < threshold:
            predicted.append([-1])
        else:
            predicted.append([int(db_ids[i]) for i in np.argsort(-s, kind="stable")[:k]])
    actual = [[c] for _, c in rows]
    tp = sum(p[:1] == [-1] and a == [-1] for p, a in zip(predicted, actual))
    fp = sum(p[:1] == [-1] and a != [-1] for p, a in zip(predicted, actual))
    fn = sum(p[:1] != [-1] and a == [-1] for p, a in zip(predicted, actual))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return predicted, mapk(actual, predicted, 1), mapk(actual, predicted, 5), f1


def known_scores(S, db_ids, rows):
    """For each known painting with a row: (score of the correct image, best wrong score, rank of the correct)."""
    col = {int(i): j for j, i in enumerate(db_ids)}
    out = []
    for r, c in rows:
        if r is None or c == -1:
            continue
        s = S[r]
        correct = s[col[c]]
        wrong = np.delete(s, col[c]).max()
        rank = 1 + int(np.sum(s > correct)) + int(np.sum(s[:col[c]] == correct))   # stable order on ties
        out.append((int(correct), int(wrong), rank))
    return np.array(out).reshape(-1, 3)


def unknown_best(S, rows):
    """Best score of every real unknown painting with a row."""
    return np.array([int(S[r].max()) for r, c in rows if r is not None and c == -1])


def f1_curve(pos, neg, thresholds):
    """F1 of 'unknown' detection for every threshold: pos = best scores of unknowns, neg = best scores of knowns."""
    out = []
    for t in thresholds:
        tp, fn, fp = np.sum(pos < t), np.sum(pos >= t), np.sum(neg < t)
        out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", default=os.path.join("outputs", "week4", "scores"))
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week4"))
    parser.add_argument("--figures-dir", default=os.path.join("slides", "figures"))
    args = parser.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)
    gts = {s: pickle.load(open(os.path.join(d, "gt_corresps.pkl"), "rb")) for s, d in GT_DIRS.items()}
    R, lines = {}, []

    # 1. Local feature (ratio 0.75 + RANSAC for all), known paintings of qsd1_w4.
    lines.append("== 1. Local features: known paintings of qsd1_w4, ratio 0.75 + RANSAC ==")
    R["features"] = {}
    for m in METHODS:
        d = load(args.scores_dir, m, "qsd1_w4", feature_tag(m))
        if d is None:
            continue
        rows = painting_rows(gts["qsd1_w4"], d["counts"])
        known = [(r, c) for r, c in rows if c != -1]
        _, m1, m5, _ = evaluate(d[DEFAULT], d["db_ids"], known)
        ks = known_scores(d[DEFAULT], d["db_ids"], rows)
        val = {}
        for s in ("qsd2_w3", "qsd1_w3"):
            dv = load(args.scores_dir, m, s, feature_tag(m))
            if dv is not None:
                _, v1, v5, _ = evaluate(dv[DEFAULT], dv["db_ids"], painting_rows(gts[s], dv["counts"]))
                val[s] = (v1, v5)
        R["features"][m] = dict(map1=m1, map5=m5, n_known=len(known), correct_median=float(np.median(ks[:, 0])),
                                wrong_max_median=float(np.median(ks[:, 1])), seconds_per_query=float(d["seconds_per_query"]),
                                keypoints_query_median=float(np.median(d["n_keypoints"])),
                                keypoints_db_median=float(np.median(d["n_keypoints_db"])), validation=val)
        lines.append(f"{m:<9} mAP@1 {m1:.3f} mAP@5 {m5:.3f}  inliers correct (median) {np.median(ks[:, 0]):.0f}  "
                     f"best wrong (median) {np.median(ks[:, 1]):.0f}  {float(d['seconds_per_query']):.2f} s/painting  "
                     f"keypoints q/db {np.median(d['n_keypoints']):.0f}/{np.median(d['n_keypoints_db']):.0f}  "
                     + "  ".join(f"{s} {v[0]:.3f}/{v[1]:.3f}" for s, v in val.items()))
    best_feature = max(R["features"], key=lambda m: (R["features"][m]["map1"], R["features"][m]["map5"],
                                                     -R["features"][m]["seconds_per_query"]))
    R["best_feature"] = best_feature
    lines.append(f"-> chosen feature: {best_feature}")

    # 1b. SIFT contrast threshold (low-contrast paintings have few DoG extrema above it).
    lines += ["", f"== 1b. Contrast threshold of the DoG detector ({best_feature}, ratio 0.75 + RANSAC) =="]
    R["contrast"] = {}
    for c, tag in CONTRASTS.items():
        d = load(args.scores_dir, best_feature, "qsd1_w4", tag)
        if d is None:
            continue
        rows = painting_rows(gts["qsd1_w4"], d["counts"])
        known = [(r, c_) for r, c_ in rows if c_ != -1]
        _, m1, m5, _ = evaluate(d[DEFAULT], d["db_ids"], known)
        ks = known_scores(d[DEFAULT], d["db_ids"], rows)
        weak = [int(x) for x in np.sort(ks[:, 0])[:3]]
        unk_c = unknown_best(d[DEFAULT], rows)
        best_known_c = np.array([d[DEFAULT][r].max() for r, c_ in known if r is not None])
        f1_c = f1_curve(unk_c, best_known_c, np.arange(1, 101))
        plateau_c = np.arange(1, 101)[f1_c == f1_c.max()]
        t_c = int(np.round(np.median(plateau_c)))
        val = {}
        for s_ in ("qsd2_w3", "qsd1_w3"):
            dv = load(args.scores_dir, best_feature, s_, tag)
            if dv is not None:
                rv = painting_rows(gts[s_], dv["counts"])
                _, v1, v5, _ = evaluate(dv[DEFAULT], dv["db_ids"], rv)
                kv = known_scores(dv[DEFAULT], dv["db_ids"], rv)
                kb = np.array([dv[DEFAULT][r].max() for r, _ in rv if r is not None])
                val[s_] = (v1, v5, float(f1_curve(kv[:, 1], kb, [t_c])[0]))
        R["contrast"][c] = dict(map1=m1, map5=m5, keypoints_db_median=float(np.median(d["n_keypoints_db"])),
                                keypoints_db_min=int(np.min(d["n_keypoints_db"])), weakest_correct=weak,
                                seconds_per_query=float(d["seconds_per_query"]), validation=val,
                                correct_median=float(np.median(ks[:, 0])), plateau=[int(plateau_c.min()), int(plateau_c.max())],
                                threshold=t_c,
                                plateau_f1=float(f1_c.max()))
        lines.append(f"contrast {c:<5} unknown-threshold interval with the best F1 ({f1_c.max():.3f}): "
                     f"{plateau_c.min()}-{plateau_c.max()}; correct match median {np.median(ks[:, 0]):.0f} inliers")
        lines.append(f"contrast {c:<5} mAP@1 {m1:.3f} mAP@5 {m5:.3f}  museum keypoints median {np.median(d['n_keypoints_db']):.0f} "
                     f"min {np.min(d['n_keypoints_db'])}  3 weakest correct matches {weak}  "
                     f"{float(d['seconds_per_query']):.2f} s/painting  " + "  ".join(
                         f"{k} mAP {v[0]:.3f}/{v[1]:.3f}, unknown F1 at {t_c}: {v[2]:.3f}" for k, v in val.items()))
    best_c = max(R["contrast"], key=lambda c: (R["contrast"][c]["map1"], R["contrast"][c]["map5"], c))
    R["best_contrast"] = best_c
    tag = CONTRASTS[best_c]
    lines.append(f"-> chosen contrast threshold: {best_c}")

    # 2. Ratio test and RANSAC for the chosen feature.
    d = load(args.scores_dir, best_feature, "qsd1_w4", tag)
    rows = painting_rows(gts["qsd1_w4"], d["counts"])
    known = [(r, c) for r, c in rows if c != -1]
    lines += ["", f"== 2. Ratio test and RANSAC ({best_feature}, known paintings of qsd1_w4) =="]
    R["ratio"] = []
    for key in sorted(k for k in d if k.startswith("r")):
        ratio, mode = float(key[1:].split("_")[0]), key.split("_")[1]
        _, m1, m5, _ = evaluate(d[key], d["db_ids"], known)
        ks = known_scores(d[key], d["db_ids"], rows)
        unk = unknown_best(d[key], rows)
        margin_known = ks[:, 0] - ks[:, 1]
        R["ratio"].append(dict(ratio=ratio, ransac=mode == "ransac", map1=m1, map5=m5,
                               margin_median=float(np.median(margin_known)), margin_min=int(margin_known.min()),
                               correct_median=float(np.median(ks[:, 0])), wrong_median=float(np.median(ks[:, 1])),
                               unknown_max=int(unk.max()) if len(unk) else None))
        lines.append(f"ratio {ratio:<4} {mode:<6} mAP@1 {m1:.3f} mAP@5 {m5:.3f}  correct-best wrong: median "
                     f"{np.median(margin_known):.0f}, min {margin_known.min()}  correct median {np.median(ks[:, 0]):.0f}  "
                     f"best wrong median {np.median(ks[:, 1]):.0f}  unknowns max {unk.max() if len(unk) else '-'}")
    # Chosen setting: best mAP@1, then mAP@5, then the largest median margin relative to the wrong matches.
    best = max(R["ratio"], key=lambda x: (x["map1"], x["map5"], x["ransac"],
                                          x["margin_median"] / max(x["wrong_median"], 1)))
    key = f"r{best['ratio']}_{'ransac' if best['ransac'] else 'plain'}"
    R["best_setting"] = key
    lines.append(f"-> chosen setting: {key}")

    # 3. Unknown threshold on qsd1_w4 (all paintings), tested on qsd2_w3 and qsd1_w3 with simulated unknowns.
    S = d[key]
    ks = known_scores(S, d["db_ids"], rows)
    unk = unknown_best(S, rows)
    known_best = np.array([S[r].max() for r, c in known if r is not None])
    thresholds = np.arange(1, 101)
    f1 = f1_curve(unk, known_best, thresholds)
    best_f1 = f1.max()
    plateau = thresholds[f1 == best_f1]
    threshold = int(np.round(np.median(plateau)))
    predicted, m1, m5, f1_dev = evaluate(S, d["db_ids"], rows, threshold)
    _, m1_no, m5_no, _ = evaluate(S, d["db_ids"], rows, None)
    misses = [(i, c, p[:3]) for i, ((r, c), p) in enumerate(zip(rows, predicted)) if p[:1] != [c]]
    R["unknown"] = dict(threshold=threshold, plateau=[int(plateau.min()), int(plateau.max())], f1=float(f1_dev),
                        unknown_max=int(unk.max()), known_best_min=int(known_best.min()),
                        correct_min=int(ks[:, 0].min()), map1=m1, map5=m5, map1_without=m1_no, map5_without=m5_no,
                        n_paintings=len(rows), n_unknown=int(sum(c == -1 for _, c in rows)),
                        misses=[dict(painting=i, correct=c, predicted=p) for i, c, p in misses])
    lines += ["", f"== 3. Unknown threshold ({best_feature}, {key}) ==",
              f"qsd1_w4: best score of the unknowns <= {unk.max()}; best score of the knowns >= {known_best.min()}; "
              f"F1 = {best_f1:.3f} for thresholds {plateau.min()}-{plateau.max()} -> threshold {threshold}",
              f"qsd1_w4, all {len(rows)} paintings: mAP@1 {m1:.3f} mAP@5 {m5:.3f} (without the unknown test "
              f"{m1_no:.3f} / {m5_no:.3f}); unknown F1 {f1_dev:.3f}",
              "paintings not right at rank 1 (index, correct, predicted top 3): " + str(misses)]
    R["validation"] = {}
    for s in ("qsd2_w3", "qsd1_w3"):
        dv = load(args.scores_dir, best_feature, s, tag)
        if dv is None:
            continue
        rv = painting_rows(gts[s], dv["counts"])
        kv = known_scores(dv[key], dv["db_ids"], rv)
        known_best_v = np.array([dv[key][r].max() for r, c in rv if r is not None])
        sim_unknown = kv[:, 1]   # the correct image removed: the best score left is the best wrong one
        f1_v = f1_curve(sim_unknown, known_best_v, [threshold])[0]
        _, v1, v5, _ = evaluate(dv[key], dv["db_ids"], rv, threshold)
        R["validation"][s] = dict(f1=float(f1_v), map1=v1, map5=v5, n=len(rv), sim_unknown_max=int(sim_unknown.max()),
                                  known_best_min=int(known_best_v.min()), correct_min=int(kv[:, 0].min()),
                                  false_unknown=int(np.sum(known_best_v < threshold)),
                                  missed_unknown=int(np.sum(sim_unknown >= threshold)),
                                  f1_curve=f1_curve(sim_unknown, known_best_v, thresholds).tolist())
        lines.append(f"{s} (not used to choose): {len(rv)} known paintings + {len(sim_unknown)} simulated unknowns; "
                     f"knowns best >= {known_best_v.min()}, simulated unknowns best <= {sim_unknown.max()}; at threshold "
                     f"{threshold}: F1 {f1_v:.3f}, {np.sum(known_best_v < threshold)} known called unknown, "
                     f"{np.sum(sim_unknown >= threshold)} unknown missed; mAP@1 {v1:.3f} mAP@5 {v5:.3f}")

    # Figures.
    fig, ax = plt.subplots(figsize=(6, 3.4))
    names = list(R["features"])
    x = np.arange(len(names))
    ax.bar(x - 0.2, [R["features"][n]["correct_median"] for n in names], 0.4, color="#2f855a", label="correct image")
    ax.bar(x + 0.2, [R["features"][n]["wrong_max_median"] for n in names], 0.4, color="#c53030", label="best wrong image")
    for i, n in enumerate(names):
        f = R["features"][n]
        ax.text(i, f["correct_median"] * 1.15, f"mAP@1 {f['map1']:.2f}\n{f['seconds_per_query']:.1f} s",
                ha="center", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(1, 2000); ax.set_xticks(x, names)
    ax.set_ylabel("inliers (median, known paintings)"); ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(args.figures_dir, "features.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for ransac, style in ((False, "--"), (True, "-")):
        pts = sorted((x for x in R["ratio"] if x["ransac"] == ransac), key=lambda x: x["ratio"])
        ax.plot([x["ratio"] for x in pts], [x["correct_median"] for x in pts], style, color="#2f855a", marker="o",
                label=f"correct image{' + RANSAC' if ransac else ''}")
        ax.plot([x["ratio"] for x in pts], [x["wrong_median"] for x in pts], style, color="#c53030", marker="o",
                label=f"best wrong image{' + RANSAC' if ransac else ''}")
    ax.set_yscale("symlog", linthresh=10); ax.set_xlabel("ratio threshold (d1 < ratio · d2)")
    ax.set_ylabel("matches kept (median)"); ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(args.figures_dir, "ratio.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, 200)
    groups = [("correct image\n(known)", ks[:, 0], "#2f855a"), ("best wrong image\n(known)", ks[:, 1], "#c53030"),
              ("best image\n(unknown)", unk, "#6b46c1")]
    for i, (label, v, c) in enumerate(groups):
        ax.scatter(i + jitter[:len(v)], np.maximum(v, 0.8), color=c, s=18, alpha=0.8)
    ax.axhline(threshold, color="black", lw=1, ls="--")
    ax.text(2.45, threshold, f"threshold {threshold}", va="bottom", ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_xticks(range(3), [g[0] for g in groups], fontsize=8)
    ax.set_ylabel("verified matches (inliers)"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(args.figures_dir, "separation.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(thresholds, f1, color="black", label="qsd1_w4 (chosen here)")
    for s, c in (("qsd2_w3", "#2b6cb0"), ("qsd1_w3", "#c05621")):
        if s in R["validation"]:
            ax.plot(thresholds, R["validation"][s]["f1_curve"], color=c, label=f"{s} (validation)")
    ax.axvspan(plateau.min(), plateau.max(), color="#e2e8f0"); ax.axvline(threshold, color="black", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("unknown threshold (inliers of the best match)"); ax.set_ylabel("F1 (unknown)")
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(args.figures_dir, "f1_threshold.png"), dpi=150); plt.close(fig)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.output_dir, "keypoint_results.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(f"\nSaved {os.path.join(args.output_dir, 'keypoint_results.txt')}, results.json and figures in {args.figures_dir}")


if __name__ == "__main__":
    main()
