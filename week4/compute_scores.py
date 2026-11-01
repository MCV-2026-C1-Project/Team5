"""Tasks 1-2: score every query painting against every museum image, for every setting at once.

For each local feature (SIFT, RootSIFT, ORB, AKAZE) and query set, the paintings of every image are found as in
Week 3 (noise filtered, watershed extraction; the whole image for cropped sets), each painting is described, its
two nearest museum descriptors are found once per museum image, and the number of verified matches is counted for
every ratio threshold, with and without RANSAC. The result is one score matrix (paintings x museum) per setting,
saved to outputs/week4/scores/, which week4/analyze_keypoints.py turns into tables and figures.
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
from src.painting_retrieval import crop_painting, find_paintings

try:
    from tqdm import tqdm
except ImportError:   # pip install tqdm for a progress bar
    def tqdm(iterable, **kwargs):
        return iterable

RATIOS = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0)   # 1.0 = no ratio test
# (ratio, RANSAC?) settings scored; RANSAC on all 1,000 unfiltered matches of every pair is too slow and is left out
SETTINGS = [(r, ransac) for r in RATIOS for ransac in (False, True) if not (ransac and r == 1.0)]
# name -> (folder, find paintings?)
QUERY_SETS = {
    "qsd1_w4": (os.path.join("data", "qsd1_w4", "qsd1_w4"), True),
    "qsd2_w3": (os.path.join("data", "qsd2_w3", "qsd2_w3"), True),
    "qsd1_w3": (os.path.join("data", "qsd1_w3", "qsd1_w3"), False),
}

_DB = None   # (points, descriptors) of the museum, per worker


def _init(db, contrast):
    global _DB
    _DB = db
    K.CONTRAST = contrast   # workers re-import the module, so the setting is passed again


def _score_painting(task):
    """Score matrix rows of one painting: {(ratio, ransac): (n_museum,)}, plus its number of keypoints."""
    gray, method = task
    pts, desc = K.describe(gray, method)
    out = {setting: np.zeros(len(_DB), np.int32) for setting in SETTINGS}
    for j, (pts_db, desc_db) in enumerate(_DB):
        match = K.knn_matches(desc, desc_db, method)
        for r, ransac in SETTINGS:
            out[(r, ransac)][j] = K.score(match, pts, pts_db, ratio=r, ransac=ransac)
    return out, len(pts)


def query_paintings(folder, segment, noise_filter, max_side):
    """Grey crop of every painting found, and how many paintings each image has."""
    crops, counts = [], []
    for p in tqdm(list_images(folder), desc=f"  paintings {os.path.basename(folder)}", unit="img"):
        img, masks = find_paintings(read_image(p), segment, noise_filter, extractor="watershed")
        crops += [K.prepare(crop_painting(img, m, margin=0)[0], max_side) for m in masks]
        counts.append(len(masks))
    return crops, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--methods", nargs="+", default=list(K.FEATURES), choices=list(K.FEATURES))
    parser.add_argument("--sets", nargs="+", default=list(QUERY_SETS), choices=list(QUERY_SETS))
    parser.add_argument("--filter", default="median 3x3", help="noise filter for the noisy queries ('none' to skip)")
    parser.add_argument("--contrast", type=float, default=K.CONTRAST, help="SIFT contrast threshold")
    parser.add_argument("--max-side", type=int, default=K.MAX_SIDE, help="longer side of every image")
    parser.add_argument("--tag", default="", help="extra suffix of the output files (e.g. '_nofilter')")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week4", "scores"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    db_paths = list_images(os.path.join(args.data, "BBDD"))
    db_ids = [image_id(p) for p in db_paths]
    K.CONTRAST = args.contrast
    db_grays = [K.prepare(read_image(p), args.max_side) for p in tqdm(db_paths, desc="BBDD", unit="img")]
    paintings = {s: query_paintings(QUERY_SETS[s][0], QUERY_SETS[s][1], args.filter, args.max_side) for s in args.sets}

    for method in args.methods:
        t0 = time.time()
        db = [K.describe(g, method) for g in db_grays]
        t_db = (time.time() - t0) / len(db_grays)
        with Pool(args.workers, initializer=_init, initargs=(db, args.contrast)) as pool:
            for s in args.sets:
                crops, counts = paintings[s]
                t0 = time.time()
                rows = list(tqdm(pool.imap(_score_painting, [(c, method) for c in crops]), total=len(crops),
                                 desc=f"{method} {s}", unit="painting"))
                t_query = (time.time() - t0) / len(crops)
                scores = {f"r{r}_{'ransac' if ransac else 'plain'}": np.stack([row[(r, ransac)] for row, _ in rows])
                          for r, ransac in SETTINGS}
                suffix = f"_c{args.contrast}" if "SIFT" in method else ""   # the contrast threshold only affects SIFT
                np.savez_compressed(os.path.join(args.output_dir, f"{method}_{s}{suffix}{args.tag}.npz"), **scores,
                                    counts=np.array(counts), db_ids=np.array(db_ids),
                                    n_keypoints=np.array([n for _, n in rows]),
                                    n_keypoints_db=np.array([len(p) for p, _ in db]),
                                    seconds_per_query=t_query, seconds_per_db_image=t_db)
                print(f"{method} {s}: {len(crops)} paintings, {t_query:.2f} s per painting", flush=True)


if __name__ == "__main__":
    main()
