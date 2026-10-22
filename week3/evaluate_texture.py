"""Task 2: retrieval on qsd1_w3 with texture descriptors only, with and without noise filtering.

Every texture descriptor of src/texture.py is evaluated with its distances on several versions of
the queries:
- as given (some of them noisy);
- filtered with each filter of src/denoise.py, only on the queries detected as noisy: the full
  Task 1 + 2 pipeline, so the filter is judged by what it does to retrieval, not only by PSNR;
- the clean originals (non_augmented/), the upper bound if the noise were removed perfectly.
Besides mAP@1 / mAP@5 on the 30 queries, it counts how many of the noisy queries are found at rank 1,
where the effect of the filter shows.
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.denoise import FILTERS, denoise
from src.descriptors import image_id, list_images, read_image
from src.evaluation import mapk
from src.retrieval import retrieve
from src.texture import DESCRIPTORS, prepare

try:
    from tqdm import tqdm
except ImportError:   # pip install tqdm for a progress bar
    def tqdm(iterable, **kwargs):
        return iterable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing BBDD and the query set")
    parser.add_argument("--query-set", default="qsd1_w3")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week3"))
    args = parser.parse_args()

    query_dir = os.path.join(args.data, args.query_set)
    paths = list_images(query_dir)
    images = [read_image(p) for p in paths]
    with open(os.path.join(query_dir, "augmentations.pkl"), "rb") as f:
        noisy = np.array(["Noise" in str(a) for a in pickle.load(f)])
    with open(os.path.join(query_dir, "gt_corresps.pkl"), "rb") as f:
        gt = pickle.load(f)
    versions = {"no filter": [prepare(img) for img in images]}
    versions.update({f"{name} if noisy": [prepare(denoise(img, name)) for img in images] for name in FILTERS})
    versions["clean originals"] = [prepare(read_image(os.path.join(query_dir, "non_augmented", os.path.basename(p)))) for p in paths]

    # The museum images are read and converted to grey once.
    db_paths = list_images(os.path.join(args.data, "BBDD"))
    db_ids = [image_id(p) for p in db_paths]
    db_grays = [prepare(read_image(p)) for p in tqdm(db_paths, desc="BBDD", unit="img")]

    lines = [f"{args.query_set}: {len(images)} queries, {noisy.sum()} noisy. 'noisy @1' = noisy queries found at rank 1.", "",
             f"{'descriptor':<36} {'measure':<10} {'queries':<26} {'mAP@1':>6} {'mAP@5':>6} {'noisy @1':>9}"]
    for name, (describe, measures) in DESCRIPTORS.items():
        db_desc = np.stack([describe(g) for g in db_grays])
        query_desc = {v: np.stack([describe(g) for g in grays]) for v, grays in versions.items()}
        for measure in measures:
            for v in versions:
                predicted = retrieve(query_desc[v], db_desc, db_ids, measure, k=5)
                found = sum(p[0] == g[0] for p, g, n in zip(predicted, gt, noisy) if n)
                lines.append(f"{name:<36} {measure:<10} {v:<26} {mapk(gt, predicted, 1):>6.3f} {mapk(gt, predicted, 5):>6.3f} "
                             f"{f'{found}/{noisy.sum()}':>9}")
            lines.append("")
        print(f"done {name}", flush=True)

    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "texture_results.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
