"""Task 1: choose the thresholds of the noise detector on clean images only, then test it.

Calibration uses the museum images (BBDD), which are clean, so no query annotation is used:
- detect_diff: the smallest candidate at which 99% of the clean images have fewer than 1 impulse
  per 10,000 pixels, so an "impulse" is something that almost never happens in a clean painting;
- noisy_ratio: twice the largest impulse ratio found in any clean image.

The detector is then tested on the query sets, which were not used to choose anything, against
their annotation (augmentations.pkl). The chosen values are the constants of src/denoise.py.
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.denoise import isolation
from src.descriptors import list_images, read_image

try:
    from tqdm import tqdm
except ImportError:   # pip install tqdm for a progress bar
    def tqdm(iterable, **kwargs):
        return iterable

CANDIDATES = (30, 40, 50, 60, 70, 80)   # detect_diff values tried
CLEAN_SHARE = 99                         # percentile of clean images that must stay below MAX_CLEAN_RATIO
MAX_CLEAN_RATIO = 1e-4                   # 1 impulse per 10,000 pixels
SAFETY = 2                               # noisy_ratio = SAFETY x the largest ratio of a clean image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="folder containing BBDD and the query sets")
    parser.add_argument("--test-sets", nargs="+", default=["qsd1_w3", "qsd2_w3"], help="query sets with augmentations.pkl")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "week3"))
    args = parser.parse_args()

    # Calibration on the clean museum images: the isolation map is computed once per image.
    clean = [isolation(read_image(p)) for p in tqdm(list_images(os.path.join(args.data, "BBDD")), desc="BBDD", unit="img")]
    lines = [f"Calibration on {len(clean)} clean museum images (BBDD)",
             f"{'detect_diff':>11} {'p' + str(CLEAN_SHARE) + ' ratio':>10} {'max ratio':>10}"]
    detect_diff = None
    for t in CANDIDATES:
        ratios = np.array([np.mean(iso > t) for iso in clean])
        lines.append(f"{t:>11} {np.percentile(ratios, CLEAN_SHARE):>10.5f} {ratios.max():>10.5f}")
        if detect_diff is None and np.percentile(ratios, CLEAN_SHARE) < MAX_CLEAN_RATIO:
            detect_diff, noisy_ratio = t, SAFETY * ratios.max()
    if detect_diff is None:
        raise SystemExit("No candidate keeps the clean images below MAX_CLEAN_RATIO")
    lines += [f"Chosen: detect_diff = {detect_diff}, noisy_ratio = {noisy_ratio:.5f} "
              f"({SAFETY} x the largest ratio of a clean image)", ""]

    # Test on the query sets (held out).
    for query_set in args.test_sets:
        query_dir = os.path.join(args.data, query_set)
        with open(os.path.join(query_dir, "augmentations.pkl"), "rb") as f:
            noisy = np.array(["Noise" in str(a) for a in pickle.load(f)])
        ratios = np.array([np.mean(isolation(read_image(p)) > detect_diff) for p in list_images(query_dir)])
        detected = ratios > noisy_ratio
        lines.append(f"{query_set}: {np.sum(detected == noisy)}/{len(noisy)} correct, "
                     f"{np.sum(detected & ~noisy)} false alarms, {np.sum(~detected & noisy)} missed; "
                     f"impulse ratio of the {noisy.sum()} noisy images {ratios[noisy].min():.4f}-{ratios[noisy].max():.4f}, "
                     f"of the others {ratios[~noisy].min():.4f}-{ratios[~noisy].max():.4f}")

    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "noise_calibration.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
