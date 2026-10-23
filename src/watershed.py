"""Week 3, Task 3: paintings found by watershed, seeded from the Otsu masks.

segment_paintings (src/masks.py) decides how many paintings there are (up to two) and roughly where. The core of
each of its masks (shrunk) is sure painting, thin strips along the image border are sure wall, and watershed grows
all of them at once over the image: wall and painting meet where the image changes most (usually the outer edge of
the frame), not where a colour threshold falls. Each painting is then its filled, convex region.
"""

import cv2
import numpy as np

from src.masks import convex_hull, fill_holes, largest_component, reading_order, segment_paintings

SIZE = 600       # longest side of the image used for the watershed
BORDER = 0.03    # width of the border strips that are sure wall (fraction of the shorter side)
CORE = 0.15      # each Otsu mask is shrunk by this fraction of its box's shorter side to get a sure core


def odd(n):
    return max(3, int(n) | 1)


def bounding_box(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(ys) else None


def watershed_masks(img_bgr):
    """One bool mask per painting (up to two), in reading order, like segment_paintings."""
    scale = SIZE / max(img_bgr.shape[:2])
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    h, w = small.shape[:2]
    markers = np.zeros((h, w), np.int32)   # 0 unknown, 1 wall, 2 and 3 the paintings
    k = max(2, int(BORDER * min(h, w)))
    markers[:k], markers[-k:], markers[:, :k], markers[:, -k:] = 1, 1, 1, 1
    seeds = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for m in segment_paintings(img_bgr, hull=False)]
    labels = []
    for i, seed in enumerate(seeds):
        box = bounding_box(seed)
        if box is None:
            continue
        size = odd(CORE * min(box[2] - box[0], box[3] - box[1]))
        core = cv2.erode(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        if core.any():
            markers[core > 0] = i + 2
            labels.append(i + 2)
    if not labels:
        return []
    cv2.watershed(np.ascontiguousarray(cv2.GaussianBlur(small, (3, 3), 0)), markers)
    masks = [convex_hull(largest_component(fill_holes((markers == label).astype(np.uint8)))) for label in labels]
    masks = [m for m in masks if m.any()]
    masks = [masks[i] for i in reading_order([bounding_box(m) for m in masks])]
    return [cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]