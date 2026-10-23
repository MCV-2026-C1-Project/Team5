"""Tasks 3-4: background removal using only colour, and mask evaluation (precision, recall, F1)."""

import cv2
import numpy as np

WORK_SIZE = 400   # longest side used for segmentation; the mask is resized back afterwards
MIN_STD = 3.0     # floor for the background spread (Lab units), so a very flat wall does not give huge distances
WALL_LIMIT = 4.0  # Mahalanobis distance above which a pixel is never wall: 99.9% of a 3D Gaussian is closer


def to_lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def strip_width(img, border):
    """Width in pixels of the border strips used to learn what the background looks like."""
    return max(2, int(round(border * min(img.shape[:2]))))


def border_pixels(img, k):
    """Pixels of the four strips of width k along the image border, shape (n, channels)."""
    img = img.reshape(img.shape[0], img.shape[1], -1)
    strips = [img[:k], img[-k:], img[:, :k], img[:, -k:]]
    return np.concatenate([s.reshape(-1, img.shape[2]) for s in strips])


def foreground_from_labels(class_a, k):
    """Two-class map -> foreground: the class that is NOT the majority on the border."""
    return ~class_a if border_pixels(class_a, k).mean() > 0.5 else class_a


def otsu_on_distance(d):
    """Otsu threshold on a distance-to-background map (large distance = foreground)."""
    d8 = np.clip(d / max(np.percentile(d, 99), 1e-6) * 255, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary > 0


# methods (each returns a raw bool mask)

def gray_otsu(img_bgr, border):
    """Baseline: Otsu on grayscale (throws the colour away)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return foreground_from_labels(binary > 0, strip_width(img_bgr, border))


def border_otsu(img_bgr, border):
    """One Gaussian colour model of the background from the border strips, then Otsu on the distance."""
    return otsu_on_distance(wall_distance(img_bgr, border))


def border_otsu_log(img_bgr, border):
    """border_otsu with Otsu on log(1 + distance).

    The log compresses the large distances, so the threshold falls in the valley between the wall and
    everything that is not wall. Without it, with two paintings the threshold separates the wall from
    the most different painting, and a dark or greyish second painting can fall below it.

    Otsu only splits two groups; when there are three (wall, a dark part, a bright red part) it can
    still cut between the two painting groups. So a pixel further than WALL_LIMIT is always painting.
    """
    d = wall_distance(img_bgr, border)
    return otsu_on_distance(np.log1p(d)) | (d > WALL_LIMIT)


def wall_distance(img_bgr, border):
    """Mahalanobis distance in Lab of every pixel to the wall colour learnt from the border strips."""
    lab = to_lab(img_bgr)
    pix = border_pixels(lab, strip_width(lab, border))
    centre = np.median(pix, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(pix - centre), axis=0), MIN_STD)
    pix = pix[np.all(np.abs(pix - centre) < 3 * spread, axis=1)]   # drop border pixels far from the wall colour
    inv_cov = np.linalg.inv(np.cov(pix.T) + np.eye(3) * MIN_STD ** 2)
    diff = lab - pix.mean(axis=0)
    return np.sqrt(np.maximum(np.einsum("hwi,ij,hwj->hw", diff, inv_cov, diff), 0))


def kmeans_two(img_bgr, border):
    """K-means with 2 colour clusters in Lab; the background is the cluster that dominates the border."""
    lab = to_lab(img_bgr)
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(lab.reshape(-1, 3), 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return foreground_from_labels(labels.reshape(lab.shape[:2]) == 0, strip_width(lab, border))


METHODS = {"gray_otsu": gray_otsu, "border_otsu": border_otsu, "border_otsu_log": border_otsu_log, "kmeans": kmeans_two}


# clean-up shared by all methods

def fill_holes(m):
    """Everything that cannot be reached from outside the mask becomes foreground."""
    padded = np.pad(m, 1)
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 2)
    return (padded | (flood == 0))[1:-1, 1:-1].astype(np.uint8)


def largest_component(m):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    return (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def large_components(m, min_region):
    """Every region with at least min_region times the area of the largest one.

    Keeps paintings made of several panels (separated by strips of wall or by a frame of
    the same colour as the wall), which largest_component would cut down to one panel.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    return np.isin(labels, 1 + np.flatnonzero(areas >= min_region * areas.max())).astype(np.uint8)


def clean(mask, hull=False, min_region=None):
    """Close gaps, fill holes, remove specks, keep the largest region (optionally its convex hull).

    With min_region (a fraction, e.g. 0.05) every region at least that fraction of the largest
    one is kept instead of only the largest; the convex hull then covers all of them.
    """
    m = morphology(mask)
    m = large_components(m, min_region) if min_region else largest_component(m)
    return convex_hull(m) if hull else m


def morphology(mask):
    """Close small gaps, fill holes, remove specks (elliptic element of 1.5% of the shorter side)."""
    size = max(3, int(round(0.015 * min(mask.shape))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    m = fill_holes(m)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)


def convex_hull(m):
    """The filled convex hull of the mask pixels (the mask itself if it is empty)."""
    points = cv2.findNonZero(m)
    return m if points is None else cv2.fillConvexPoly(np.zeros_like(m), cv2.convexHull(points), 1)


def segment(img_bgr, method, border=0.04, hull=False, min_region=None):
    """Foreground mask (bool, same size as the image). min_region: see clean (None = largest region only)."""
    h, w = img_bgr.shape[:2]
    scale = WORK_SIZE / max(h, w)
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    mask = clean(METHODS[method](small, border), hull, min_region)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


# Week 3, Task 3: up to two paintings per image

def group_regions(m, min_region=0.05, min_area=0.01, gap=0.03):
    """Regions of a cleaned mask grouped into paintings: list of label sets, plus the label image.

    Regions smaller than min_area of the image or than min_region of the largest region are dropped
    (labels on the wall, specks). Regions whose bounding boxes are closer than gap (fraction of the
    longer side) belong to the same painting: the panels of one work, or a painting split by a frame
    of the wall colour, are a few pixels apart, while two paintings hang with wall between them.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return [], labels
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero((areas >= min_region * areas.max()) & (areas >= min_area * m.size))
    pad = gap * max(m.shape)
    groups = []   # [labels, x0, y0, x1, y1]
    for i in keep:
        x, y, w, h = stats[i, :4]
        groups.append([{i}, x, y, x + w, y + h])
    merged = True
    while merged:   # merge any two groups whose boxes are closer than the gap, until none are
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                A, B = groups[a], groups[b]
                if A[1] - pad <= B[3] and B[1] - pad <= A[3] and A[2] - pad <= B[4] and B[2] - pad <= A[4]:
                    groups[a] = [A[0] | B[0], min(A[1], B[1]), min(A[2], B[2]), max(A[3], B[3]), max(A[4], B[4])]
                    del groups[b]
                    merged = True
                    break
            if merged:
                break
    return groups, labels


def reading_order(boxes):
    """Indices of the boxes (x0, y0, x1, y1) from left to right; top to bottom if they share columns."""
    def before(a, b):
        overlap = min(a[2], b[2]) - max(a[0], b[0])
        if overlap > 0.5 * min(a[2] - a[0], b[2] - b[0]):   # one above the other
            return a[1] < b[1]
        return a[0] < b[0]
    order = list(range(len(boxes)))
    for i in range(len(order)):   # insertion sort with the pairwise rule (at most a few boxes)
        j = i
        while j > 0 and before(boxes[order[j]], boxes[order[j - 1]]):
            order[j], order[j - 1] = order[j - 1], order[j]
            j -= 1
    return order


def segment_paintings(img_bgr, method="border_otsu_log", border=0.04, hull=True, max_paintings=2,
                      min_region=0.05, min_area=0.01, gap=0.03, bridge=0.05):
    """One bool mask per painting (at most max_paintings, the largest ones), in reading order.

    Same colour segmentation and clean-up as segment. Then an opening with an element of bridge times
    the shorter side cuts thin bridges (a rail or a line on the wall that joins two paintings), the
    regions are grouped into paintings (group_regions) and, with hull, each painting is replaced by
    its own convex hull, so the wall between two paintings is never filled.
    """
    h, w = img_bgr.shape[:2]
    scale = WORK_SIZE / max(h, w)
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    m = morphology(METHODS[method](small, border))
    if bridge:
        size = max(3, int(round(bridge * min(m.shape))) | 1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    groups, labels = group_regions(m, min_region, min_area, gap)
    masks = [np.isin(labels, list(g[0])).astype(np.uint8) for g in groups]
    best = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))[:max_paintings]
    masks, boxes = [masks[i] for i in best], [groups[i][1:] for i in best]
    masks = [masks[i] for i in reading_order(boxes)]
    if hull:
        masks = [convex_hull(m) for m in masks]
    return [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]


# Task 4: evaluation

def mask_scores(pred, gt):
    """Precision, recall and F1 of a predicted mask against the annotation (both bool arrays)."""
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1