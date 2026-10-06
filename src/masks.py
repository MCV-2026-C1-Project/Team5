"""Tasks 3-4: background removal using only colour, and mask evaluation (precision, recall, F1)."""

import cv2
import numpy as np

WORK_SIZE = 400   # longest side used for segmentation; the mask is resized back afterwards
MIN_STD = 3.0     # floor for the background spread (Lab units), so a very flat wall does not give huge distances


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
    lab = to_lab(img_bgr)
    pix = border_pixels(lab, strip_width(lab, border))
    centre = np.median(pix, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(pix - centre), axis=0), MIN_STD)
    pix = pix[np.all(np.abs(pix - centre) < 3 * spread, axis=1)]   # drop border pixels far from the wall colour
    inv_cov = np.linalg.inv(np.cov(pix.T) + np.eye(3) * MIN_STD ** 2)
    diff = lab - pix.mean(axis=0)
    d = np.sqrt(np.maximum(np.einsum("hwi,ij,hwj->hw", diff, inv_cov, diff), 0))   # Mahalanobis distance
    return otsu_on_distance(d)


def kmeans_two(img_bgr, border):
    """K-means with 2 colour clusters in Lab; the background is the cluster that dominates the border."""
    lab = to_lab(img_bgr)
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(lab.reshape(-1, 3), 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return foreground_from_labels(labels.reshape(lab.shape[:2]) == 0, strip_width(lab, border))


METHODS = {"gray_otsu": gray_otsu, "border_otsu": border_otsu, "kmeans": kmeans_two}


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


def clean(mask, hull=False):
    """Close gaps, fill holes, remove specks, keep the largest region (optionally its convex hull)."""
    size = max(3, int(round(0.015 * min(mask.shape))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    m = fill_holes(m)
    m = largest_component(cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel))
    points = cv2.findNonZero(m)
    if hull and points is not None:
        m = cv2.fillConvexPoly(np.zeros_like(m), cv2.convexHull(points), 1)
    return m


def segment(img_bgr, method, border=0.04, hull=False):
    """Foreground mask (bool, same size as the image)."""
    h, w = img_bgr.shape[:2]
    scale = WORK_SIZE / max(h, w)
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    mask = clean(METHODS[method](small, border), hull)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


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