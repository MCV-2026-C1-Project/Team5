"""Week 3, Task 1: noise filtering.

Only some queries are noisy, and the noise is impulse noise: isolated pixels replaced by random
values. An impulse differs from all of its 8 neighbours, while a pixel on an edge or in a texture
is always close to some of them. That is used to decide whether an image is noisy, so clean
images are not filtered.

The two thresholds are calibrated on the museum images, which are clean, with
week3/calibrate_noise_detector.py; the query annotations are only used to test them.
"""

import cv2
import numpy as np

DETECT_DIFF = 70        # a pixel this far (any channel) from all its 8 neighbours is an impulse
NOISY_RATIO = 0.00018   # an image with more impulses than this fraction of its pixels is noisy


def isolation(img):
    """For every pixel of a BGR image, its smallest absolute difference to its 8 neighbours (max over channels)."""
    img = img.astype(np.int16)
    h, w = img.shape[:2]
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    smallest = None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                diff = np.abs(img - padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])
                smallest = diff if smallest is None else np.minimum(smallest, diff)
    return smallest.max(axis=2)


def impulse_ratio(img):
    """Fraction of pixels that look like impulses."""
    return float(np.mean(isolation(img) > DETECT_DIFF))


def is_noisy(img):
    """True if the image has more impulses than a clean image would."""
    return impulse_ratio(img) > NOISY_RATIO


# filters (each takes and returns a BGR uint8 image)

def gaussian(img, size=3):
    """Linear baseline: weighted mean of the neighbourhood. Spreads impulses instead of removing them."""
    return cv2.GaussianBlur(img, (size, size), 0)


def median(img, size=3):
    """Median of the neighbourhood, per channel: an isolated outlier never reaches the median."""
    return cv2.medianBlur(img, size)


def bilateral(img, size=9, sigma_color=75, sigma_space=75):
    """Mean of the neighbours that have a similar colour: smooths flat areas, keeps edges."""
    return cv2.bilateralFilter(img, size, sigma_color, sigma_space)


def nl_means(img, h=10):
    """Non-local means: averages pixels whose surrounding patches look alike, anywhere in a window."""
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


FILTERS = {
    "gaussian 3x3": gaussian,
    "median 3x3": median,
    "median 5x5": lambda img: median(img, 5),
    "bilateral": bilateral,
    "non-local means": nl_means,
}


def denoise(img, method="median 3x3"):
    """Filter an image with one of FILTERS, only if it is detected as noisy."""
    return FILTERS[method](img) if is_noisy(img) else img
