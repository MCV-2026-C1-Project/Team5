"""Task 1: colour histogram descriptors (1D, joint 2D/3D, blocks and spatial pyramid)."""

import itertools
import os
import re

import cv2
import numpy as np

COLOR_SPACES = {
    "gray": (cv2.COLOR_BGR2GRAY, (256,)),
    "rgb": (cv2.COLOR_BGR2RGB, (256, 256, 256)),
    "hsv": (cv2.COLOR_BGR2HSV, (180, 256, 256)),
    "lab": (cv2.COLOR_BGR2LAB, (256, 256, 256)),
    "ycrcb": (cv2.COLOR_BGR2YCrCb, (256, 256, 256)),
}

def list_images(folder):
    """Return the .jpg paths of a folder in sorted filename order."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".jpg"))
    if not names:
        raise FileNotFoundError(f"No .jpg files in {folder}")
    return [os.path.join(folder, n) for n in names]

def image_id(path):
    """Integer ID from a filename: bbdd_00120.jpg -> 120, 00007.jpg -> 7."""
    name = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"No numeric ID in filename: {path}")
    return int(match.group(1))

def read_image(path):
    """Read an image as BGR uint8 (OpenCV loads BGR, not RGB)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img

# Colour spaces where channel 0 is brightness and channels 1, 2 are colour.
NORMALIZABLE = ("lab", "ycrcb")
# Target mean of every channel and target standard deviation of the brightness channel.
NORM_MEAN = 128.0
NORM_STD = 50.0

def to_float_space(img_bgr, color_space):
    """Lab or YCrCb image as float32 on the OpenCV 8-bit scale, without rounding to integers.

    In a very dark photo the colour channels vary by less than one unit, so the uint8
    conversion would round them to a single value before normalize_shift can stretch them.
    """
    img = cv2.cvtColor(img_bgr.astype(np.float32) / 255, COLOR_SPACES[color_space][0])
    if color_space == "lab":
        # Float Lab has L in [0, 100] and a, b centred on 0; move them to the 8-bit scale.
        return img * np.float32([255 / 100, 1, 1]) + np.float32([0, 128, 128])
    return img * 255

def normalize_shift(img):
    """Lab or YCrCb image (float32) with the global brightness and colour change removed.

    Every channel is moved to a fixed mean, and all three are stretched by the factor that
    brings the brightness (channel 0) to a fixed standard deviation. Less light shrinks
    brightness and colour differences by the same amount, so the colour channels are
    stretched by the same factor. The shape of each histogram is kept, but a query that is
    darker, lighter or tinted maps back close to the original painting.
    """
    # meanStdDev works in float64; a float32 mean over millions of pixels is not accurate enough.
    mean, std = (v.ravel() for v in cv2.meanStdDev(img))
    scale = np.full(3, NORM_STD / max(std[0], 1e-6))
    img = (img.astype(np.float32) - mean.astype(np.float32)) * scale.astype(np.float32) + NORM_MEAN
    # calcHist ignores values outside the range, so clip them into the first and last bin.
    return np.clip(img, 0, 255.99)

def channel_groups(n_channels, joint):
    """Which channels share one histogram: 1 = each alone, 2 = every pair, 3 = all three together."""
    if joint not in (1, 2, 3) or joint > n_channels:
        raise ValueError(f"joint={joint} needs a colour space with at least {joint} channels")
    return [(0, 1, 2)] if joint == 3 else list(itertools.combinations(range(n_channels), joint))

def block_histograms(img, uppers, bins, grid, joint):
    """Histograms of every block of a grid x grid split, concatenated; the result sums to 1."""
    hists = []
    for rows in np.array_split(img, grid, axis=0):
        for block in np.array_split(rows, grid, axis=1):
            block = np.ascontiguousarray(block)
            for channels in channel_groups(len(uppers), joint):
                ranges = [edge for c in channels for edge in (0, uppers[c])]
                hist = cv2.calcHist([block], list(channels), None, [bins] * len(channels), ranges)
                hist = hist.ravel().astype(np.float64)
                hists.append(hist / hist.sum())
    # Divide by the number of histograms so the whole descriptor sums to 1.
    return np.concatenate(hists) / len(hists)

def convert_image(img_bgr, color_space, normalize=False):
    """BGR image in the chosen colour space (optionally normalised), shape (H, W, channels)."""
    if color_space not in COLOR_SPACES:
        raise ValueError(f"Unknown colour space '{color_space}', choose from {list(COLOR_SPACES)}")
    if normalize:
        if color_space not in NORMALIZABLE:
            raise ValueError(f"normalize is only defined for {NORMALIZABLE}, not '{color_space}'")
        img = normalize_shift(to_float_space(img_bgr, color_space))
    else:
        img = cv2.cvtColor(img_bgr, COLOR_SPACES[color_space][0])
    return img[:, :, None] if img.ndim == 2 else img

def histogram_descriptor(img, color_space, bins, grid=1, joint=1, level_weights=None, cache=None):
    """Colour histogram descriptor (sums to 1) of an image already converted with convert_image.

    grid: an int G splits the image into G x G blocks (1 = the whole image); a tuple such
        as (1, 2, 4) is a spatial pyramid: the descriptors of those levels, concatenated.
    joint: 1 = one 1D histogram per channel, 2 = one 2D histogram per pair of channels,
        3 = a single 3D histogram of the three channels together.
    level_weights: weight of each pyramid level (default: all levels count the same).
    cache: optional dict shared between calls on the same image, so a level that appears in
        several settings (for example 2x2 blocks and a pyramid) is computed only once.
    """
    uppers = COLOR_SPACES[color_space][1]
    levels = (int(grid),) if np.ndim(grid) == 0 else tuple(int(g) for g in grid)
    weights = np.ones(len(levels)) if level_weights is None else np.asarray(level_weights, dtype=np.float64)
    if len(weights) != len(levels) or min(levels) < 1:
        raise ValueError("grid sizes must be >= 1 and level_weights needs one weight per level")
    weights = weights / weights.sum()
    if cache is None:
        cache = {}
    for g in levels:
        if (bins, joint, g) not in cache:
            cache[(bins, joint, g)] = block_histograms(img, uppers, bins, g, joint)
    return np.concatenate([w * cache[(bins, joint, g)] for w, g in zip(weights, levels)])

def compute_histogram(img_bgr, color_space, bins, normalize=False, grid=1, joint=1, level_weights=None):
    """Colour histogram descriptor of a BGR image (see histogram_descriptor for the options)."""
    return histogram_descriptor(convert_image(img_bgr, color_space, normalize), color_space, bins, grid, joint, level_weights)

def compute_descriptors(folder, color_space, bins, normalize=False, grid=1, joint=1, level_weights=None):
    """Descriptors (N, D) and integer IDs (N) of all .jpg images in a folder, in sorted order."""
    paths = list_images(folder)
    descriptors = np.stack([compute_histogram(read_image(p), color_space, bins, normalize, grid, joint, level_weights) for p in paths])
    return descriptors, [image_id(p) for p in paths]