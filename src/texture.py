"""Week 3, Task 2: texture descriptors (LBP, multiscale LBP, DCT, Haar wavelet).

All of them work on the grey-level image resized to SIZE x SIZE, so a photo of a painting and the
museum image are described at the same scale. They ignore colour, so a change of hue does not
change them. Like the colour histograms, the image is split into grid x grid blocks and the
descriptors of the blocks are concatenated.
"""

import cv2
import numpy as np

SIZE = 256   # side of the square grey image every descriptor is computed on


def prepare(img_bgr):
    """Grey-level image resized to SIZE x SIZE, as float32."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)


def blocks(img, grid):
    """The grid x grid non-overlapping blocks of an image, row by row."""
    return [block for rows in np.array_split(img, grid, axis=0) for block in np.array_split(rows, grid, axis=1)]


# LBP

def lbp_codes(gray, radius=1, points=8):
    """Local binary pattern of every pixel.

    Each of the `points` neighbours on a circle of `radius` gives one bit: 1 if it is at least as
    bright as the centre, 0 otherwise. Neighbours that do not fall on a pixel are bilinearly
    interpolated. With 8 points the code is a number from 0 to 255.
    """
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    codes = np.zeros((h, w), np.int32)
    for k in range(points):
        angle = 2 * np.pi * k / points
        map_x = (xs + radius * np.cos(angle)).astype(np.float32)
        map_y = (ys - radius * np.sin(angle)).astype(np.float32)
        neighbour = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        codes |= (neighbour >= gray).astype(np.int32) << k
    return codes


def lbp_descriptor(gray, grid=4, radii=(1,), points=8):
    """Histogram of LBP codes per block, for one or several radii (multiscale LBP); sums to 1."""
    hists = []
    for radius in radii:
        for block in blocks(lbp_codes(gray, radius, points), grid):
            hist = np.bincount(block.ravel(), minlength=2 ** points).astype(np.float64)
            hists.append(hist / hist.sum())
    return np.concatenate(hists) / len(hists)


# DCT

def zigzag(n):
    """(row, column) of the n x n DCT coefficients in zigzag order, from low to high frequency."""
    return sorted(((i, j) for i in range(n) for j in range(n)),
                  key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 else p[1]))


def dct_descriptor(gray, grid=8, coefficients=10):
    """The first DCT coefficients (zigzag order, lowest frequencies) of every block; unit length.

    The DC coefficient (the mean brightness of the block) is left out, so the descriptor describes
    how the brightness varies inside the block, not how bright it is.
    """
    parts = blocks(gray, grid)
    order = zigzag(min(parts[0].shape))[1:coefficients + 1]   # all blocks have the same size
    features = []
    for block in parts:
        coeffs = cv2.dct(block)
        features.append([coeffs[i, j] for i, j in order])
    features = np.ravel(features)
    return features / max(np.linalg.norm(features), 1e-12)


# Haar wavelet

def haar_step(img):
    """One level of the 2D Haar transform: approximation and horizontal, vertical, diagonal details."""
    a, b = img[0::2, 0::2], img[0::2, 1::2]
    c, d = img[1::2, 0::2], img[1::2, 1::2]
    return (a + b + c + d) / 4, (a + b - c - d) / 4, (a - b + c - d) / 4, (a - b - c + d) / 4


def wavelet_descriptor(gray, grid=4, levels=3):
    """Energy (mean absolute value) of each detail sub-band, per level and block; sums to 1.

    Each level halves the resolution, so the levels describe the texture at finer and coarser scales.
    """
    energies, approx = [], gray
    for _ in range(levels):
        approx, *details = haar_step(approx)
        energies += [np.abs(block).mean() for band in details for block in blocks(band, grid)]
    energies = np.array(energies)
    return energies / max(energies.sum(), 1e-12)


# name -> (function of the prepared grey image, distances to try)
HISTOGRAM_MEASURES = ["l1", "chi2", "hellinger"]
DESCRIPTORS = {
    "LBP r=1, 4x4 blocks": (lambda g: lbp_descriptor(g, grid=4), HISTOGRAM_MEASURES),
    "multiscale LBP r=1,2,3, 4x4 blocks": (lambda g: lbp_descriptor(g, grid=4, radii=(1, 2, 3)), HISTOGRAM_MEASURES),
    "DCT 8x8 blocks, 10 coefficients": (lambda g: dct_descriptor(g, grid=8, coefficients=10), ["euclidean", "cosine", "l1"]),
    "Haar wavelet 3 levels, 4x4 blocks": (lambda g: wavelet_descriptor(g, grid=4, levels=3), HISTOGRAM_MEASURES),
}
