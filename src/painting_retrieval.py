"""Week 3, Task 4: retrieval of every painting of a photo (denoise, segment, crop, describe, rank).

Each painting is cropped to the bounding box of its mask, and a margin is then cut from every side of
the crop and of the museum images alike: the edge of a crop is where the mask errors are (a strip of
wall, a piece of frame), and the block descriptors (DCT, wavelet) shift when it moves. Every painting is
described with several descriptors; their distances to the museum are put on the same scale (shifted
to 0 and divided by their median over the museum) and averaged.
"""

import numpy as np

from src.denoise import denoise
from src.descriptors import compute_histogram, convert_image, histogram_descriptor, image_id, list_images, read_image
from src.distances import to_distance
from src.watershed import watershed_masks
from src.masks import segment_paintings
from src.texture import DESCRIPTORS, prepare

MARGIN = 0.10   # fraction cut from every side of the crops and of the museum images
# name -> (texture descriptor of src/texture.py, distance), with the distance that was best in Task 2
TEXTURES = {
    "multiscale LBP": ("multiscale LBP r=1,2,3, 4x4 blocks", "l1"),
    "DCT": ("DCT 8x8 blocks, 10 coefficients", "cosine"),
    "wavelet": ("Haar wavelet 3 levels, 4x4 blocks", "l1"),
    "LBP": ("LBP r=1, 4x4 blocks", "l1"),
}
COLOR = ("ycrcb", 32, (1, 2, 4), "chi2")   # Week 2 colour descriptor, normalised, on the masked pixels only
FUSION = ("multiscale LBP", "DCT", "wavelet", "colour")   # descriptors averaged in the final method


def cut_margin(a, margin):
    """The image (or mask) without a strip of margin times its size on every side."""
    h, w = a.shape[:2]
    dy, dx = int(margin * h), int(margin * w)
    return a[dy:h - dy, dx:w - dx]


def crop_painting(img, mask, margin=MARGIN):
    """Crop of the image and of the mask to the bounding box of the mask, minus the margin."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        ys, xs = np.array([0, img.shape[0] - 1]), np.array([0, img.shape[1] - 1])
        mask = np.ones(img.shape[:2], bool)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return cut_margin(img[box], margin), cut_margin(mask[box], margin)


def describe_database(bbdd_dir, names=FUSION, margin=MARGIN):
    """Descriptors of the museum images, {name: (N, D)}, and their integer IDs."""
    paths = list_images(bbdd_dir)
    images = [cut_margin(read_image(p), margin) for p in paths]
    db = {}
    grays = [prepare(img) for img in images] if any(n != "colour" for n in names) else []
    for name in names:
        if name == "colour":
            color_space, bins, grid, _ = COLOR
            db[name] = np.stack([compute_histogram(img, color_space, bins, True, grid) for img in images])
        else:
            describe = DESCRIPTORS[TEXTURES[name][0]][0]
            db[name] = np.stack([describe(g) for g in grays])
    return db, [image_id(p) for p in paths]


def painting_distances(crops, db):
    """Distance of every crop (image, mask) to every museum image, for each descriptor in db: {name: (n, N)}."""
    out = {}
    for name, db_desc in db.items():
        if name == "colour":
            color_space, bins, grid, measure = COLOR
            rows = []
            for img, mask in crops:
                m = mask.astype(np.uint8)
                rows.append(to_distance(histogram_descriptor(convert_image(img, color_space, normalize=True, mask=m),
                                                             color_space, bins, grid, mask=m), db_desc, measure))
        else:
            key, measure = TEXTURES[name]
            rows = [to_distance(DESCRIPTORS[key][0](prepare(img)), db_desc, measure) for img, _ in crops]
        out[name] = np.stack(rows)
    return out


def scaled(d):
    """Distances shifted to start at 0 and divided by their median over the museum (per painting), so
    different descriptors add up; similarities come negated from to_distance and can be negative."""
    d = d - d.min(axis=1, keepdims=True)
    return d / np.maximum(np.median(d, axis=1, keepdims=True), 1e-12)


def fuse(dist, names=FUSION):
    """Mean of the scaled distances of the given descriptors."""
    return sum(scaled(dist[n]) for n in names) / len(names)


def rankings(dist, db_ids, k=10):
    """IDs of the k nearest museum images for every row of a distance matrix, best first."""
    return [[int(db_ids[i]) for i in np.argsort(row, kind="stable")[:k]] for row in dist]


# How the paintings are found: the Otsu masks (default), or watershed seeded from them (src/watershed.py).
EXTRACTORS = {"otsu": segment_paintings, "watershed": watershed_masks}


def find_paintings(img, segment=True, noise_filter="median 3x3", extractor="otsu"):
    """Denoised image and one mask per painting (the whole image if segment is False)."""
    if noise_filter != "none":
        img = denoise(img, noise_filter)
    masks = EXTRACTORS[extractor](img) if segment else []
    return img, masks or [np.ones(img.shape[:2], bool)]


def retrieve_paintings(img, db, db_ids, k=10, segment=True, names=FUSION, margin=MARGIN, extractor="otsu"):
    """Per painting of the photo, in reading order, the IDs of the k best museum images; and the masks."""
    img, masks = find_paintings(img, segment, extractor=extractor)
    dist = painting_distances([crop_painting(img, m, margin) for m in masks], db)
    return rankings(fuse(dist, names), db_ids, k), masks