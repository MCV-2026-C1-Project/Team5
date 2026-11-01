"""Week 4, Tasks 1-2: keypoints, local descriptors, matching and verification.

Every image (a painting crop or a museum image) is converted to grey and resized so its longer side is MAX_SIDE:
the descriptors are computed on gradients of the intensity, so the hue changes of the queries do not reach them, and
a common size keeps the number of keypoints of a photo and of its museum image comparable.

Matching a query painting against a museum image:
1. tentative matches: for every query descriptor, its two nearest museum descriptors (L2 for SIFT, Hamming for the
   binary ORB and AKAZE descriptors);
2. ratio test: keep a match only if the nearest is clearly closer than the second nearest (d1 < RATIO * d2), since
   a descriptor that matches two places equally well is not distinctive;
3. geometric verification: a painting is planar, so true matches are related by one homography; RANSAC finds it and
   the score of the pair is its number of inliers.
The museum is ranked by score; if even the best score is below the unknown threshold, the painting is unknown (-1).
"""

import cv2
import numpy as np

MAX_SIDE = 512     # longer side of every image before detecting keypoints
N_FEATURES = 1000  # keypoints kept per image (the strongest)
CONTRAST = 0.02    # SIFT: DoG extrema with less contrast are rejected (OpenCV's default 0.04 leaves flat paintings bare)
RATIO = 0.75       # Lowe's ratio test
RANSAC_PX = 5.0    # RANSAC reprojection threshold, in pixels of the MAX_SIDE image
MIN_MATCHES = 4    # a homography needs 4 point pairs
UNKNOWN_THRESHOLD = 44   # fewer inliers in the best match -> unknown painting (week4/analyze_keypoints.py)


def _sift():
    return cv2.SIFT_create(nfeatures=N_FEATURES, contrastThreshold=CONTRAST)


def _orb():
    return cv2.ORB_create(nfeatures=N_FEATURES)


def _akaze():
    return cv2.AKAZE_create()


# name -> (detector/descriptor factory, distance for matching, transform of the descriptors)
FEATURES = {
    "SIFT": (_sift, cv2.NORM_L2, None),
    "RootSIFT": (_sift, cv2.NORM_L2, "root"),   # L1-normalise + sqrt: L2 between them is the Hellinger distance
    "ORB": (_orb, cv2.NORM_HAMMING, None),
    "AKAZE": (_akaze, cv2.NORM_HAMMING, None),
}


def prepare(img_bgr, max_side=MAX_SIDE):
    """Grey image with its longer side resized to max_side."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    scale = max_side / max(gray.shape)
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def describe(gray, method="SIFT"):
    """Keypoint coordinates (N, 2) and descriptors (N, D) of a grey image."""
    factory, _, transform = FEATURES[method]
    detector = factory()
    keypoints, desc = detector.detectAndCompute(gray, None)
    if method == "AKAZE" and len(keypoints) > N_FEATURES:   # AKAZE has no nfeatures: keep the strongest
        order = np.argsort([-k.response for k in keypoints])[:N_FEATURES]
        keypoints, desc = [keypoints[i] for i in order], desc[order]
    if desc is None or len(keypoints) == 0:
        return np.zeros((0, 2), np.float32), None
    if transform == "root":
        desc = np.sqrt(desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-7)).astype(np.float32)
    return np.float32([k.pt for k in keypoints]), desc


def knn_matches(desc_q, desc_db, method="SIFT"):
    """For every query descriptor, its nearest and second nearest museum descriptor.

    Returns (query index, museum index, d1, d2) arrays; empty if either side has fewer than 2 descriptors.
    """
    if desc_q is None or desc_db is None or len(desc_q) < 2 or len(desc_db) < 2:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0)
    matcher = cv2.BFMatcher(FEATURES[method][1])
    pairs = [p for p in matcher.knnMatch(desc_q, desc_db, k=2) if len(p) == 2]
    q = np.array([p[0].queryIdx for p in pairs], int)
    t = np.array([p[0].trainIdx for p in pairs], int)
    d1 = np.array([p[0].distance for p in pairs], float)
    d2 = np.array([p[1].distance for p in pairs], float)
    return q, t, d1, d2


def verified_matches(match, pts_q, pts_db, ratio=RATIO, ransac=True, ransac_px=RANSAC_PX):
    """Indices (into the knn arrays) of the matches that pass the ratio test and, if ransac, the homography check."""
    q, t, d1, d2 = match
    good = np.flatnonzero(d1 < ratio * d2)
    if not ransac or len(good) < MIN_MATCHES:
        return good if not ransac else good[:0]
    _, inliers = cv2.findHomography(pts_q[q[good]], pts_db[t[good]], cv2.RANSAC, ransac_px)
    return good[inliers.ravel() > 0] if inliers is not None else good[:0]


def score(match, pts_q, pts_db, **kwargs):
    """Score of a query-museum pair: number of verified matches."""
    return len(verified_matches(match, pts_q, pts_db, **kwargs))


def rank(scores, db_ids, k=10, unknown_threshold=None):
    """IDs of the k best museum images (highest score first); [-1] if the best score is below the threshold."""
    if unknown_threshold is not None and scores.max(initial=0) < unknown_threshold:
        return [-1]
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(db_ids[i]) for i in order]
