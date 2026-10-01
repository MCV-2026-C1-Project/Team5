"""Task 2: similarity measures between a query q (D,) and a database db (N, D)."""

import numpy as np

def euclidean(q, db):
    """Euclidean distance."""
    return np.sqrt(np.sum((q - db) ** 2, axis=1))

def l1(q, db):
    """L1 (city block) distance."""
    return np.sum(np.abs(q - db), axis=1)

def chi2(q, db):
    """Chi-squared distance."""
    # eps avoids 0/0 in bins that are empty in both histograms.
    return np.sum((q - db) ** 2 / (q + db + 1e-10), axis=1)

def intersection(q, db):
    """Histogram intersection (similarity, 1 for identical histograms)."""
    return np.sum(np.minimum(q, db), axis=1)

def hellinger(q, db):
    """Hellinger kernel (similarity, 1 for identical histograms)."""
    return np.sum(np.sqrt(q * db), axis=1)

# name -> (function, is_similarity)
MEASURES = {
    "euclidean": (euclidean, False),
    "l1": (l1, False),
    "chi2": (chi2, False),
    "intersection": (intersection, True),
    "hellinger": (hellinger, True),
}

def to_distance(q, db, measure):
    """Scores of q against every row of db where lower always means more similar."""
    if measure not in MEASURES:
        raise ValueError(f"Unknown measure '{measure}', choose from {list(MEASURES)}")
    function, is_similarity = MEASURES[measure]
    scores = function(q, db)
    # Similarities are negated so that retrieval can always sort ascending.
    return -scores if is_similarity else scores
