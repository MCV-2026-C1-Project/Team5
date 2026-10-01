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

# EMD accounts for how far apart bins are, so a darker or lighter image costs little.
def emd(q, db):
    """1D Earth Mover's distance (L1 between cumulative histograms)."""
    # Channels sum to the same value in q and db, so one cumsum equals per-channel cumsums.
    return np.sum(np.abs(np.cumsum(q - db, axis=1)), axis=1)

# Jensen-Shannon uses logs, so it is more sensitive to differences in rare bins.
def jensen_shannon(q, db):
    """Jensen-Shannon divergence (symmetric, bounded)."""
    m = (q + db) / 2
    kl_q = np.sum(q * np.log((q + 1e-10) / (m + 1e-10)), axis=1)
    kl_db = np.sum(db * np.log((db + 1e-10) / (m + 1e-10)), axis=1)
    return (kl_q + kl_db) / 2

# name -> (function, is_similarity)
MEASURES = {
    "euclidean": (euclidean, False),
    "l1": (l1, False),
    "chi2": (chi2, False),
    "intersection": (intersection, True),
    "hellinger": (hellinger, True),
    "emd": (emd, False),
    "js": (jensen_shannon, False),
}

def to_distance(q, db, measure):
    """Scores of q against every row of db where lower always means more similar."""
    if measure not in MEASURES:
        raise ValueError(f"Unknown measure '{measure}', choose from {list(MEASURES)}")
    function, is_similarity = MEASURES[measure]
    scores = function(q, db)
    # Similarities are negated so that retrieval can always sort ascending.
    return -scores if is_similarity else scores