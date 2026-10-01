"""Task 3: rank the database for each query."""

import numpy as np

from src.distances import to_distance

def retrieve(query_desc, db_desc, db_ids, measure, k):
    """For each query row, the IDs of the k most similar database images, best first."""
    results = []
    for q in query_desc:
        scores = to_distance(q, db_desc, measure)
        # Stable sort: ties keep the sorted-filename order, so results are reproducible.
        order = np.argsort(scores, kind="stable")[:k]
        results.append([int(db_ids[i]) for i in order])
    return results
