"""Task 3: mean Average Precision at K."""

def apk(actual, predicted, k):
    """Average precision at k for one query (actual: correct IDs, predicted: ranked IDs)."""
    if not actual:
        return 0.0
    hits = 0
    score = 0.0
    for rank, p in enumerate(predicted[:k], start=1):
        if p in actual and p not in predicted[: rank - 1]:
            hits += 1
            score += hits / rank
    return score / min(len(actual), k)

def mapk(actual, predicted, k):
    """Mean of apk over all queries."""
    if len(actual) != len(predicted):
        raise ValueError(f"{len(actual)} ground-truth entries but {len(predicted)} predictions")
    return sum(apk(a, p, k) for a, p in zip(actual, predicted)) / len(actual)
