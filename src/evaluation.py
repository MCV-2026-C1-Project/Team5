"""Evaluation: mean Average Precision at K (retrieval) and SSIM (image quality)."""

from typing import NamedTuple

import cv2
import numpy as np


class BootstrapResult(NamedTuple):
    mean: float
    ci_lower: float
    ci_upper: float
    std_err: float


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


def per_query_apk(actual, predicted, k):
    """Array of average precision at k for each query."""
    if len(actual) != len(predicted):
        raise ValueError(f"{len(actual)} ground-truth entries but {len(predicted)} predictions")
    return np.array([apk(a, p, k) for a, p in zip(actual, predicted)], dtype=np.float64)


def mapk(actual, predicted, k):
    """Mean of apk over all queries."""
    if len(actual) != len(predicted):
        raise ValueError(f"{len(actual)} ground-truth entries but {len(predicted)} predictions")
    return float(np.mean(per_query_apk(actual, predicted, k)))


def flatten_paintings(actual, predicted):
    """Several paintings per image -> one query per painting of the ground truth.

    actual: per image, the correct ID of each painting ([[a], [b, c], ...]); predicted: per image,
    one ranked list per painting found, in the same order. A painting that was not found gets an
    empty ranking (AP 0); extra paintings found are ignored.
    """
    flat_actual, flat_predicted = [], []
    for ids, rankings in zip(actual, predicted):
        for j, painting in enumerate(ids):
            flat_actual.append([painting])
            flat_predicted.append(rankings[j] if j < len(rankings) else [])
    return flat_actual, flat_predicted


def mapk_paintings(actual, predicted, k):
    """mAP@k over the paintings of the ground truth (see flatten_paintings)."""
    return mapk(*flatten_paintings(actual, predicted), k)


def bootstrap_mapk(actual, predicted, k, n_bootstraps=1000, ci=0.95, seed=None):
    """Non-parametric bootstrap confidence interval and standard error for mAP@k.
    
    Resamples queries with replacement to estimate sampling variability.
    """
    scores = per_query_apk(actual, predicted, k)
    n = len(scores)
    point_estimate = float(np.mean(scores))
    rng = np.random.default_rng(seed)

    # Resample queries with replacement: shape (n_bootstraps, n)
    indices = rng.integers(0, n, size=(n_bootstraps, n))
    bootstrap_means = np.mean(scores[indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(bootstrap_means, 100.0 * alpha))
    ci_upper = float(np.percentile(bootstrap_means, 100.0 * (1.0 - alpha)))
    std_err = float(np.std(bootstrap_means, ddof=1))

    return BootstrapResult(
        mean=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_err=std_err,
    )


def ssim(a, b):
    """Structural similarity of two images of the same size (1 = identical), averaged over pixels and channels."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def local_mean(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = local_mean(a), local_mean(b)
    var_a = local_mean(a * a) - mu_a ** 2
    var_b = local_mean(b * b) - mu_b ** 2
    cov = local_mean(a * b) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(ssim_map.mean())
