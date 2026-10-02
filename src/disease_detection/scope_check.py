"""Reject images that do not look like leaves from the training data.

The classifier always picks one of its 38 classes and its softmax confidence is
often ~100% on unrelated pictures (faces, scenes, noise), so confidence cannot
tell a leaf from a non-leaf. This check compares the image's ResNet50 features
(the 2048-value global-average-pooling output) with a reference set of real
training leaves:

    score = mean cosine similarity to the K most similar reference leaves

Real leaves score high; images unlike anything in the training data score low.
The reference set and threshold are built by scripts/build_leaf_reference.py.

Limitation: this detects images that differ from the training photos. A leaf of
an unsupported plant can still look similar enough to pass.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from src.preprocessing import config

REFERENCE_PATH = config.ROOT / "models" / "resnet50" / "leaf_reference.npz"
FEATURE_LAYER = "gap"
DEFAULT_K = 5
# Chosen in scripts/build_leaf_reference.py: lowest real-leaf score on validation
# was 0.641 and highest out-of-scope score was 0.530.
DEFAULT_SCOPE_THRESHOLD = 0.60


def _normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def knn_similarity(features, reference, k=DEFAULT_K):
    """Mean cosine similarity of each feature vector to its k nearest references."""
    sims = _normalize(features) @ reference.T
    k = min(k, reference.shape[0])
    return np.sort(sims, axis=1)[:, -k:].mean(axis=1)


class LeafScopeChecker:
    """Scores feature vectors against the saved reference leaves."""

    def __init__(self, reference, threshold=DEFAULT_SCOPE_THRESHOLD, k=DEFAULT_K):
        self.reference = _normalize(reference)
        self.threshold = float(threshold)
        self.k = int(k)

    @classmethod
    def from_file(cls, path=REFERENCE_PATH, threshold=None):
        data = _load_reference(Path(path))
        return cls(data["features"],
                   threshold if threshold is not None else float(data["threshold"]),
                   int(data["k"]))

    def scores(self, features):
        return knn_similarity(features, self.reference, self.k)

    def is_in_scope(self, score):
        return score >= self.threshold


@lru_cache(maxsize=2)
def _load_reference(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_scope_checker(path=REFERENCE_PATH):
    """The checker, or None when the reference file has not been built."""
    path = Path(path)
    return LeafScopeChecker.from_file(path) if path.is_file() else None
