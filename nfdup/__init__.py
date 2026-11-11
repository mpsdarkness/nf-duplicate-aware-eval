"""Duplicate-aware evaluation utilities for NetFlow intrusion-detection datasets."""
from .split import (  # noqa: F401
    EXCLUDE_COLS,
    SplitResult,
    feature_hashes,
    hash_group_split,
    load_features,
    overlap_audit,
    random_row_split,
    splitmix64,
)
