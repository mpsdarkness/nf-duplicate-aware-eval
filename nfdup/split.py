"""Hash-group-disjoint splitting (Algorithm 1 of the paper).

This module is a standalone copy of the split logic that the experiment
notebooks ran. It has no dependency on the notebooks, PyTorch or Google Drive.
``tests/test_against_notebook.py`` checks that it returns the same assignment,
salt, objective and row order as the functions inside
``notebooks/01_step4_full_data_group_disjoint.ipynb``.

Typical use::

    from nfdup import load_features, feature_hashes, hash_group_split, overlap_audit

    X, labels, family_codes, family_names, feature_names = load_features("NF-UNSW-NB15-v2.csv")
    hashes = feature_hashes(X, feature_names)
    split = hash_group_split(hashes, family_codes, seed=42)
    print(split.salt, overlap_audit(hashes, split.train_idx, split.test_idx))
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Columns that are not model inputs. The 39 remaining NF-v2 columns, in release
# order, form the feature vector that defines a group.
EXCLUDE_COLS = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "Label",
    "Attack",
]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
TARGET_SPLIT_FRACTIONS = np.asarray([0.70, 0.10, 0.20], dtype=np.float64)
N_CANDIDATE_SALTS = 24
MIN_FAMILY_TOTAL_FOR_BALANCE = 30
SHUFFLE_SEED_OFFSET = 17_171
HASH_CHUNK_SIZE = 250_000


def load_features(
    csv_path: str,
    exclude_cols: Sequence[str] = EXCLUDE_COLS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Read an NF CSV and return the pre-scaling float32 model-input matrix.

    Non-numeric entries, NaN, +inf and -inf become 0.0, column by column, after
    the cast to float32. Returns ``(X, labels, family_codes, family_names,
    feature_names)``; family codes follow first appearance in the file, and
    benign traffic is one of the families.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    feature_names = [c for c in df.columns if c not in exclude_cols]
    X = np.empty((len(df), len(feature_names)), dtype=np.float32)
    for j, col in enumerate(feature_names):
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        X[:, j] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    labels = pd.to_numeric(df["Label"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    family_codes, family_names = pd.factorize(
        df["Attack"].fillna("<MISSING_ATTACK>").astype(str), sort=False
    )
    return (
        X,
        labels,
        family_codes.astype(np.int16, copy=False),
        np.asarray(family_names, dtype=str),
        feature_names,
    )


def feature_hashes(X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """uint64 key per row from ``pandas.util.hash_pandas_object(index=False)``.

    Rows with equal keys form one group. Signed zeros are not canonicalized.
    """
    hashes = np.empty(len(X), dtype=np.uint64)
    for start in range(0, len(X), HASH_CHUNK_SIZE):
        end = min(start + HASH_CHUNK_SIZE, len(X))
        frame = pd.DataFrame(X[start:end], columns=list(feature_names), copy=False)
        hashes[start:end] = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    return hashes


def splitmix64(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x


def assign_groups(unique_hashes: np.ndarray, salt: int) -> np.ndarray:
    """0 = train, 1 = validation, 2 = test, as a function of key and salt only."""
    mixed = splitmix64(np.asarray(unique_hashes) ^ np.uint64(salt))
    unit = mixed.astype(np.float64) / float(2**64)
    return np.where(
        unit < TRAIN_FRAC, 0, np.where(unit < TRAIN_FRAC + VAL_FRAC, 1, 2)
    ).astype(np.int8)


def assignment_objective(
    group_assignment: np.ndarray,
    group_inverse: np.ndarray,
    group_counts: np.ndarray,
    family_codes: np.ndarray,
    n_families: int,
) -> Dict[str, Any]:
    """Line 7 of Algorithm 1: J = 5 MAE_row + wMAE + 0.25 max + 10 #empty cells."""
    counts = np.asarray(group_counts, dtype=np.int64)
    split_rows = np.bincount(
        group_assignment.astype(np.int64), weights=counts, minlength=3
    ).astype(np.int64)

    split_code = group_assignment[np.asarray(group_inverse)].astype(np.int64)
    combined = split_code * n_families + np.asarray(family_codes).astype(np.int64)
    split_family = np.bincount(combined, minlength=3 * n_families).reshape(3, n_families)

    row_fractions = split_rows / max(split_rows.sum(), 1)
    global_error = float(np.abs(row_fractions - TARGET_SPLIT_FRACTIONS).mean())

    totals = split_family.sum(axis=0)
    eligible = totals >= MIN_FAMILY_TOTAL_FOR_BALANCE
    family_fraction = np.divide(
        split_family,
        totals[None, :],
        out=np.zeros_like(split_family, dtype=np.float64),
        where=totals[None, :] > 0,
    )
    family_error = np.abs(family_fraction - TARGET_SPLIT_FRACTIONS[:, None])
    if eligible.any():
        weights = np.sqrt(totals[eligible].astype(np.float64))
        weights /= weights.sum()
        family_mae = float((family_error[:, eligible].mean(axis=0) * weights).sum())
        max_family_error = float(family_error[:, eligible].max())
    else:
        family_mae = 0.0
        max_family_error = 0.0

    expected = TARGET_SPLIT_FRACTIONS[:, None] * totals[None, :]
    missing_required = int(((split_family == 0) & (expected >= 5)).sum())

    objective = 5.0 * global_error + family_mae + 0.25 * max_family_error + 10.0 * missing_required
    return {
        "objective": float(objective),
        "global_fraction_mae": global_error,
        "family_fraction_weighted_mae": family_mae,
        "max_family_fraction_error": max_family_error,
        "missing_required_family_cells": missing_required,
        "split_rows": split_rows.tolist(),
        "split_row_fractions": row_fractions.tolist(),
    }


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    salt: int
    metadata: Dict[str, Any]
    candidates: List[Dict[str, Any]] = field(default_factory=list)


def hash_group_split(
    hashes: np.ndarray,
    family_codes: np.ndarray,
    seed: int,
    n_candidates: int = N_CANDIDATE_SALTS,
    salt: Optional[int] = None,
) -> SplitResult:
    """Assign complete groups of identical vectors to train, validation or test.

    ``family_codes`` holds one integer per row (benign included as a family).
    With ``salt=None`` the candidate search of Algorithm 1 runs; passing a
    recorded salt reproduces a stored split without the search. Rows inside each
    split are shuffled with ``default_rng(seed + 17171)``.
    """
    unique_hashes, inverse, counts = np.unique(
        np.asarray(hashes), return_inverse=True, return_counts=True
    )
    n_families = int(np.asarray(family_codes).max()) + 1

    if salt is None:
        salts = np.random.default_rng(seed).integers(
            1, np.iinfo(np.uint64).max, size=n_candidates, dtype=np.uint64
        )
    else:
        salts = np.asarray([salt], dtype=np.uint64)

    best, best_assignment, best_salt, candidates = None, None, None, []
    for index, candidate in enumerate(salts):
        assignment = assign_groups(unique_hashes, candidate)
        metrics = assignment_objective(assignment, inverse, counts, family_codes, n_families)
        candidates.append({"candidate": index, "salt": int(candidate), **metrics})
        if best is None or metrics["objective"] < best["objective"]:  # first minimum wins
            best, best_assignment, best_salt = metrics, assignment, int(candidate)

    row_assignment = best_assignment[inverse]
    shuffle_rng = np.random.default_rng(seed + SHUFFLE_SEED_OFFSET)
    train_idx = shuffle_rng.permutation(np.flatnonzero(row_assignment == 0))
    val_idx = shuffle_rng.permutation(np.flatnonzero(row_assignment == 1))
    test_idx = shuffle_rng.permutation(np.flatnonzero(row_assignment == 2))

    metadata = {
        "seed": int(seed),
        "n_feature_groups": int(len(unique_hashes)),
        "candidate_salts_tested": int(len(salts)),
        "selected_salt": best_salt,
        "selected_objective": best["objective"],
        "selected_global_fraction_mae": best["global_fraction_mae"],
        "selected_family_fraction_weighted_mae": best["family_fraction_weighted_mae"],
        "selected_max_family_fraction_error": best["max_family_fraction_error"],
        "selected_missing_required_family_cells": best["missing_required_family_cells"],
        "full_assignment_split_rows": best["split_rows"],
        "full_assignment_split_row_fractions": best["split_row_fractions"],
    }
    return SplitResult(train_idx, val_idx, test_idx, best_salt, metadata, candidates)


def random_row_split(
    family_codes: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random-row baseline: stratified by family, 70/10/20, one generator per seed."""
    codes = np.asarray(family_codes)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes.astype(np.int64))
    offsets = np.r_[0, np.cumsum(counts)]
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []
    for i in range(len(counts)):
        perm = rng.permutation(order[offsets[i]:offsets[i + 1]].astype(np.int32))
        cut_train = int(len(perm) * TRAIN_FRAC)
        cut_val = int(len(perm) * (TRAIN_FRAC + VAL_FRAC))
        train_parts.append(perm[:cut_train])
        val_parts.append(perm[cut_train:cut_val])
        test_parts.append(perm[cut_val:])
    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    test_idx = rng.permutation(np.concatenate(test_parts))
    return train_idx, val_idx, test_idx


def overlap_audit(hashes: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    """Share of test rows whose feature vector also occurs in training."""
    hashes = np.asarray(hashes)
    train_keys = np.unique(hashes[train_idx])
    seen = np.isin(hashes[test_idx], train_keys)
    shared = np.intersect1d(train_keys, np.unique(hashes[test_idx])).size
    return {
        "test_rows": int(len(test_idx)),
        "test_rows_with_vector_in_train": int(seen.sum()),
        "test_overlap_rate": float(seen.mean()) if len(test_idx) else 0.0,
        "groups_shared_train_test": int(shared),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash-group-disjoint split of an NF CSV file.")
    parser.add_argument("csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--salt", type=int, default=None, help="reuse a recorded salt, skip the search")
    parser.add_argument("--out", default=None, help="write train/val/test row indices to this .npz")
    args = parser.parse_args()

    X, labels, family_codes, family_names, feature_names = load_features(args.csv)
    hashes = feature_hashes(X, feature_names)
    n_groups = len(np.unique(hashes))
    print(
        f"rows {len(hashes):,} | features {len(feature_names)} | groups {n_groups:,} | "
        f"repeated-row rate {100 * (1 - n_groups / len(hashes)):.2f}%"
    )

    split = hash_group_split(hashes, family_codes, args.seed, salt=args.salt)
    random_train, _, random_test = random_row_split(family_codes, args.seed)
    report = {
        **split.metadata,
        "group_disjoint_overlap": overlap_audit(hashes, split.train_idx, split.test_idx),
        "random_row_overlap": overlap_audit(hashes, random_train, random_test),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        np.savez_compressed(
            args.out,
            train_idx=split.train_idx,
            val_idx=split.val_idx,
            test_idx=split.test_idx,
            salt=np.uint64(split.salt),
        )


if __name__ == "__main__":
    main()
