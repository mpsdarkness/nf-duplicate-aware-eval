"""Check that nfdup reproduces the split code that the experiments ran.

The functions are read from the as-run notebook, executed on synthetic data
and compared with the standalone module. No dataset is needed.
"""
import ast
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

import nfdup
from nfdup.split import N_CANDIDATE_SALTS

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_step4_full_data_group_disjoint.ipynb"
WANTED = {
    "splitmix64",
    "evaluate_group_assignment",
    "smoke_limit_indices",
    "create_exact_feature_group_split",
    "create_attack_stratified_random_split",
}


def notebook_functions(tmp_path):
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    source = "\n\n".join(
        "".join(c["source"]) for c in cells if c["cell_type"] == "code"
    )
    source = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))
    )
    tree = ast.parse(source)
    chosen = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    assert {n.name for n in chosen} == WANTED
    namespace = {
        "np": np, "pd": pd, "gc": gc, "Path": Path,
        "Dict": Dict, "Any": Any, "Tuple": Tuple, "DatasetBundle": object,
        "TRAIN_FRAC": 0.70, "VAL_FRAC": 0.10,
        "TARGET_SPLIT_FRACTIONS": np.asarray([0.70, 0.10, 0.20], dtype=np.float64),
        "GROUP_CANDIDATE_SALTS": 24,
        "GROUP_EVALUATION_CHUNK_SIZE": 1_000_000,
        "MIN_FAMILY_TOTAL_FOR_BALANCE": 30,
        "SMOKE_MAX_FLOWS_PER_SPLIT": None,
        "GROUP_SPLIT_VERSION": "hash_group_splitmix64_candidate_search_v2",
    }
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def synthetic(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    prototypes = rng.integers(0, 6, (300, 39)).astype(np.float32)
    X = prototypes[rng.integers(0, 300, n)]
    unique = rng.random(n) < 0.35
    X[unique] += rng.random((int(unique.sum()), 39)).astype(np.float32)
    names = [f"F{i}" for i in range(39)]
    labels = (X[:, 0] + X[:, 1] > 5).astype(np.int8)
    family = np.where(labels == 0, "Benign", np.where(X[:, 2] > 3, "DoS", np.where(X[:, 3] > 4, "Rare", "Exploits")))
    codes, classes = pd.factorize(pd.Series(family), sort=False)
    codes = codes.astype(np.int16)
    order = np.argsort(codes, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(codes.astype(np.int64)))]
    bundle = SimpleNamespace(
        labels=labels, attack_codes=codes, attack_classes=np.asarray(classes, dtype=str),
        attack_group_indices=[order[offsets[i]:offsets[i + 1]].astype(np.int32) for i in range(len(classes))],
    )
    return X, names, labels, codes, bundle


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_group_split_matches_notebook(tmp_path, seed):
    original = notebook_functions(tmp_path)
    X, names, labels, codes, bundle = synthetic()
    hashes = nfdup.feature_hashes(X, names)
    unique, inverse, counts = np.unique(hashes, return_inverse=True, return_counts=True)

    train, val, test, meta = original["create_exact_feature_group_split"](
        bundle, unique, inverse.astype(np.int32), counts.astype(np.int32), seed, tmp_path
    )
    mine = nfdup.hash_group_split(hashes, codes, seed)

    assert mine.salt == meta["selected_salt"]
    assert mine.metadata["selected_objective"] == meta["selected_objective"]
    assert mine.metadata["full_assignment_split_rows"] == meta["full_assignment_split_rows"]
    np.testing.assert_array_equal(mine.train_idx, train)
    np.testing.assert_array_equal(mine.val_idx, val)
    np.testing.assert_array_equal(mine.test_idx, test)

    # a recorded salt reproduces the same partition without the search
    again = nfdup.hash_group_split(hashes, codes, seed, salt=mine.salt)
    np.testing.assert_array_equal(again.test_idx, test)


def test_random_row_split_matches_notebook(tmp_path):
    original = notebook_functions(tmp_path)
    _, _, _, codes, bundle = synthetic()
    expected = original["create_attack_stratified_random_split"](bundle, 42)
    mine = nfdup.random_row_split(codes, 42)
    for a, b in zip(mine, expected):
        np.testing.assert_array_equal(a, b)


def test_splitmix64_matches_notebook(tmp_path):
    original = notebook_functions(tmp_path)
    values = np.random.default_rng(1).integers(0, np.iinfo(np.uint64).max, 10_000, dtype=np.uint64)
    np.testing.assert_array_equal(nfdup.splitmix64(values), original["splitmix64"](values))


def test_group_disjoint_and_random_overlap():
    X, names, labels, codes, _ = synthetic()
    hashes = nfdup.feature_hashes(X, names)
    split = nfdup.hash_group_split(hashes, codes, 42)
    assert nfdup.overlap_audit(hashes, split.train_idx, split.test_idx)["test_rows_with_vector_in_train"] == 0
    assert nfdup.overlap_audit(hashes, split.train_idx, split.val_idx)["test_rows_with_vector_in_train"] == 0
    assert len(split.train_idx) + len(split.val_idx) + len(split.test_idx) == len(hashes)
    train, _, test = nfdup.random_row_split(codes, 42)
    assert nfdup.overlap_audit(hashes, train, test)["test_overlap_rate"] > 0.3


def test_recorded_salts_are_candidates_of_their_seed():
    recorded = json.loads((ROOT / "splits" / "paper_full_data_splits.json").read_text())["splits"]
    for entry in recorded:
        salts = np.random.default_rng(entry["seed"]).integers(
            1, np.iinfo(np.uint64).max, size=N_CANDIDATE_SALTS, dtype=np.uint64
        ).tolist()
        assert entry["selected_salt"] in salts, entry
        for part in ("train", "validation", "test"):
            rows, benign, attack = entry[part]
            assert rows == benign + attack


def test_load_features_normalizes_non_finite(tmp_path):
    frame = pd.DataFrame({
        "IPV4_SRC_ADDR": ["a", "b", "c"], "L4_SRC_PORT": [1, 2, 3],
        "IPV4_DST_ADDR": ["d", "e", "f"], "L4_DST_PORT": [4, 5, 6],
        "PROTOCOL": [6, 17, 6], "IN_BYTES": [1.0, np.inf, np.nan], "OUT_BYTES": ["7", "x", "9"],
        "Label": [0, 1, 1], "Attack": ["Benign", "DoS", "DoS"],
    })
    path = tmp_path / "toy.csv"
    frame.to_csv(path, index=False)
    X, labels, codes, names, features = nfdup.load_features(str(path))
    assert features == ["PROTOCOL", "IN_BYTES", "OUT_BYTES"]
    assert X.dtype == np.float32 and np.isfinite(X).all()
    np.testing.assert_array_equal(X[:, 1], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(X[:, 2], [7.0, 0.0, 9.0])
    assert names.tolist() == ["Benign", "DoS"] and codes.tolist() == [0, 1, 1]
