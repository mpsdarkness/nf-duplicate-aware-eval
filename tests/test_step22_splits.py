"""Check that nfdup reproduces the split functions of notebook 12 (experiment A).

The shared code cell of ``notebooks/12_step22_experimentA_rr_vs_gd.ipynb``
carries its own copy of the hash-group-disjoint split (``hash_group_split``)
and of the random-row baseline (``random_row_split``). This test reads those
functions from the notebook, runs them on synthetic data and requires
``nfdup`` to return the same per-row part, the same salt and the same row
order. No dataset is needed.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nfdup
from nfdup.split import N_CANDIDATE_SALTS

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "12_step22_experimentA_rr_vs_gd.ipynb"
WANTED = {
    "splitmix64",
    "assign_groups",
    "assignment_objective",
    "hash_group_split",
    "split_order",
    "random_row_split",
}


def notebook_functions():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    source = "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    source = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))
    )
    tree = ast.parse(source)
    chosen = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    assert {n.name for n in chosen} == WANTED
    namespace = {
        "np": np, "pd": pd,
        "TRAIN_FRAC": 0.70, "VAL_FRAC": 0.10,
        "TARGET_SPLIT_FRACTIONS": np.asarray([0.70, 0.10, 0.20], dtype=np.float64),
        "N_CANDIDATE_SALTS": 24,
        "MIN_FAMILY_TOTAL_FOR_BALANCE": 30,
        "SHUFFLE_SEED_OFFSET": 17_171,
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
    codes, _ = pd.factorize(pd.Series(family), sort=False)
    return X, names, codes.astype(np.int16)


def part_from_indices(train_idx, val_idx, test_idx, n):
    part = np.full(n, -1, dtype=np.int8)
    part[train_idx] = 0
    part[val_idx] = 1
    part[test_idx] = 2
    assert (part >= 0).all()
    return part


@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize("n_candidates", [24, 8])
def test_hash_group_split_matches_notebook(seed, n_candidates):
    original = notebook_functions()
    X, names, codes = synthetic()
    hashes = nfdup.feature_hashes(X, names)

    part, salt, meta, candidates = original["hash_group_split"](hashes, codes, seed, n_candidates=n_candidates)
    mine = nfdup.hash_group_split(hashes, codes, seed, n_candidates=n_candidates)

    assert mine.salt == salt
    assert int(meta["selected_salt"]) == mine.salt
    assert meta["candidate_salts_tested"] == len(mine.candidates) == n_candidates
    assert meta["selected_objective"] == mine.metadata["selected_objective"]
    np.testing.assert_array_equal(
        part_from_indices(mine.train_idx, mine.val_idx, mine.test_idx, len(hashes)), part
    )
    # the notebook's split_order reproduces nfdup's shuffled row order
    train, val, test = original["split_order"](part, seed)
    np.testing.assert_array_equal(mine.train_idx, train)
    np.testing.assert_array_equal(mine.val_idx, val)
    np.testing.assert_array_equal(mine.test_idx, test)


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_hash_group_split_with_recorded_salt_matches_notebook(seed):
    original = notebook_functions()
    X, names, codes = synthetic()
    hashes = nfdup.feature_hashes(X, names)
    mine = nfdup.hash_group_split(hashes, codes, seed)
    part, salt, meta, _ = original["hash_group_split"](hashes, codes, seed, salt=mine.salt)
    assert salt == mine.salt and meta["candidate_salts_tested"] == 1
    np.testing.assert_array_equal(
        part_from_indices(mine.train_idx, mine.val_idx, mine.test_idx, len(hashes)), part
    )


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_random_row_split_matches_notebook(seed):
    original = notebook_functions()
    _, _, codes = synthetic()
    part = original["random_row_split"](codes, seed)
    train, val, test = nfdup.random_row_split(codes, seed)
    np.testing.assert_array_equal(part_from_indices(train, val, test, len(codes)), part)


def test_recorded_step22_salts_are_candidates_of_their_seed():
    """Every new frozen-cohort split of experiment A chose one of the 24 candidates of its seed,
    and the full-cohort splits reuse the salts of splits/paper_full_data_splits.json."""
    paper = {
        (e["dataset"], e["seed"]): e["selected_salt"]
        for e in json.loads((ROOT / "splits" / "paper_full_data_splits.json").read_text())["splits"]
    }
    split_dir = ROOT / "results" / "12_step22_experimentA_rr_vs_gd" / "splits"
    checked = 0
    for path in sorted(split_dir.glob("*__hash_group_disjoint__split*.json")):
        meta = json.loads(path.read_text())
        salt = int(meta["selected_salt"])
        if meta["cohort"] == "full":
            assert meta["candidate_salts_tested"] == 1
            assert salt == paper[(meta["dataset"], meta["split_seed"])]
        else:
            assert meta["candidate_salts_tested"] == N_CANDIDATE_SALTS
            salts = np.random.default_rng(meta["split_seed"]).integers(
                1, np.iinfo(np.uint64).max, size=N_CANDIDATE_SALTS, dtype=np.uint64
            ).tolist()
            assert salt in salts, path.name
        checked += 1
    assert checked == 12
