# Duplicate-aware evaluation of NetFlow intrusion detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24125935.svg)](https://doi.org/10.5281/zenodo.24125935)

Code for the paper *Duplicate-Aware Evaluation of NetFlow Intrusion Detection: A Hash-Group-Disjoint Protocol for the NF-v2 Datasets* (N. Q. Nguyen and V. H. Le).

The repository has two parts:

- `nfdup/`: a small standalone implementation of the hash-group-disjoint split (Algorithm 1) and the overlap audit. It needs only NumPy and pandas.
- `notebooks/`: the Google Colab notebooks that produced the results, kept as they were run, with their outputs.

## Using the protocol on a dataset

```bash
pip install numpy pandas
python -m nfdup NF-UNSW-NB15-v2.csv --seed 42 --out unsw_seed42.npz
```

This reads the CSV, builds the 39-column float32 feature vector, hashes every row, assigns complete groups of identical vectors to train, validation or test (70/10/20), and prints the selected salt, the realized row shares, and the train-test overlap of both the group-disjoint split and a random-row split of the same data.

From Python:

```python
from nfdup import load_features, feature_hashes, hash_group_split, overlap_audit

X, labels, family_codes, family_names, feature_names = load_features("NF-UNSW-NB15-v2.csv")
hashes = feature_hashes(X, feature_names)
split = hash_group_split(hashes, family_codes, seed=42)
print(split.salt, overlap_audit(hashes, split.train_idx, split.test_idx))
```

`hash_group_split(..., salt=<recorded salt>)` skips the candidate search and rebuilds a stored partition. The salts and row counts of the six full-data splits used in the paper are in `splits/paper_full_data_splits.json`.

`nfdup` loads the whole CSV into memory, as the notebooks do. NF-CSE-CIC-IDS2018-v2 (18.9 million rows) needs a high-memory machine; notebooks 10 and 11 contain a chunked reader that fits in less memory.

### How the feature vector is defined

- Excluded columns: the two IP addresses, the two ports, `Label`, `Attack`, and the flow timestamps where present. On NF-v2 this leaves 39 columns, kept in release order.
- Each column is cast to float32; non-numeric values, NaN, +inf and -inf are set to 0.0.
- The group key is `pandas.util.hash_pandas_object(frame, index=False)` on the float32 frame, before any scaling. Signed zeros are not canonicalized.

## Datasets

The datasets are not redistributed here. Download them from the University of Queensland "ML-Based NIDS Datasets" page and cite the dataset papers:

| File | Used in |
|---|---|
| `NF-UNSW-NB15-v2.csv`, `NF-CSE-CIC-IDS2018-v2.csv` | all experiments |
| `NF-UNSW-NB15-v3.csv`, `NF-CSE-CIC-IDS2018-v3.csv` | notebooks 08 and 09 |

## Notebooks and the results they produce

| Notebook | Content | Paper |
|---|---|---|
| `01_step4_full_data_group_disjoint` | Duplicate and label-conflict audit of both NF-v2 datasets; random-row overlap; Algorithm 1 with seeds 42, 123 and 456; MLP, No-G, AE, hybrid and Shuffled variants in the full-data setting | Sections V-A and V-B, Tables 3, 10 and 11 |
| `02_step5_frozen_benchmark` | Construction of the frozen attack-enriched subset and its split; MLP, No-G, hybrid and Shuffled variants on it | Table 1, Table 8 |
| `03_step6_graph_baselines` | Controlled E-GraphSAGE and GraphIDS-style baselines on the frozen benchmark | Tables 8 and 9 |
| `04_step7_lightgbm_out_of_fold_context` | LightGBM on the frozen benchmark; out-of-fold context scoring of the hybrid detector (Full-OOF, called Full-LOO in the notebook) | Tables 8 and 9 |
| `05_step8_audits_and_sensitivity` | Value-equality check of sampled groups, operating-point aggregation, gate values, score-weight sweep, group bootstrap, batch-partition sensitivity | Section III-A, Table 9, Sections V-G, V-I, V-J and V-K |
| `06_step9_context_target` | Retraining with the context reconstruction target detached or dropped | Section V-K |
| `07_step10_gradient_at_checkpoint` | Gradient decomposition on the gate at initialization and at trained checkpoints | Section V-K |
| `08_step19_v2_v3_audit_and_split_simulation` | Repetition audit of the v2 and v3 releases; host-pair-disjoint and global chronological split simulation | Tables 4 and 5 |
| `09_step19b_temporal_split_variants` | Chronological split variants on v3 (per day, per session, per day and family) | Table 5 |
| `10_step20_seen_unseen_partition` | LightGBM under the random-row split, scored on seen and unseen test vectors; lookup reference | Tables 6 and 7 |
| `11_step21_lightgbm_group_disjoint` | LightGBM under the stored group-disjoint splits of the full-data setting | LightGBM row of Table 10, LightGBM column of Table 11 |
| `12_step22_experimentA_rr_vs_gd` | Experiment A: the same cohort of rows (frozen subset; full dataset) split by the random-row and by the hash-group-disjoint protocol, 3 split seeds x 3 model seeds; LightGBM and MLP on the frozen cohort, LightGBM on the full cohort; per-run split audit, thresholds, row-weighted and group-balanced metrics, per-family recall | Section IV-D, V-F, Table expA |
| `13_step23_experimentB_exposure_control` | Experiment B: exposure control on a fixed test set; LightGBM trained at exposure levels B0, B50 and B100 of the same reserve, with a paired group bootstrap of the differences | Section IV-E, V-G, Table expB |
| `14_step24_paired_model_bootstrap` | Paired group bootstrap of the difference between two models scored on the same test rows: LightGBM versus MLP pairs of experiment A, and the legacy frozen-benchmark predictions (hybrid, No-G, MLP, LightGBM, E-GraphSAGE) at split seed 2027 | Section IV-F, V-J, Table paired |

### Running order

Notebook 01 writes the feature-hash cache and the full-data splits. Notebook 02 reads that cache and writes the frozen subset, its split and its scaler. Notebooks 03 to 07 read those files, and some of them also load checkpoints or scores written by earlier notebooks. Notebooks 08 to 11 read the dataset CSV files directly. Notebooks 12 to 14 are described in the section "Experiments A, B and the paired bootstrap (v1.1.0)" below.

The notebooks were written for Google Colab. They mount Google Drive, read the datasets from `/content/drive/MyDrive/` and write results under `/content/drive/MyDrive/THAT_IDS_RERUN/`. Edit the path variables in the configuration cell of each notebook to run them elsewhere. Notebooks 01 to 07 ran on an NVIDIA A100 with PyTorch 2.11.0; notebooks 08 to 11 run on CPU.

Notebooks 01 to 09 and 12 to 14 are stored with the outputs of their last run. A notebook that was run one dataset or one phase at a time shows only the last of those runs; the complete numbers are in the result files (see `results/`).

### Notes for readers of the notebooks

- **Naming.** The notebooks call the hybrid reconstruction-classification detector of the paper *THAT-IDS*, and its variants `full` (hybrid), `no_graph` (No-G), `ae_only` (AE), `mlp` and `full_shuffled_edges` (Shuffled). Experiment tags and output paths keep their original names.
- **Clipping.** In notebooks 01 to 07 the MinMax-scaled features are clipped to [-10, 10] before they reach a model. Notebooks 10 and 11 do not clip; clipping values that already lie outside the training range does not change the split points a tree model learned on training data.
- **Random-row split.** Notebook 01 builds the random-row baseline with one random generator per seed, shared across attack families (`nfdup.random_row_split`). Notebook 10 rebuilt the split with a generator re-seeded for every family, so its partition differs from that of notebook 01; its overlap is 93.47% (UNSW-v2) and 54.80% (CIC-v2), against 93.48% and 54.81% in the audit.
- **Row order.** Notebook 11 rebuilds the stored partitions from their recorded salts and checks the row, benign and attack counts of every split against `splits/paper_full_data_splits.json`. Rows inside each split are in file order, not in the shuffled order of notebook 01.
- **Additional experiments.** Notebook 01 also trains a `graph_classifier` variant, notebook 05 contains a relation ablation of the graph encoder, and notebook 02 has a screening phase before its confirmation phase. These are not reported in the paper.
- **Markdown cells.** The descriptive markdown cells of notebooks 01 to 11 were removed for this repository; the table above states what each notebook does. Notebooks 12 to 14 keep their markdown cells. Code cells and outputs are unchanged from the runs.

## Experiments A, B and the paired bootstrap (v1.1.0)

Notebooks 12 to 14 were added in version 1.1.0. They share one code cell (data loading, hash groups, Algorithm 1, the random-row split, thresholds, metrics and the paired group bootstrap), copied verbatim into each notebook so that every notebook is self-contained.

- **Cohorts.** Two cohorts of rows per dataset: the frozen attack-enriched subset of notebook 02 (245,994 UNSW-v2 rows, 496,384 CIC-v2 rows) and the full dataset.
- **Seeds.** Split seeds 42, 123 and 456 crossed with model seeds 42, 123 and 456 (3 x 3 per dataset, cohort, protocol and model).
- **Splits.** The new group-disjoint splits of the frozen cohort run Algorithm 1 with 24 candidate salts per split seed; the selected salt, the split audit and the 24 candidates are in `results/12_step22_experimentA_rr_vs_gd/splits/`. The full-cohort group-disjoint splits are rebuilt from the salts of `splits/paper_full_data_splits.json` (one candidate, no search). The random-row splits use `random_row_split` of notebook 01 (one generator per seed, stratified by family).
- **Preprocessing.** MinMax scaler fitted on the training split of each run; no clipping (`CLIP = None`), unlike the neural pipeline of notebooks 01 to 07.
- **Thresholds.** Selected on validation only: the threshold that maximizes F1-macro, and the thresholds at FPR <= 0.1% and FPR <= 0.01% (the 0.01% point is flagged when the validation split has too few benign rows to resolve it).
- **Fixed seeds.** Reserve of experiment B: `800000 + split seed`; exposure draw of experiment B: `900000 + split seed`; every bootstrap: 2,000 resamples with seed `20261111`.
- **Experiment B.** For each split seed, one donor row per multi-row, label-consistent test group is moved from the training side to a reserve; B0 trains without any group of the test set, B50 and B100 put back floor(0.5 b_f) and b_f donor rows per stratum, where b_f is at most 10% of the stratum's rows. The exposure pairs and budgets of every split are in `results/13_step23_experimentB_exposure_control/exposure/`.
- **Models.** LightGBM with the configuration of notebook 11 (deterministic, early stopping after 100 rounds on validation log-loss); the MLP of experiment A is the MLP control of notebooks 01 to 07 reimplemented in the notebook (same sizes, focal loss, optimizer and budget).

**What notebooks 12 and 13 need that the repository does not contain.** Both load the row indices of the frozen subset from `conference_step5_literature_subset_v1/subset_definition/{UNSW_V2,CIC_V2}/subset_indices_int32.npy` on Google Drive. This file is not in the repository, consistent with the list of excluded artifacts under "Result files" below; running notebook 02 on the dataset rebuilds it from the data and the recorded seed. Notebook 14 additionally reads the per-row scores written by notebook 12 (`journal_step22_rr_vs_gd_v1/predictions/`) and the legacy prediction files of notebooks 02, 03 and 04 (`test_predictions.npz`, `split_indices.npz`); none of these is in the repository.

**Not included** for steps 22 to 24: `predictions/` (per-row scores of every run), `splits/*.npz` (row indices of the 24 splits) and `exposure/*.npz` (row indices of the reserve and exposure sets). The notebooks rebuild them from the data, the seeds and the recorded salts.

**Runtime.** CIC-v2 is loaded whole, so the two notebooks that train on it need a high-memory machine (the runs used Google Colab). The MLP runs of notebook 12 need a GPU, or a long wait on CPU; LightGBM runs on CPU. Notebook 12 ran with torch 2.14.1+cu130, notebooks 13 and 14 with torch 2.14.1+cpu (`results/environment.json`, entry `notebooks_12_to_14`).

## Result files

`results/` holds the small result files (CSV, JSON, Markdown, LaTeX and text) written by the notebooks, one folder per notebook. `results/MANIFEST.csv` lists every file with its size and SHA-256 digest, and `results/environment.json` records the library versions of the session that collected them on 11 November 2026. Checkpoints, hash caches, per-row predictions, the row-index arrays of splits and exposure sets (`*.npz`), and `subset_indices_int32.npy` of the frozen subset are not included.

## Tests

```bash
pip install pytest
python -m pytest -q
```

The tests need no dataset. They extract the split functions from `notebooks/01_step4_full_data_group_disjoint.ipynb`, run them on synthetic data, and require `nfdup` to return the same salt, objective, partition and row order. They also check that every salt in `splits/paper_full_data_splits.json` is one of the 24 candidates generated from its seed. `tests/test_step22_splits.py` repeats the comparison for the split functions of `notebooks/12_step22_experimentA_rr_vs_gd.ipynb` and checks the salts recorded in `results/12_step22_experimentA_rr_vs_gd/splits/`.

## Citation

See `CITATION.cff`. Archived at Zenodo: concept DOI [10.5281/zenodo.24125935](https://doi.org/10.5281/zenodo.24125935) (all versions); v1.0.0 is [10.5281/zenodo.24125936](https://doi.org/10.5281/zenodo.24125936).

## License

MIT. See `LICENSE`. The datasets have their own terms of use.
