# Changelog

## v1.1.1 (2026-11-17)

Documentation and metadata fixes (Experiment B donor description, notebook-to-table map, per-notebook environment evidence); adds Table 2 group counts, figure sources and the group-weighted attack-share table; no result changes.

Added:

- `results/02_step5_frozen_benchmark/table1_group_counts_standalone.py` and `table1_group_counts_result.json`: train, validation and test group counts of the frozen benchmark (Table 2).
- `results/figures/`: `make_figures.py`, `deltaA.csv`, `expB.csv`, the two rendered figures (PNG) and the four layout-check reports (JSON) of Fig. 3 and Fig. 4.
- `results/15_step25_group_weighted_attack_share/`: the standalone cell and its CSV and JSON output.

Changed:

- `README.md`: Experiment B donor description corrected (donors are drawn from the test side H of the group-disjoint partition, none from training; the factors 0.5 and 1 scale the budget b_f); rows 12 to 14 of the notebook table use the table and figure numbers of the submitted manuscript; link to the official dataset page; new sections "Table 2 group counts", "Group-weighted attack share", "Figure sources" and "What the package does not contain".
- `results/environment.json`: one entry per group of notebooks with its evidence; the collection session of 11 November 2026 is a separate entry.
- `results/MANIFEST.csv`: 14 rows appended for the new files; earlier rows unchanged.
- `CITATION.cff` and `.zenodo.json`: version 1.1.1.

No file in `nfdup/`, no notebook and no earlier result file was changed.

## v1.1.0 (2026-11-12)

Zenodo DOI 10.5281/zenodo.24155487 (concept DOI 10.5281/zenodo.24125935).

Adds experiments A (random-row versus group-disjoint, crossed seeds), B (exposure control on a fixed test set) and the paired group bootstraps (steps 22 to 24).

Added:

- `notebooks/12_step22_experimentA_rr_vs_gd.ipynb`, `notebooks/13_step23_experimentB_exposure_control.ipynb`, `notebooks/14_step24_paired_model_bootstrap.ipynb` (executed, with outputs).
- `results/12_step22_experimentA_rr_vs_gd/`: `reports/` (9 files), `splits/` (24 split metadata files and the 6 candidate tables of the new frozen-cohort splits), `runs/` (108 run files).
- `results/13_step23_experimentB_exposure_control/`: `reports/` (5 files), `exposure/` (6 metadata files, 6 exposure-pair tables, 6 budget tables), `runs/` (54 run files).
- `results/14_step24_paired_model_bootstrap/reports/` (3 files).
- `tests/test_step22_splits.py`: checks the split functions of notebook 12 against `nfdup`.
- `CHANGELOG.md`.

Changed:

- `README.md`: notebook table extended; new section "Experiments A, B and the paired bootstrap (v1.1.0)".
- `results/MANIFEST.csv`: 227 rows appended for the new files; earlier rows unchanged.
- `results/environment.json`: entry `notebooks_12_to_14` with the library versions printed by the new notebooks.
- `CITATION.cff` and `.zenodo.json`: version 1.1.0.

Not included, as before: per-row predictions (`predictions/`), the row-index arrays of the splits (`splits/*.npz`) and of the exposure sets (`exposure/*.npz`), model checkpoints, and `subset_indices_int32.npy` of the frozen subset.

## v1.0.0 (2026-11-11)

Initial release. Zenodo DOI 10.5281/zenodo.24125936 (concept DOI 10.5281/zenodo.24125935).
