# Changelog

## v1.1.0 (unreleased)

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
