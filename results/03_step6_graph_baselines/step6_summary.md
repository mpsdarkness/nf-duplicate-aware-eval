# THAT-IDS Step 6 controlled public-baseline summary

- Step-5 benchmark: `conference_step5_literature_subset_v1/confirmation`
- Step-6 tag: `conference_step6_public_baselines_v1`
- Fixed split seed: `2027`
- Model seeds: `[42, 123, 456]`
- Public baselines: `['egraphsage_public', 'graphids_public']`
- Comparison type: controlled reimplementation on frozen Step-5 rows/split/scaler

## CIC_V2
- Decision: **PUBLIC_BASELINE_COMPARISON_COMPLETE**
- Full vs egraphsage_public: ΔF1=+0.002476, ΔPR=+0.018871, ΔFPR=-0.003603. Full vs graphids_public: ΔF1=+0.133064, ΔPR=+0.246369, ΔFPR=-0.072162.

## UNSW_V2
- Decision: **PUBLIC_BASELINE_COMPARISON_COMPLETE**
- Full vs egraphsage_public: ΔF1=-0.001189, ΔPR=+0.000815, ΔFPR=+0.001238. Full vs graphids_public: ΔF1=+0.208159, ΔPR=+0.325811, ΔFPR=-0.146245.
