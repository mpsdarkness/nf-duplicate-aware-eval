# THAT-IDS Step 5 literature-aligned benchmark

- Phase: **screening**
- Fixed split seed: 2027
- Model seeds: [42]
- Models: ['mlp', 'no_graph', 'full']

## UNSW_V2
- Decision: **REVIEW_BEFORE_CONFIRMATION**
- Random test overlap: 0.868776
- Exact-group test overlap: 0.000000
- Interpretation: The fixed-subset screening does not yet show a clear advantage. Inspect F1, PR-AUC, FPR, TPR, and hard classes before spending compute on public baselines.

## CIC_V2
- Decision: **ADVANCE_TO_CONFIRMATION**
- Random test overlap: 0.553361
- Exact-group test overlap: 0.000000
- Interpretation: Full shows a useful signal over No-Graph and remains competitive with MLP on the fixed subset. Switch PHASE to 'confirmation' before running public baselines.
