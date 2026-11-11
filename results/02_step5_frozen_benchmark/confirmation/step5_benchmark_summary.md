# THAT-IDS Step 5 literature-aligned benchmark

- Phase: **confirmation**
- Fixed split seed: 2027
- Model seeds: [42, 123, 456]
- Models: ['mlp', 'no_graph', 'full', 'full_shuffled_edges']

## UNSW_V2
- Decision: **CONTROLLED_NEAR_TIE_OR_NEGATIVE**
- Random test overlap: 0.868776
- Exact-group test overlap: 0.000000
- Interpretation: Full does not consistently improve over No-Graph on the fixed subset. Keep Step 4 as the robustness result and avoid a strong model-superiority claim.

## CIC_V2
- Decision: **READY_FOR_PUBLIC_BASELINES**
- Random test overlap: 0.553361
- Exact-group test overlap: 0.000000
- Interpretation: Full is supported against No-Graph and shuffled edges on the fixed subset across model seeds. Freeze the benchmark and run E-GraphSAGE and GraphIDS on the same indices.
