# Step 7 manuscript fill guide

Use values exactly as shown in `step7_paper_fill_rows.csv`. FPR values are percentages, not rates.

## Table VII — NF-UNSW-NB15-v2
- LightGBM: F1=0.9964±0.0001; PR-AUC=0.9991±0.0000; FPR=0.429±0.000%; TPR=0.9997.
- Full-LOO: F1=0.9944±0.0018; PR-AUC=0.9975±0.0002; FPR=0.631±0.209%; TPR=0.9985.

## Table VIII — NF-CSE-CIC-IDS2018-v2
- LightGBM: F1=0.9786±0.0001; PR-AUC=0.9747±0.0001; FPR=0.0441±0.0103%; TPR=0.9377.
- Full-LOO: F1=0.9737±0.0007; PR-AUC=0.9825±0.0014; FPR=0.0144±0.0049%; TPR=0.9226.

## Suggested prose — UNSW_V2
- Full vs LightGBM: ΔF1=-0.001618, ΔPR-AUC=-0.001666, ΔFPR=+0.1771 percentage points.
- Full vs Full-LOO: ΔF1=+0.000430, ΔPR-AUC=-0.000086, ΔFPR=-0.0252 percentage points.
- Interpret Full-LOO as a conservative no-self diagnostic, not an exact one-at-a-time retraining experiment.

## Suggested prose — CIC_V2
- Full vs LightGBM: ΔF1=-0.004994, ΔPR-AUC=+0.009287, ΔFPR=-0.0218 percentage points.
- Full vs Full-LOO: ΔF1=-0.000103, ΔPR-AUC=+0.001457, ΔFPR=+0.0079 percentage points.
- Interpret Full-LOO as a conservative no-self diagnostic, not an exact one-at-a-time retraining experiment.
