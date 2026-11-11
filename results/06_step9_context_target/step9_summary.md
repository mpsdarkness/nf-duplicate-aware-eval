# Step 9 — Context-target gradient control

Gate initialization: `theta0 = -1.5` -> `g0 = 0.182426`.

Frozen Step-5 rows and split (benchmark seed 2027), Full variant only,
model seeds [42, 123, 456]. Only the context-target gradient path differs.

## Gate comparison

| dataset   | context_target_mode   |   gate_mean |   gate_min |   gate_max |   max_abs_shift_from_init |   n_seeds |
|:----------|:----------------------|------------:|-----------:|-----------:|--------------------------:|----------:|
| CIC_V2    | published_step5_step8 |    0.18511  |   0.184949 |   0.185217 |                0.00279148 |         3 |
| CIC_V2    | detach                |    0.185924 |   0.185715 |   0.186041 |                0.00361498 |         3 |
| CIC_V2    | drop                  |    0.185947 |   0.185758 |   0.186053 |                0.00362795 |         3 |
| UNSW_V2   | published_step5_step8 |    0.184268 |   0.184123 |   0.184362 |                0.00193648 |         3 |
| UNSW_V2   | detach                |    0.185025 |   0.184952 |   0.185096 |                0.00267007 |         3 |
| UNSW_V2   | drop                  |    0.185047 |   0.18497  |   0.185117 |                0.00269197 |         3 |

## Frozen-benchmark metrics

| context_target_mode   | dataset   |   n_seeds |   gate_mean |    gate_std |   gate_min |   gate_max |   f1_macro_mean |   f1_macro_std |   pr_auc_mean |   pr_auc_std |   fpr_percent_mean |   fpr_percent_std |   tpr_mean |     tpr_std |   gate_max_abs_shift_from_init |
|:----------------------|:----------|----------:|------------:|------------:|-----------:|-----------:|----------------:|---------------:|--------------:|-------------:|-------------------:|------------------:|-----------:|------------:|-------------------------------:|
| detach                | CIC_V2    |         3 |    0.185924 | 0.000181372 |   0.185715 |   0.186041 |        0.97288  |    0.00110101  |      0.983809 |  0.0011302   |          0.0231852 |        0.00424991 |   0.920611 | 0.00328308  |                     0.00361498 |
| detach                | UNSW_V2   |         3 |    0.185025 | 7.17366e-05 |   0.184952 |   0.185096 |        0.994802 |    0.00212767  |      0.997349 |  0.00026412  |          0.609775  |        0.235361   |   0.999212 | 0.000676161 |                     0.00267007 |
| drop                  | CIC_V2    |         3 |    0.185947 | 0.000164075 |   0.185758 |   0.186053 |        0.97242  |    0.000532689 |      0.984007 |  0.000955485 |          0.0199393 |        0.00212496 |   0.919191 | 0.00146552  |                     0.00362795 |
| drop                  | UNSW_V2   |         3 |    0.185047 | 7.41094e-05 |   0.18497  |   0.185117 |        0.994939 |    0.0019112   |      0.997335 |  0.000294716 |          0.595687  |        0.21097    |   0.99928  | 0.000627216 |                     0.00269197 |

## How to read this

If the `detach` gate stays inside the published 0.1817-0.1853 band, the
non-detached target was not what pinned the gate: the paper keeps its
current numbers and states this explicitly. If it moves materially, the
main results must be regenerated from the `detach` arm and the gate
discussion rewritten.