# Standalone Colab cell: train / validation / test group counts of the frozen benchmark (Table 1).
# No training. Standard runtime is enough: only the uint64 hash key of each row is kept in memory.
# Runtime: a few minutes for UNSW-v2, 10 to 20 minutes for CIC-v2 (CSV reading dominates).
from google.colab import drive
drive.mount('/content/drive')

import os, time, numpy as np, pandas as pd

DATA_DIR = '/content/drive/MyDrive'
STEP5 = '/content/drive/MyDrive/THAT_IDS_RERUN/conference_step5_literature_subset_v1'
EXCLUDE_COLS = ['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR',
                'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'Label', 'Attack']
DATASETS = {
    'UNSW_V2': dict(file='NF-UNSW-NB15-v2.csv', rows=2_390_275, groups=236_999, frozen_rows=245_994,
                    frozen_groups=41_640, expected_test_groups=8_379, expected_trainval_groups=33_261),
    'CIC_V2':  dict(file='NF-CSE-CIC-IDS2018-v2.csv', rows=18_893_708, groups=8_981_062, frozen_rows=496_384,
                    frozen_groups=233_966, expected_test_groups=46_867, expected_trainval_groups=187_099),
}


def row_hash_keys(path, n_rows, chunk=1_000_000):
    """Same preprocessing and hashing as the paper: 39 float32 columns in release order,
    non-finite values set to 0.0, pandas hash_pandas_object(index=False) -> uint64 per row."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    feat = [c for c in header if c not in EXCLUDE_COLS]
    assert len(feat) == 39, f'{len(feat)} feature columns, expected 39'
    keys = np.empty(n_rows, dtype=np.uint64)
    pos, t0 = 0, time.time()
    for ch in pd.read_csv(path, usecols=feat, chunksize=chunk):
        a = ch[feat].apply(pd.to_numeric, errors='coerce').astype(np.float32).to_numpy(copy=True)
        a[~np.isfinite(a)] = 0.0
        f = pd.DataFrame(a, columns=feat)
        keys[pos:pos + len(f)] = pd.util.hash_pandas_object(f, index=False).to_numpy()
        pos += len(f)
        print(f'\r  {pos:,} rows hashed ({time.time() - t0:.0f}s)', end='')
    print()
    assert pos == n_rows, f'read {pos:,} rows, expected {n_rows:,}'
    return keys


results = {}
for ds, cfg in DATASETS.items():
    print(f'== {ds}')
    keys = row_hash_keys(os.path.join(DATA_DIR, cfg['file']), cfg['rows'])
    n_groups = len(np.unique(keys))
    print(f'  full dataset groups: {n_groups:,} (audit {cfg["groups"]:,})')
    sub = np.load(f'{STEP5}/subset_definition/{ds}/subset_indices_int32.npy').astype(np.int64)
    assert len(sub) == cfg['frozen_rows'], f'subset has {len(sub):,} rows, expected {cfg["frozen_rows"]:,}'
    k = keys[sub]                                    # keys of the subset rows, in subset order
    assert len(np.unique(k)) == cfg['frozen_groups'], 'subset group count differs from the paper'
    z = np.load(f'{STEP5}/confirmation/{ds}/exact_feature_group/shared/seed_2027/split_indices.npz')
    counts = {name: int(len(np.unique(k[z[name]]))) for name in ['train_idx', 'val_idx', 'test_idx']}
    for a, b in [('train_idx', 'val_idx'), ('train_idx', 'test_idx'), ('val_idx', 'test_idx')]:
        assert not np.intersect1d(np.unique(k[z[a]]), np.unique(k[z[b]])).size, f'{a} and {b} share a group'
    print(f'  groups: train {counts["train_idx"]:,} | validation {counts["val_idx"]:,} | test {counts["test_idx"]:,}')
    print(f'  check: test {counts["test_idx"]:,} vs expected {cfg["expected_test_groups"]:,}; '
          f'train+val {counts["train_idx"] + counts["val_idx"]:,} vs expected {cfg["expected_trainval_groups"]:,}')
    results[ds] = counts
    del keys, k
print('\nSend these two lines back:')
for ds, c in results.items():
    print(f'{ds}: train={c["train_idx"]}, validation={c["val_idx"]}, test={c["test_idx"]}')
