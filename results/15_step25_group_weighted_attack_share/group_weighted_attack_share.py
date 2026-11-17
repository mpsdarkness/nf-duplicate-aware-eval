# Standalone Colab cell: attack share of the frozen-benchmark test split under row weighting and
# under group-balanced weighting, with the number of benign and attack groups.
# Supports the group-balanced paragraph of Section V-L (supplementary table). No training.
# Standard runtime is enough: only the uint64 hash key and the label of each row are kept.
from google.colab import drive
drive.mount('/content/drive')
import os, time, json, numpy as np, pandas as pd

DATA_DIR = '/content/drive/MyDrive'
STEP5 = '/content/drive/MyDrive/THAT_IDS_RERUN/conference_step5_literature_subset_v1'
OUT = '/content/drive/MyDrive/THAT_IDS_RERUN/journal_step25_group_weighted_attack_share_v1'
os.makedirs(OUT, exist_ok=True)
EXCLUDE_COLS = ['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR',
                'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'Label', 'Attack']
DATASETS = {'UNSW_V2': dict(file='NF-UNSW-NB15-v2.csv', rows=2_390_275, groups=236_999, frozen_rows=245_994, test_groups=8_379),
            'CIC_V2':  dict(file='NF-CSE-CIC-IDS2018-v2.csv', rows=18_893_708, groups=8_981_062, frozen_rows=496_384, test_groups=46_867)}


def keys_and_labels(path, n_rows, chunk=1_000_000):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    feat = [c for c in header if c not in EXCLUDE_COLS]
    assert len(feat) == 39
    keys = np.empty(n_rows, dtype=np.uint64); y = np.empty(n_rows, dtype=np.int8); pos, t0 = 0, time.time()
    for ch in pd.read_csv(path, usecols=feat + ['Label'], chunksize=chunk):
        a = ch[feat].apply(pd.to_numeric, errors='coerce').astype(np.float32).to_numpy(copy=True)
        a[~np.isfinite(a)] = 0.0
        keys[pos:pos + len(a)] = pd.util.hash_pandas_object(pd.DataFrame(a, columns=feat), index=False).to_numpy()
        y[pos:pos + len(a)] = ch['Label'].to_numpy().astype(np.int8); pos += len(a)
        print(f'\r  {pos:,} rows ({time.time() - t0:.0f}s)', end='')
    print(); assert pos == n_rows
    return keys, y


rows_out = []
for ds, cfg in DATASETS.items():
    print('==', ds)
    keys, y = keys_and_labels(os.path.join(DATA_DIR, cfg['file']), cfg['rows'])
    assert len(np.unique(keys)) == cfg['groups']
    sub = np.load(f'{STEP5}/subset_definition/{ds}/subset_indices_int32.npy').astype(np.int64)
    assert len(sub) == cfg['frozen_rows']
    z = np.load(f'{STEP5}/confirmation/{ds}/exact_feature_group/shared/seed_2027/split_indices.npz')
    for split in ['train_idx', 'val_idx', 'test_idx']:
        rows = sub[z[split]]; k, lab = keys[rows], y[rows]
        uk, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
        g_label = np.zeros(len(uk), dtype=np.int8); np.maximum.at(g_label, inv, lab)   # label per group (groups are label-consistent)
        g_min = np.ones(len(uk), dtype=np.int8); np.minimum.at(g_min, inv, lab)
        assert (g_label == g_min).all(), 'a group mixes binary labels'
        w = 1.0 / cnt[inv]                                                           # group-balanced row weights
        r = dict(dataset=ds, split=split.replace('_idx', ''), rows=int(len(rows)), groups=int(len(uk)),
                 attack_rows=int(lab.sum()), benign_rows=int((lab == 0).sum()),
                 attack_groups=int(g_label.sum()), benign_groups=int((g_label == 0).sum()),
                 attack_share_row_pct=round(100 * lab.mean(), 4),
                 attack_share_group_balanced_pct=round(100 * float(w[lab == 1].sum() / w.sum()), 4),
                 largest_group_rows=int(cnt.max()), largest_benign_group_rows=int(cnt[g_label == 0].max()),
                 largest_attack_group_rows=int(cnt[g_label == 1].max()))
        rows_out.append(r); print(r)
    if cfg['test_groups'] != [r for r in rows_out if r['dataset'] == ds and r['split'] == 'test'][0]['groups']:
        raise RuntimeError('test group count differs from Table 2')
    del keys, y
df = pd.DataFrame(rows_out)
df.to_csv(f'{OUT}/group_weighted_attack_share.csv', index=False)
json.dump(dict(generated='standalone cell, frozen benchmark split seed 2027', rows=rows_out), open(f'{OUT}/group_weighted_attack_share.json', 'w'), indent=1)
print('\nSaved to', OUT); print(df.to_string(index=False))
