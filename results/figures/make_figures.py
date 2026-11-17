"""Reader-oriented versions of the Experiment A and B figures (Python backend).
Fig A: dumbbell chart, group-disjoint -> random-row, means with per-split-seed range, delta printed.
Fig B: F1 on the fixed test set under B0/B50/B100 with realized exposure, plus the paired difference."""
import sys, numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '.'); from audit_panel_alignment import require_matplotlib_panel_alignment
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7, 'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': 0.6, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'legend.fontsize': 6.5, 'legend.frameon': False})
GD, RR, SIGNAL, LIGHT = '#7F7F7F', '#1F4E79', '#B2182B', '#D9D9D9'
W = 7.16
def save(fig, name):
    require_matplotlib_panel_alignment(fig, json_out=f'{name}.alignment.json', overlay_svg=f'{name}.alignment.svg',
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    for ext, kw in [('pdf', {}), ('svg', {}), ('png', {'dpi': 600})]:
        fig.savefig(f'{name}.{ext}', bbox_inches='tight', pad_inches=0.03, **kw)
def plabel(ax, s): ax.text(-0.10, 1.14, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='right')

# ------------------------------------------------------------------ Figure A (dumbbell)
# level-2 means (mean of split-seed means) from the released summary; per-split means give the range
L = {  # (cohort, detector): (F1 GD, F1 RR, recall GD, recall RR)
 'UNSW-v2 frozen\nLightGBM': (0.9963, 0.9971, 0.769, 0.937), 'UNSW-v2 frozen\nMLP': (0.9949, 0.9956, 0.542, 0.661),
 'CIC-v2 frozen\nLightGBM': (0.9808, 0.9835, 0.948, 0.955), 'CIC-v2 frozen\nMLP': (0.9774, 0.9801, 0.916, 0.947),
 'UNSW-v2 full\nLightGBM': (0.9782, 0.9839, 0.928, 0.950), 'CIC-v2 full\nLightGBM': (0.9845, 0.9899, 0.9634, 0.9672)}
d = pd.read_csv('deltaA.csv'); d['key'] = d.dataset + '-v2 ' + d.cohort + '\n' + d.model.replace({'LGBM': 'LightGBM'})
rows = list(L)
fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={'wspace': 0.95})
for ax, (ig, ir, dcol), title, lab, fmt in zip(axes, [(0, 1, 'f1'), (2, 3, 'tpr01')],
        ['F1-macro at the F1-selected threshold', 'Recall at the threshold calibrated on validation (FPR $\\leq$ 0.1%)'],
        ['a', 'b'], ['{:+.4f}', '{:+.3f}']):
    for y, key in enumerate(rows[::-1]):
        g, r = L[key][ig], L[key][ir]
        ax.plot([g, r], [y, y], color=LIGHT, lw=2.2, solid_capstyle='round', zorder=1)
        ax.scatter([g], [y], s=22, color=GD, zorder=3); ax.scatter([r], [y], s=22, color=RR, zorder=3)
        dd = d[d.key == key][dcol]
        ax.text(1.02, y, fmt.format(dd.mean()) + f'  ({int((dd > 0).sum())}/9)', transform=ax.get_yaxis_transform(), fontsize=6, ha='left', va='center', color=SIGNAL)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows[::-1], fontsize=6.5)
    ax.set_xlabel(['F1-macro', 'Recall on test'][ig // 2]); ax.set_title(title, loc='left', pad=16); ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.text(1.02, 1.0, 'RR $-$ GD (pairs > 0)', transform=ax.transAxes, fontsize=6, ha='left', va='bottom', color=SIGNAL); plabel(ax, lab)
axes[0].set_xlim(0.974, 1.001); axes[1].set_xlim(0.50, 1.01)
h = [plt.Line2D([], [], marker='o', ls='', color=GD, ms=5, label='Group-disjoint split'),
     plt.Line2D([], [], marker='o', ls='', color=RR, ms=5, label='Random-row split (same rows)')]
fig.legend(handles=h, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
fig.text(0.5, -0.14, 'Each row: mean over three split seeds and three model seeds. Red column: mean of random-row minus group-disjoint over the nine seed pairs, and the number of pairs in which it is positive.',
         ha='center', fontsize=6.5, color='#444444')
fig.subplots_adjust(left=0.12, right=0.90, top=0.82, bottom=0.22)
save(fig, 'fig_expA_v2')

# ------------------------------------------------------------------ Figure B (conditions on a fixed test set)
b = pd.read_csv('expB.csv')
exp = {'CIC': {'B0': 0, 'B50': 24.4, 'B100': 53.7}, 'UNSW': {'B0': 0, 'B50': 43.0, 'B100': 88.0}}
fig, axes = plt.subplots(1, 3, figsize=(W, 2.5), gridspec_kw={'wspace': 0.55})
for ax, ds, lab in zip(axes[:2], ['CIC', 'UNSW'], ['a', 'b']):
    sub = b[b.dataset == ds]
    for x, cond in enumerate(['B0', 'B50', 'B100']):
        v = sub[sub.cond == cond].f1
        ax.scatter(np.full(len(v), x) + np.linspace(-0.12, 0.12, len(v)), v, s=10, color=LIGHT, zorder=2)
        ax.hlines(v.mean(), x - 0.3, x + 0.3, color=RR, lw=2, zorder=3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([f'{c}\n{exp[ds][c]:.0f}% of $T$\nexposed' for c in ['B0', 'B50', 'B100']]); ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel('F1-macro on the fixed test set $T$'); ax.set_title(f'{ds}-v2: same test set,\nmore of it exposed', loc='left', pad=6)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4)); ax.margins(y=0.2); plabel(ax, lab)
piv = b.pivot_table(index=['dataset', 'split_seed', 'model_seed'], columns='cond', values='f1')
ax = axes[2]
for x, (ds, cond) in enumerate([('CIC', 'B50'), ('CIC', 'B100'), ('UNSW', 'B50'), ('UNSW', 'B100')]):
    v = (piv.loc[ds, cond] - piv.loc[ds, 'B0']).to_numpy()
    ax.scatter(np.full(9, x) + np.linspace(-0.15, 0.15, 9), v, s=10, color=LIGHT, zorder=2)
    ax.hlines(v.mean(), x - 0.3, x + 0.3, color=RR, lw=2, zorder=3)
    ax.text(x, 0.0021, f'{v.mean():+.4f}', ha='center', fontsize=6, color=SIGNAL)
ax.axhline(0, color='#999999', lw=0.6, ls='--', zorder=1)
ax.set_xticks(range(4)); ax.set_xticklabels(['B50', 'B100', 'B50', 'B100']); ax.set_xlim(-0.6, 3.6); ax.set_ylim(-0.0016, 0.0026)
ax.text(0.5, -0.30, 'CIC-v2', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.5)
ax.text(2.5, -0.30, 'UNSW-v2', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.5)
ax.set_ylabel('Change in F1-macro versus B0'); ax.set_title('B50/B100 minus B0\n(paired, same test set)', loc='left', pad=6); plabel(ax, 'c')
h = [plt.Line2D([], [], marker='o', ls='', color=LIGHT, ms=4, label='One run (3 split seeds $\\times$ 3 model seeds)'),
     plt.Line2D([], [], color=RR, lw=2, label='Mean of nine runs')]
fig.legend(handles=h, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.06))
fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.36)
save(fig, 'fig_expB_v2'); print('saved')
