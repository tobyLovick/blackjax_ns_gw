"""
Plot PP (probability-probability) curves from pp_summary.npy.

Usage:
    python plot_pp.py [--summary pp_summary.npy] [--out pp_test.pdf]

Produces a 2×2 grid of panels (one per mode). In each panel:
  - One PP curve per physical parameter (7 lines)
  - Diagonal y=x reference
  - KS p-value annotated for each parameter
  - 1σ / 2σ confidence bands for N realisations of a uniform distribution
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

parser = argparse.ArgumentParser()
parser.add_argument('--summary', default='pp_summary.npy')
parser.add_argument('--out',     default='pp_test.pdf')
args = parser.parse_args()

summary  = np.load(args.summary, allow_pickle=True).item()
Q        = summary['quantiles']   # (N, 4, 7)
MODES    = summary['modes']
PARAMS   = summary['params']
N        = Q.shape[0]

PARAM_LABELS = {
    'M_c':     r'$\mathcal{M}_c$',
    'q':       r'$q$',
    'iota':    r'$\iota$',
    'd_L':     r'$d_L$',
    't_c':     r'$t_c$',
    'phase_c': r'$\phi_c$',
    'psi':     r'$\psi$',
}
MODE_TITLES = {
    'norm_true':  'Norm  |  True PSD',
    'norm_welch': 'Norm  |  Welch PSD',
    'alcs_true':  'ALCS  |  True PSD',
    'alcs_welch': 'ALCS  |  Welch PSD',
}

# Confidence bands from order statistics of U[0,1]
cdf_x    = np.linspace(0, 1, 500)
n_draws  = 10000
fake_qs  = np.sort(np.random.uniform(0, 1, (n_draws, N)), axis=1)
ecdf_sim = np.linspace(1/N, 1, N)
lo1, hi1 = np.percentile(fake_qs, [16, 84], axis=0)
lo2, hi2 = np.percentile(fake_qs, [ 2, 98], axis=0)

colors = plt.cm.tab10(np.linspace(0, 0.7, len(PARAMS)))

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)

for mi, (mode, ax) in enumerate(zip(MODES, axes.flat)):
    ax.fill_between(ecdf_sim, lo2, hi2, color='grey', alpha=0.15, linewidth=0)
    ax.fill_between(ecdf_sim, lo1, hi1, color='grey', alpha=0.30, linewidth=0)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, zorder=5)

    ks_lines = []
    for pi, (param, col) in enumerate(zip(PARAMS, colors)):
        qs_sorted = np.sort(Q[:, mi, pi])
        ecdf      = np.linspace(1/N, 1, N)
        ks_p      = kstest(Q[:, mi, pi], 'uniform').pvalue
        lbl       = f"{PARAM_LABELS.get(param, param)}  (p={ks_p:.2f})"
        ax.step(qs_sorted, ecdf, color=col, lw=1.2, label=lbl, where='post')
        ks_lines.append((ks_p, lbl, col))

    ax.set_title(MODE_TITLES[mode], fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=6.5, loc='upper left', framealpha=0.7)

for ax in axes[1]:
    ax.set_xlabel('Quantile of truth in posterior', fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel('Empirical CDF', fontsize=10)

fig.suptitle(f'PP test  (N={N} injections, IMRPhenomD, H1+L1)',
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(args.out, bbox_inches='tight')
print(f"Saved {args.out}")
plt.show()
