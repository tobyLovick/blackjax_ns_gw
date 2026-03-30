from anesthetic import read_chains, make_2d_axes
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Parameters to show and their labels
params = ['M_c', 'q', 'd_L', 'iota', 's1_z', 's2_z']
paramnames = [r'$\mathcal{M}_c\ [M_\odot]$', r'$q$', r'$d_L\ [\mathrm{Mpc}]$',
              r'$\iota\ [\mathrm{rad}]$', r'$\chi_1$', r'$\chi_2$']

# Injection truths (4s signal)
truths = {
    'M_c':   35.0,
    'q':     0.90,
    'd_L':   1000.0,
    'iota':  0.40,
    's1_z':  0.40,
    's2_z': -0.30,
}

fig, axes = make_2d_axes(params, figsize=(7, 6.5), upper=False,
                         labels=dict(zip(params, paramnames)))

# --- blackjax-ns baseline (fixed PSD) ---
try:
    samples_bj = read_chains('blackjaxns_nlive1400.csv')
    samples_bj.plot_2d(axes, c='steelblue',
                       kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                       label='blackjax-ns (fixed PSD)', bw_method=0.4)
except FileNotFoundError:
    pass

# --- blackjax-ns ALCS (noise-marginalised) ---
samples_alcs = read_chains('blackjaxns_alcs_nlive1400.csv')
samples_alcs.plot_2d(axes, c='firebrick',
                     kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                     label='blackjax-ns ALCS', bw_method=0.4)

# --- bilby chain (optional) ---
try:
    samples_bilby = read_chains('bilby_4s.csv')
    samples_bilby.plot_2d(axes, c='darkorange',
                          kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                          label='bilby', bw_method=0.4)
except FileNotFoundError:
    pass

# --- Injection truth lines ---
for i, p in enumerate(params):
    axes.iloc[i, i].axvline(truths[p], color='k', lw=0.8, ls='--')
    for j in range(len(params)):
        if j < i:
            axes.iloc[i, j].axvline(truths[p],         color='k', lw=0.8, ls='--', alpha=0.6)
            axes.iloc[i, j].axhline(truths[params[j]], color='k', lw=0.8, ls='--', alpha=0.6)

# --- Legend ---
handles = [
    mlines.Line2D([], [], color='steelblue', lw=2, label='blackjax-ns (fixed PSD)'),
    mlines.Line2D([], [], color='firebrick', lw=2, label='blackjax-ns ALCS'),
    mlines.Line2D([], [], color='darkorange', lw=2, label='bilby'),
    mlines.Line2D([], [], color='k', lw=0.8, ls='--', label='injection'),
]
axes.iloc[0, 0].legend(handles=handles, loc='lower center',
                       bbox_to_anchor=(3.5, 1.05), ncol=4, frameon=False)

fig.savefig('triangle_alcs_4s.pdf', bbox_inches='tight')
print("Saved triangle_alcs_4s.pdf")
