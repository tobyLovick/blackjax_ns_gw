from anesthetic import read_chains, make_2d_axes

params = ['M_c', 'q', 'd_L', 'iota', 's1_z', 's2_z', 'tau', 'alpha_0']
paramnames = [
    r'$\mathcal{M}_c\ [M_\odot]$', r'$q$', r'$d_L\ [\mathrm{Mpc}]$',
    r'$\iota\ [\mathrm{rad}]$', r'$\chi_1$', r'$\chi_2$',
    r'$\tau$', r'$\alpha_0$',
]

fig, axes = make_2d_axes(params, figsize=(10, 9), upper=False,
                         labels=dict(zip(params, paramnames)))

# baseline (fixed PSD, no tau/alpha_0 column — anesthetic skips missing params)
try:
    samples_bj = read_chains('blackjaxns_nlive1400.csv')
    samples_bj.plot_2d(axes, c='steelblue',
                       kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                       bw_method=0.4)
except FileNotFoundError:
    pass

# ALCS — has tau column
try:
    samples_alcs = read_chains('blackjaxns_alcs_nlive1400.csv')
    samples_alcs.plot_2d(axes, c='firebrick',
                         kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                         bw_method=0.4)
except FileNotFoundError:
    pass

# Inverse-gamma — has alpha_0 column
try:
    samples_invg = read_chains('blackjaxns_invg_nlive1400.csv')
    samples_invg.plot_2d(axes, c='darkorange',
                         kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                         bw_method=0.4)
except FileNotFoundError:
    pass

fig.savefig('triangle_all_4s.pdf', bbox_inches='tight')
print("Saved triangle_all_4s.pdf")
