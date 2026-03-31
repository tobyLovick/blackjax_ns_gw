import matplotlib.lines as mlines
from anesthetic import read_chains, make_2d_axes

# Normalisation correction to bring the standard fixed-PSD logZ onto the same
# scale as the fully-normalised ALCS/InvGamma evidences. Computed once from
# the detector data/PSD and hardcoded here:
#   Σ[-log Ŝᵢ] + N·log(2Δf/π) - 2Δf·Σ|dᵢ|²/Ŝᵢ
#   = +1293289.61 - 22148.26 - 12818.36 = +1258322.99 nats
logZ_correction = 1258322.99

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
params = ['M_c', 'q', 'd_L', 'iota', 's1_z', 's2_z', 'tau', 'alpha_0']
paramnames = [
    r'$\mathcal{M}_c\ [M_\odot]$', r'$q$', r'$d_L\ [\mathrm{Mpc}]$',
    r'$\iota\ [\mathrm{rad}]$', r'$\chi_1$', r'$\chi_2$',
    r'$\tau$', r'$\alpha_0$',
]

fig, axes = make_2d_axes(params, figsize=(10, 9), upper=False,
                         labels=dict(zip(params, paramnames)))

legend_handles = []
logZ_std_corrected = None

# baseline (fixed PSD) — logZ corrected onto same normalisation scale
try:
    samples_bj = read_chains('blackjaxns_nlive1400.csv')
    samples_bj.plot_2d(axes, c='steelblue', alpha=0.5,
                       kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                       bw_method=0.4)
    logZ_std_corrected = samples_bj.logZ() + logZ_correction
    legend_handles.append(mlines.Line2D([], [], color='steelblue', lw=2, label='fixed PSD'))
except FileNotFoundError:
    pass

# ALCS — has tau column
logZ_alcs = None
try:
    samples_alcs = read_chains('blackjaxns_alcs_nlive1400.csv')
    samples_alcs.columns = samples_alcs.columns.set_levels(
        [l if l != 'Unnamed: 13_level_1' else r'$\tau$'
         for l in samples_alcs.columns.get_level_values(1)], level=1)
    samples_alcs.plot_2d(axes, c='firebrick', alpha=0.5,
                         kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                         bw_method=0.4)
    logZ_alcs = samples_alcs.logZ()
    legend_handles.append(mlines.Line2D([], [], color='firebrick', lw=2, label=r'ALCS ($\tau$)'))
except FileNotFoundError:
    pass

# Inverse-gamma — has alpha_0 column
logZ_invg = None
try:
    samples_invg = read_chains('blackjaxns_invg_nlive1400.csv')
    samples_invg.plot_2d(axes, c='darkorange', alpha=0.5,
                         kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                         bw_method=0.4)
    logZ_invg = samples_invg.logZ()
    legend_handles.append(mlines.Line2D([], [], color='darkorange', lw=2, label=r'InvGamma ($\alpha_0$)'))
except FileNotFoundError:
    pass

if legend_handles:
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=len(legend_handles), frameon=False)

# Evidence annotation: ΔlogZ relative to corrected standard
evidence_lines = []
if logZ_std_corrected is not None and logZ_alcs is not None:
    dZ = logZ_alcs - logZ_std_corrected
    evidence_lines.append(
        rf'$\Delta\ln\mathcal{{Z}}_\mathrm{{ALCS}} = {dZ.mean():.1f} \pm {dZ.std():.1f}$')
if logZ_std_corrected is not None and logZ_invg is not None:
    dZ = logZ_invg - logZ_std_corrected
    evidence_lines.append(
        rf'$\Delta\ln\mathcal{{Z}}_\mathrm{{InvG}} = {dZ.mean():.1f} \pm {dZ.std():.1f}$')

ax44 = axes.iloc[4, 4]
ax44.annotate('\n'.join(evidence_lines),
              xy=(0.5, 1.35), xycoords='axes fraction',
              va='bottom', ha='center', fontsize=8, linespacing=1.8)

fig.savefig('triangle_all_4s.pdf', bbox_inches='tight')
print("Saved triangle_all_4s.pdf")
