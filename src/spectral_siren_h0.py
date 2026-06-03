"""
Toy spectral siren H_0 inference.

Demonstrates that Welch-induced M_c bias propagates into biased H_0.

The spectral siren idea:
    M_c^det = M_c^source * (1 + z),   z = z(d_L, H_0)

Knowing the source-frame mass distribution shape (its peak), the observed
detector-frame M_c combined with d_L constrains H_0.

Per-event likelihood (importance-sampled from PE posterior):
    L_i(H_0) = mean_s [ p_pop(M_c_s / (1+z_s)) / (1+z_s) ]
where z_s = H_0 * d_L_s / C_KM_S  (Hubble law; adequate for this comparison)
and M_c_s, d_L_s are PE posterior samples for event i.

Combined log-posterior: log p(H_0) = sum_i log L_i(H_0)

Population model: Power Law + Gaussian peak in M_c space (fixed to truth).
Selection effects: not corrected — the relative comparison between modes is valid.

Usage:
    # Synthetic test (before catalog runs complete):
    python spectral_siren_h0.py --mock

    # Real PE samples from population mock runs:
    python spectral_siren_h0.py --results pop_results --catalog injection_catalog.npy
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as sp_norm

try:
    from anesthetic import read_chains
    HAS_ANESTHETIC = True
except ImportError:
    HAS_ANESTHETIC = False

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--results',          default='pop_results')
parser.add_argument('--catalog',          default='injection_catalog.npy')
parser.add_argument('--h0-true',          type=float, default=70.0)
parser.add_argument('--h0-min',           type=float, default=40.0)
parser.add_argument('--h0-max',           type=float, default=120.0)
parser.add_argument('--h0-steps',         type=int,   default=300)
parser.add_argument('--out',              default='h0_result.npy')
parser.add_argument('--mock',             action='store_true',
                    help='Use synthetic PE samples (for testing)')
parser.add_argument('--mock-n-samples',   type=int,   default=2000,
                    help='PE samples per event in mock mode')
parser.add_argument('--mock-bias-sigma',  type=float, default=1.0,
                    help='Systematic Welch M_c bias in units of sigma_Mc')
args = parser.parse_args()

C_KM_S   = 2.998e5   # km/s
H0_TRUE  = args.h0_true
H0_GRID  = np.linspace(args.h0_min, args.h0_max, args.h0_steps)
MODES    = ['norm_true', 'norm_welch', 'alcs_welch']

# ---------------------------------------------------------------------------
# Population model in M_c space: Power Law + Gaussian peak
#
# The injection catalog draws m1 from Power Law+Peak (mu_m1=34, sigma_m1=4).
# In M_c space this maps to a peak around Mc ~ 26 Msun (for typical q~0.8).
# We define the population model directly in Mc space for this toy analysis.
# ---------------------------------------------------------------------------
POP_ALPHA   = 2.5     # power-law slope
POP_MC_MIN  = 4.0     # lower truncation [Msun]
POP_MC_MAX  = 45.0    # upper truncation [Msun]
POP_MU_MC   = 26.0    # Gaussian peak location [Msun]
POP_SIG_MC  = 3.0     # Gaussian peak width [Msun]
POP_LAM     = 0.08    # fraction in peak (inflated slightly from m1-space lambda
                      # because the mapping m1->Mc concentrates the peak)

def _pl_norm(alpha, mc_min, mc_max):
    exp = 1.0 - alpha
    return (mc_max**exp - mc_min**exp) / exp

_PL_NORM = _pl_norm(POP_ALPHA, POP_MC_MIN, POP_MC_MAX)

def p_pop(mc):
    """Unnormalised (but consistently normalised) population model in Mc space."""
    mc = np.asarray(mc)
    in_range = (mc >= POP_MC_MIN) & (mc <= POP_MC_MAX)
    pl   = np.where(in_range, mc**(-POP_ALPHA) / _PL_NORM, 0.0)
    peak = sp_norm.pdf(mc, POP_MU_MC, POP_SIG_MC)
    return (1.0 - POP_LAM) * pl + POP_LAM * peak

# ---------------------------------------------------------------------------
# Spectral siren likelihood for one event
# ---------------------------------------------------------------------------
def event_likelihood(mc_det_samples, dl_samples, weights, h0):
    """
    Importance-sampled per-event spectral siren likelihood.
    mc_det_samples : (N,) detector-frame Mc PE samples
    dl_samples     : (N,) d_L PE samples [Mpc]
    weights        : (N,) normalised posterior weights
    h0             : scalar H_0 [km/s/Mpc]
    """
    z        = h0 * dl_samples / C_KM_S
    mc_src   = mc_det_samples / (1.0 + z)
    integrand = p_pop(mc_src) / (1.0 + z)
    return np.average(integrand, weights=weights)

# ---------------------------------------------------------------------------
# Compute log p(H_0) across all events for one mode
# ---------------------------------------------------------------------------
def logposterior_h0(all_mc, all_dl, all_w, h0_grid):
    """
    all_mc  : list of (N_s,) arrays, one per event
    all_dl  : list of (N_s,) arrays, one per event
    all_w   : list of (N_s,) weight arrays
    h0_grid : (N_h0,) grid
    Returns: (N_h0,) log-posterior (unnormalised)
    """
    logp = np.zeros(len(h0_grid))
    for mc, dl, w in zip(all_mc, all_dl, all_w):
        likelihoods = np.array([event_likelihood(mc, dl, w, h0) for h0 in h0_grid])
        likelihoods = np.clip(likelihoods, 1e-300, None)
        logp += np.log(likelihoods)
    return logp

# ---------------------------------------------------------------------------
# MOCK MODE: generate synthetic PE samples
# ---------------------------------------------------------------------------
def generate_mock_samples(catalog, rng):
    """
    For each event, generate synthetic PE samples for each mode.
    Welch samples are shifted in Mc^det by mock_bias_sigma * sigma_Mc.
    Returns: dict mode -> (list of mc arrays, list of dl arrays, list of weight arrays)
    """
    data = {m: {'mc': [], 'dl': [], 'w': []} for m in MODES}
    N_s = args.mock_n_samples

    for ev in catalog:
        mc_src  = float(ev['M_c'])
        dl_true = float(ev['d_L'])
        snr     = float(ev['snr'])

        z_true   = H0_TRUE * dl_true / C_KM_S
        mc_det   = mc_src * (1.0 + z_true)

        # Approximate PE uncertainties (rough SNR scaling)
        sigma_mc = mc_det * 0.03 * (12.0 / snr)    # ~3% at SNR=12
        sigma_dl = dl_true * 0.15 * (12.0 / snr)   # ~15% at SNR=12

        welch_shift = -args.mock_bias_sigma * sigma_mc  # systematic downward shift

        for mode in MODES:
            if mode == 'norm_true':
                mc_samps = rng.normal(mc_det,                  sigma_mc,       N_s)
                dl_samps = rng.normal(dl_true,                 sigma_dl,       N_s)
            elif mode == 'norm_welch':
                # Biased: shifted mean, slightly narrow (overconfident)
                mc_samps = rng.normal(mc_det + welch_shift,    sigma_mc * 0.9, N_s)
                dl_samps = rng.normal(dl_true,                 sigma_dl * 0.9, N_s)
            else:  # alcs_welch
                # Corrected: unshifted, slightly wider (properly calibrated)
                mc_samps = rng.normal(mc_det,                  sigma_mc * 1.1, N_s)
                dl_samps = rng.normal(dl_true,                 sigma_dl * 1.1, N_s)

            w = np.ones(N_s) / N_s
            data[mode]['mc'].append(mc_samps)
            data[mode]['dl'].append(dl_samps)
            data[mode]['w'].append(w)

    return data

# ---------------------------------------------------------------------------
# REAL DATA MODE: load PE samples from population mock CSV files
# ---------------------------------------------------------------------------
def load_real_samples(catalog, results_dir):
    if not HAS_ANESTHETIC:
        raise RuntimeError("anesthetic not installed; use --mock mode")

    data = {m: {'mc': [], 'dl': [], 'w': []} for m in MODES}

    for i, ev in enumerate(catalog):
        mc_src  = float(ev['M_c'])
        dl_true = float(ev['d_L'])
        z_true  = H0_TRUE * dl_true / C_KM_S

        ev_dir = os.path.join(results_dir, str(i))
        all_modes_present = all(
            os.path.exists(os.path.join(ev_dir, f"{m}.csv")) for m in MODES)
        if not all_modes_present:
            print(f"  [skip] event {i}: missing chains")
            continue

        for mode in MODES:
            path = os.path.join(ev_dir, mode)
            s    = read_chains(path)
            w    = s.get_weights()

            # PE samples are in source-frame Mc (no z applied in injection).
            # Convert to detector-frame using the true redshift so that the
            # spectral siren formula M_c^source = M_c^det/(1+z) is consistent.
            mc_src_samps = s['M_c'].values
            mc_det_samps = mc_src_samps * (1.0 + z_true)
            dl_samps     = s['d_L'].values

            data[mode]['mc'].append(mc_det_samps)
            data[mode]['dl'].append(dl_samps)
            data[mode]['w'].append(w)

    return data

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
catalog = np.load(args.catalog, allow_pickle=True)
rng     = np.random.default_rng(42)

if args.mock:
    print(f"Mock mode: {len(catalog)} events, "
          f"Welch bias = {args.mock_bias_sigma:.1f} sigma_Mc")
    data = generate_mock_samples(catalog, rng)
else:
    print(f"Loading PE samples from {args.results}/")
    data = load_real_samples(catalog, args.results)

n_events = len(data[MODES[0]]['mc'])
print(f"Using {n_events} events")

# ---------------------------------------------------------------------------
# Compute H_0 posteriors
# ---------------------------------------------------------------------------
results = {}
for mode in MODES:
    print(f"  Computing H_0 posterior for {mode} ...", flush=True)
    logp = logposterior_h0(
        data[mode]['mc'], data[mode]['dl'], data[mode]['w'], H0_GRID)
    # Normalise
    logp -= np.max(logp)
    p     = np.exp(logp)
    p    /= np.trapz(p, H0_GRID)
    results[mode] = p

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print(f"\nH_0 true = {H0_TRUE:.1f} km/s/Mpc\n")
for mode, p in results.items():
    mean = np.trapz(H0_GRID * p, H0_GRID)
    var  = np.trapz((H0_GRID - mean)**2 * p, H0_GRID)
    std  = np.sqrt(var)
    print(f"  {mode:<15}  H_0 = {mean:.1f} ± {std:.1f} km/s/Mpc")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.save(args.out, {
    'h0_grid':  H0_GRID,
    'h0_true':  H0_TRUE,
    'results':  results,
    'modes':    MODES,
    'n_events': n_events,
}, allow_pickle=True)
print(f"\nSaved {args.out}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

colors = {'norm_true': 'C2', 'norm_welch': 'C3', 'alcs_welch': 'C0'}
labels = {
    'norm_true':  'norm (true PSD)',
    'norm_welch': 'norm (Welch PSD)',
    'alcs_welch': 'ALCS (Welch PSD)',
}
styles = {'norm_true': '--', 'norm_welch': '-', 'alcs_welch': '-'}

for mode in MODES:
    ax.plot(H0_GRID, results[mode],
            color=colors[mode], ls=styles[mode],
            lw=1.8, label=labels[mode])

ax.axvline(H0_TRUE, color='k', ls=':', lw=1.0, label=f'Truth ({H0_TRUE:.0f})')
ax.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=12)
ax.set_ylabel(r'$p(H_0\,|\,\mathrm{data})$', fontsize=12)
ax.set_xlim(args.h0_min, args.h0_max)
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
tag = 'mock' if args.mock else 'real'
ax.set_title(f'Spectral siren $H_0$ ({tag}, N={n_events} events)', fontsize=10)
fig.tight_layout()
out_pdf = args.out.replace('.npy', '.pdf')
fig.savefig(out_pdf, bbox_inches='tight')
print(f"Saved {out_pdf}")
plt.show()
