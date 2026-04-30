"""
Per-bin B/S = 2*df*|d_k - h_k|^2 / S_k as a function of frequency,
and as a histogram vs the Exp(1) null.

Uses the same chain and setup as plot_per_bin_deltalogL.py.
Under perfectly estimated PSD and true waveform, B/S ~ Exp(1) per bin.
Excess at high B/S reveals bins where residual power exceeds the PSD.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
from astropy.time import Time
from anesthetic import read_chains
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.waveform import RippleIMRPhenomD

jax.config.update("jax_enable_x64", True)

N_SAMPLES   = 50
CHAINS_FILE = 'chains/blackjaxns_alcs_gw150914_notched.csv'
MASK_SUFFIX = ''   # t20 notch

# ---------------------------------------------------------------------------
freqs     = np.load('gw150914_frequencies.npy').astype(np.float64)
df        = float(freqs[1] - freqs[0])
freqs_jax = jnp.array(freqs)

dets = {
    'H1': {'d':    np.load('gw150914_H1_strain.npy').astype(complex),
           'S':    np.load('gw150914_H1_psd.npy').astype(np.float64),
           'mask': np.load(f'gw150914_notch_mask_H1{MASK_SUFFIX}.npy').astype(bool),
           'obj':  H1},
    'L1': {'d':    np.load('gw150914_L1_strain.npy').astype(complex),
           'S':    np.load('gw150914_L1_psd.npy').astype(np.float64),
           'mask': np.load(f'gw150914_notch_mask_L1{MASK_SUFFIX}.npy').astype(bool),
           'obj':  L1},
}
for name, det in dets.items():
    det['obj'].frequencies = freqs_jax
    det['obj'].data = jnp.array(det['d'], dtype=jnp.complex128)
    det['obj'].psd  = jnp.array(det['S'])

waveform = RippleIMRPhenomD(f_ref=20)
epoch = 4 - 2
gmst  = Time(1126259462.4, format='gps').sidereal_time('apparent', 'greenwich').rad

# ---------------------------------------------------------------------------
samples = read_chains(CHAINS_FILE)
rows    = samples.sample(N_SAMPLES, random_state=0)

all_BoverS = {name: np.full((N_SAMPLES, len(freqs)), np.nan) for name in dets}

for i, (_, row) in enumerate(rows.iterrows()):
    p = {k: float(row[k]) for k in
         ['M_c','q','s1_z','s2_z','iota','d_L','t_c','phase_c','psi','ra','dec']}
    p['gmst'] = gmst;  p['eta'] = p['q']/(1+p['q'])**2

    align_time   = jnp.exp(-1j*2*jnp.pi*freqs_jax*(epoch+p['t_c']))
    waveform_sky = waveform(freqs_jax, p)

    for name, det in dets.items():
        h      = np.array(det['obj'].fd_response(freqs_jax, waveform_sky, p)*align_time)
        BoverS = 2*df*np.abs(det['d'] - h)**2 / det['S']
        BoverS[~det['mask']] = np.nan
        all_BoverS[name][i]  = BoverS

    if (i+1) % 10 == 0:
        print(f"  {i+1}/{N_SAMPLES}")

# ---------------------------------------------------------------------------
# Figure 1: B/S vs frequency
# ---------------------------------------------------------------------------
fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, (name, det) in zip(axes1, dets.items()):
    bs_arr  = all_BoverS[name]
    mean_bs = np.nanmean(bs_arr, axis=0)

    for i in range(N_SAMPLES):
        valid = ~np.isnan(bs_arr[i])
        ax.scatter(freqs[valid], bs_arr[i][valid],
                   s=0.3, color='steelblue', alpha=0.08, linewidths=0, rasterized=True)

    valid = ~np.isnan(mean_bs)
    ax.scatter(freqs[valid], mean_bs[valid], s=1.5, color='red',
               linewidths=0, label='mean $B/S$', zorder=3)

    ax.axhline(1, color='k', lw=0.8, ls='--', label='Exp(1) mean = 1')
    ax.set_ylabel(r'$B_k / \hat S_k$')
    ax.set_title(f'{name}   mean B/S = {np.nanmean(mean_bs[valid]):.3f}   (null expectation = 1.0)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 15)
    ax.grid(True, alpha=0.2)

axes1[-1].set_xlabel('Frequency [Hz]')
axes1[-1].set_xlim(freqs[0], freqs[-1])
fig1.suptitle(r'Per-bin residual power $B_k/\hat S_k$,  GW150914 t20 notch', fontsize=11)
fig1.tight_layout()
fig1.savefig('per_bin_BoverS_freq.pdf', dpi=200, bbox_inches='tight')
print('Saved per_bin_BoverS_freq.pdf')

# ---------------------------------------------------------------------------
# Figure 2: histogram vs Exp(1)
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

x_exp    = np.linspace(0, 15, 500)
exp1_pdf = np.exp(-x_exp)

for ax, (name, det) in zip(axes2, dets.items()):
    mean_bs = np.nanmean(all_BoverS[name], axis=0)
    valid   = ~np.isnan(mean_bs)

    n_above = (mean_bs[valid] > 5).sum()
    ax.hist(mean_bs[valid], bins=80, density=True, color='steelblue',
            alpha=0.7, label=f'{valid.sum()} unmasked bins')
    ax.plot(x_exp, exp1_pdf, 'k-', lw=1.8, label='Exp(1) null')
    ax.axvline(1, color='k', lw=0.8, ls='--', alpha=0.5)

    ax.set_yscale('log')
    ax.set_xlabel(r'mean $B_k / \hat S_k$')
    ax.set_ylabel('density')
    ax.set_xlim(0, 15)
    ax.set_ylim(1e-4, 2)
    ax.set_title(f'{name}   mean={np.nanmean(mean_bs[valid]):.3f},  {n_above} bins with B/S > 5')
    ax.legend(fontsize=9)

fig2.suptitle(r'Distribution of $B_k/\hat S_k$ vs Exp(1) null,  GW150914 t20 notch', fontsize=11)
fig2.tight_layout()
fig2.savefig('per_bin_BoverS_hist.pdf', dpi=150, bbox_inches='tight')
print('Saved per_bin_BoverS_hist.pdf')
