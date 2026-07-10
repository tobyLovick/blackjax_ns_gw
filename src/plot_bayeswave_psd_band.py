"""
PSD with ALCS tau uncertainty band (BayesWave PSD), data overlaid coloured by anomaly probability.

Uses BayesWave median PSDs (LIGO-P1900011) in place of Welch.
Single chain provides both tau band and p_anom colouring (no separate BW-BAD-only run).

  - ALCS tau band + p_anom: blackjaxns_psdcomp_gw150914_alcs_bad_bayeswave.csv
  - Welch PSD overlaid as thin comparison line

Physical Delta: DELTA = 4e-38
"""

import os
os.chdir(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
from astropy.time import Time
from anesthetic import read_chains
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.waveform import RippleIMRPhenomD
from scipy.special import expit

jax.config.update("jax_enable_x64", True)

N_SAMPLES_TAU   = 200
N_SAMPLES_PANOM = 50

DELTA     = 4e-38
LOG_DELTA = np.log(DELTA)

ALCS_BAD_FILE = 'blackjaxns_psdcomp_gw150914_alcs_bad_bayeswave.csv'

freqs     = np.load('gw150914_frequencies.npy').astype(np.float64)
df        = float(freqs[1] - freqs[0])
freqs_jax = jnp.array(freqs)

det_data = {
    'H1': {'d': np.load('gw150914_H1_strain.npy').astype(complex),
           'S': np.load('gw150914_H1_psd_bayeswave.npy').astype(np.float64),
           'S_welch': np.load('gw150914_H1_psd.npy').astype(np.float64),
           'obj': H1},
    'L1': {'d': np.load('gw150914_L1_strain.npy').astype(complex),
           'S': np.load('gw150914_L1_psd_bayeswave.npy').astype(np.float64),
           'S_welch': np.load('gw150914_L1_psd.npy').astype(np.float64),
           'obj': L1},
}
for name, det in det_data.items():
    det['obj'].frequencies = freqs_jax
    det['obj'].data = jnp.array(det['d'], dtype=jnp.complex128)
    det['obj'].psd  = jnp.array(det['S'])

waveform = RippleIMRPhenomD(f_ref=20)
epoch = 4 - 2
gmst  = Time(1126259462.4, format='gps').sidereal_time('apparent', 'greenwich').rad

# ---------------------------------------------------------------------------
# tau band + p_anom from alcs_bad_bayeswave chain
alcs_samples = read_chains(ALCS_BAD_FILE)
w_alcs = alcs_samples.get_weights()
tau_draws = alcs_samples['tau'].values[
    np.random.choice(len(w_alcs), size=N_SAMPLES_TAU, p=w_alcs/w_alcs.sum())]
tau_mean = np.average(alcs_samples['tau'].values, weights=w_alcs)

rows = alcs_samples.sample(N_SAMPLES_PANOM, random_state=0)

def log_Z_norm(B, S):
    return np.log(2*df/np.pi) - np.log(S) - B/S

all_panom = {name: np.full((N_SAMPLES_PANOM, len(freqs)), np.nan) for name in det_data}

for i, (_, row) in enumerate(rows.iterrows()):
    p = {k: float(row[k]) for k in
         ['M_c','q','s1_z','s2_z','iota','d_L','t_c','phase_c','psi','ra','dec']}
    p['gmst'] = gmst; p['eta'] = p['q']/(1+p['q'])**2
    p_anom = float(row['p_anom'])

    align_time   = jnp.exp(-1j*2*jnp.pi*freqs_jax*(epoch+p['t_c']))
    waveform_sky = waveform(freqs_jax, p)

    for name, det in det_data.items():
        h         = np.array(det['obj'].fd_response(freqs_jax, waveform_sky, p)*align_time)
        B         = 2*df*np.abs(det['d'] - h)**2
        logZ      = log_Z_norm(B, det['S'])
        log_floor = np.log(p_anom) - LOG_DELTA
        all_panom[name][i] = expit(log_floor - np.log(1-p_anom) - logZ)

    if (i+1) % 10 == 0:
        print(f"  {i+1}/{N_SAMPLES_PANOM}")

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, (name, det) in zip(axes, det_data.items()):
    S        = det['S']
    psd_asd  = np.sqrt(S)
    data_asd = np.sqrt(2*df) * np.abs(det['d'])
    mean_panom = np.nanmean(all_panom[name], axis=0)

    # tau uncertainty band around BayesWave PSD
    bands_lo = np.array([psd_asd * np.exp(-tau) for tau in tau_draws])
    bands_hi = np.array([psd_asd * np.exp( tau) for tau in tau_draws])
    band_lo  = np.percentile(bands_lo, 16, axis=0)
    band_hi  = np.percentile(bands_hi, 84, axis=0)

    ax.fill_between(freqs, band_lo, band_hi,
                    color='grey', alpha=0.35, zorder=1,
                    label=rf'ALCS $1\sigma$ band ($\tau={tau_mean:.2f}$)')
    ax.loglog(freqs, psd_asd, color='k', lw=1.0, alpha=0.8, zorder=2, label='BayesWave PSD')
    ax.loglog(freqs, np.sqrt(det['S_welch']), color='C1', lw=0.5, alpha=0.5,
              zorder=2, label='Welch PSD')

    sc = ax.scatter(freqs, data_asd, c=mean_panom, cmap='coolwarm',
                    s=1, alpha=0.5, linewidths=0, rasterized=True,
                    vmin=0, vmax=1, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    plt.colorbar(sc, ax=ax, label=r'$P(\mathrm{anomalous})$')

    ax.set_ylabel(r'ASD [Hz$^{-1/2}$]')
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, which='both')

axes[-1].set_xlabel('Frequency [Hz]')
axes[-1].set_xlim(freqs[0], freqs[-1])
fig.suptitle(r'Strain ASD with ALCS band and anomaly probability  (BayesWave PSD)', fontsize=11)
fig.tight_layout()
fig.savefig('bayeswave_psd_band.pdf', dpi=150, bbox_inches='tight')
print('Saved bayeswave_psd_band.pdf')
