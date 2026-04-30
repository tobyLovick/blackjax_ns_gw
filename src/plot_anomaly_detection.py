"""
Combined ALCS + Bayesian anomaly detection per frequency bin.

Per-bin likelihood:
    L_i^{combined} = (1-p) * Z_i^{ALCS} + p / Delta_i

where Z_i^{ALCS} is the ALCS Laplace marginal and 1/Delta_i is the anomaly
(uniform) likelihood.

We choose Delta_i = C * S_i / (2*df), so the anomaly threshold scales with
the local PSD. C is a dimensionless scale factor (controls overall sensitivity).

The exact per-bin posterior anomaly probability is:
    P(anomalous | data) = sigmoid(log(p/Delta_i) - log((1-p)*Z_i^ALCS))

Dominant mask approximation (for clarity): bin i is anomalous if
    log Z_i^ALCS < log(p/Delta_i) - log(1-p)
i.e. if the ALCS likelihood is lower than the anomaly floor.

We use the mean B/S across posterior samples as our per-bin summary.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
from astropy.time import Time
from anesthetic import read_chains
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.waveform import RippleIMRPhenomD
from scipy.special import expit   # sigmoid

jax.config.update("jax_enable_x64", True)

N_SAMPLES   = 50
CHAINS_FILE = 'chains/blackjaxns_alcs_gw150914_notched.csv'
MASK_SUFFIX = ''
P_ANOM      = 0.01   # prior prob any bin is anomalous; try 0.001, 0.01, 0.05
C_SCALE     = 10.0   # anomaly range = C * S_i / (2*df); larger C = less sensitive

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

def find_map_hessian(B, s0, tau):
    s, tau_sq = s0.copy(), tau**2
    for _ in range(10):
        e2s = np.exp(-2*s)
        s  -= (-2 + 2*B*e2s - (s-s0)/tau_sq) / (-4*B*e2s - 1/tau_sq)
    H = 4*B*np.exp(-2*s) + 1/tau_sq
    return s, H

def log_Z_alcs(B, s0, tau):
    """Log ALCS Laplace marginal per bin (fully normalised)."""
    s_map, H = find_map_hessian(B, s0, tau)
    tau_sq = tau**2
    f_map  = -2*s_map - B*np.exp(-2*s_map) - (s_map - s0)**2 / (2*tau_sq)
    # Full normalised: log(2df/pi) - log(tau) + f_map - 0.5*log(H)
    return np.log(2*df/np.pi) - np.log(tau) + f_map - 0.5*np.log(H)

# ---------------------------------------------------------------------------
samples = read_chains(CHAINS_FILE)
rows    = samples.sample(N_SAMPLES, random_state=0)

all_logZ  = {name: np.full((N_SAMPLES, len(freqs)), np.nan) for name in dets}
all_BoverS = {name: np.full((N_SAMPLES, len(freqs)), np.nan) for name in dets}

for i, (_, row) in enumerate(rows.iterrows()):
    p = {k: float(row[k]) for k in
         ['M_c','q','s1_z','s2_z','iota','d_L','t_c','phase_c','psi','ra','dec']}
    p['gmst'] = gmst;  p['eta'] = p['q']/(1+p['q'])**2
    tau = float(row['tau'])

    align_time   = jnp.exp(-1j*2*jnp.pi*freqs_jax*(epoch+p['t_c']))
    waveform_sky = waveform(freqs_jax, p)

    for name, det in dets.items():
        h      = np.array(det['obj'].fd_response(freqs_jax, waveform_sky, p)*align_time)
        B      = 2*df*np.abs(det['d'] - h)**2
        s0     = 0.5*np.log(det['S'])
        logZ   = log_Z_alcs(B, s0, tau)
        BoverS = B / det['S']
        logZ[~det['mask']]   = np.nan
        BoverS[~det['mask']] = np.nan
        all_logZ[name][i]    = logZ
        all_BoverS[name][i]  = BoverS

    if (i+1) % 10 == 0:
        print(f"  {i+1}/{N_SAMPLES}")

# ---------------------------------------------------------------------------
# Per-bin anomaly probability:
#   log_anomaly_floor = log(p) - log(Delta_i) = log(p) - log(C * S_i / (2*df))
#                     = log(p * 2*df / (C * S_i))
# P(anomalous) = sigmoid(log_anomaly_floor - log(1-p) - log_Z_alcs)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, (name, det) in zip(axes, dets.items()):
    mean_logZ  = np.nanmean(all_logZ[name], axis=0)
    valid = ~np.isnan(mean_logZ)

    log_odds = np.log(P_ANOM * 2*df / (C_SCALE * det['S'])) - np.log(1-P_ANOM) - mean_logZ
    p_anom   = expit(log_odds)
    p_anom[~valid] = np.nan

    flagged = valid & (p_anom > 0.5)
    print(f"{name}: {flagged.sum()} bins flagged as anomalous (P > 0.5)  "
          f"out of {valid.sum()} unmasked")

    ax.scatter(freqs[valid], p_anom[valid], s=1, color='steelblue',
               alpha=0.4, linewidths=0, rasterized=True)
    ax.scatter(freqs[flagged], p_anom[flagged], s=4, color='red',
               linewidths=0, zorder=3, label=f'{flagged.sum()} flagged')
    ax.axhline(0.5, color='k', lw=0.8, ls='--', label='P=0.5 threshold')
    ax.set_ylabel(r'$P(\mathrm{anomalous} | \mathrm{data})$')
    ax.set_title(f'{name}   p={P_ANOM}, C={C_SCALE}')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('Frequency [Hz]')
axes[-1].set_xlim(freqs[0], freqs[-1])

fig.suptitle(rf'Per-bin anomaly probability $P(\varepsilon_i=0|\mathrm{{data}})$,  '
             rf'$p={P_ANOM}$, $C={C_SCALE}$,  GW150914 t20 notch', fontsize=11)
fig.tight_layout()
fig.savefig('per_bin_anomaly.pdf', dpi=150, bbox_inches='tight')
print('Saved per_bin_anomaly.pdf')
