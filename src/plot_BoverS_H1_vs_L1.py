"""
Scatter plot of mean B/S per bin: H1 vs L1, coloured by frequency.

If high-B/S bins are correlated between detectors (points on the diagonal),
the excess residual power is coherent — consistent with template mismatch,
since both detectors see the same GW signal.

If H1 and L1 elevated bins are independent (no diagonal structure),
the excess is incoherent — consistent with per-detector PSD drift or lines.
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
MASK_SUFFIX = ''

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
mean_H1 = np.nanmean(all_BoverS['H1'], axis=0)
mean_L1 = np.nanmean(all_BoverS['L1'], axis=0)

# Only bins unmasked in both detectors
both_valid = ~np.isnan(mean_H1) & ~np.isnan(mean_L1)
x = mean_H1[both_valid]
y = mean_L1[both_valid]
f = freqs[both_valid]

r = np.corrcoef(x, y)[0, 1]
print(f"Pearson r(H1 B/S, L1 B/S) over {both_valid.sum()} bins: {r:.4f}")

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: full range
ax = axes[0]
sc = ax.scatter(x, y, c=f, cmap='plasma', s=2, alpha=0.6,
                linewidths=0, rasterized=True)
lim = max(x.max(), y.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5, label='y = x')
ax.set_xlabel(r'H1 mean $B_k/\hat S_k$')
ax.set_ylabel(r'L1 mean $B_k/\hat S_k$')
ax.set_xlim(0, lim);  ax.set_ylim(0, lim)
ax.set_title(f'All unmasked bins  (r = {r:.3f})')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.colorbar(sc, ax=ax, label='Frequency [Hz]')

# Right: zoom into high-B/S region
ax = axes[1]
thresh = 3
hi = (x > thresh) | (y > thresh)
sc2 = ax.scatter(x[hi], y[hi], c=f[hi], cmap='plasma', s=8, alpha=0.8,
                 linewidths=0, rasterized=True)
lim2 = max(x[hi].max(), y[hi].max()) * 1.05
ax.plot([0, lim2], [0, lim2], 'k--', lw=0.8, alpha=0.5)
ax.set_xlabel(r'H1 mean $B_k/\hat S_k$')
ax.set_ylabel(r'L1 mean $B_k/\hat S_k$')
ax.set_title(f'Bins with B/S > {thresh} in either detector  (n={hi.sum()})')
ax.grid(True, alpha=0.2)
plt.colorbar(sc2, ax=ax, label='Frequency [Hz]')

fig.suptitle(r'H1 vs L1 residual power per bin — diagonal = coherent (template mismatch)',
             fontsize=11)
fig.tight_layout()
fig.savefig('BoverS_H1_vs_L1.pdf', dpi=150, bbox_inches='tight')
print('Saved BoverS_H1_vs_L1.pdf')
