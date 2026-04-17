"""Visualise GW150914 frequency-domain data, PSDs, and spectral line flags."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import jax
import jax.numpy as jnp
from astropy.time import Time
from anesthetic import read_chains
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.waveform import RippleIMRPhenomD

jax.config.update("jax_enable_x64", True)

freqs  = np.load('gw150914_frequencies.npy')
df     = freqs[1] - freqs[0]

dets = {
    'H1': {
        'd':    np.load('gw150914_H1_strain.npy'),
        'S':    np.load('gw150914_H1_psd.npy'),
        'mask': np.load('gw150914_notch_mask_H1.npy'),
        'color': '#1f77b4',
    },
    'L1': {
        'd':    np.load('gw150914_L1_strain.npy'),
        'S':    np.load('gw150914_L1_psd.npy'),
        'mask': np.load('gw150914_notch_mask_L1.npy'),
        'color': '#ff7f0e',
    },
}

# ---------------------------------------------------------------------------
# Figure 1: ASD — data vs PSD noise curve
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

for ax, (name, det) in zip(axes, dets.items()):
    d, S, mask = det['d'], det['S'], det['mask']
    c = det['color']

    # PSD noise curve
    ax.loglog(freqs, np.sqrt(S), color=c, lw=1.5, label=f'{name} ASD (PSD)')

    # Raw data amplitude spectrum in same units: sqrt(2*df)*|d|
    data_asd = np.sqrt(2 * df) * np.abs(d)
    ax.loglog(freqs, data_asd, color=c, alpha=0.4, lw=0.5, label=f'{name} data')

    # Mark notched bins
    notched_freqs = freqs[mask == 0]
    notched_asd   = np.sqrt(S[mask == 0])
    ax.scatter(notched_freqs, notched_asd * 3, marker='v', s=8,
               color='red', zorder=5, label='notched bins')

    ax.set_ylabel('ASD [strain / √Hz]')
    ax.set_ylim(1e-24, 1e-20)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title(f'GW150914 — {name}')

axes[-1].set_xlabel('Frequency [Hz]')
axes[-1].set_xlim(20, 1024)
fig.tight_layout()
fig.savefig('gw150914_asd.pdf', dpi=150, bbox_inches='tight')
print('Saved gw150914_asd.pdf')

# ---------------------------------------------------------------------------
# Figure 2: Normalised data power B_k = 2*df*|d|^2/S (expect ~1 for noise)
# Spectral lines jump far above 1; notched bins highlighted
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

THRESHOLD = 20.0

for ax, (name, det) in zip(axes, dets.items()):
    d, S, mask = det['d'], det['S'], det['mask']
    c = det['color']

    B = 2 * df * np.abs(d)**2 / S

    kept    = mask == 1
    notched = mask == 0

    ax.semilogy(freqs[kept],    B[kept],    '.', ms=1.5, color=c, alpha=0.4, label='kept')
    ax.semilogy(freqs[notched], B[notched], 'r.', ms=4,  label='notched')
    ax.axhline(THRESHOLD, color='k', lw=1, linestyle='--', label=f'threshold = {THRESHOLD}')
    ax.axhline(1.0, color='grey', lw=0.8, linestyle=':', label='expected noise floor')

    ax.set_ylabel(r'$2\Delta f\,|d|^2 / S_0$')
    ax.set_ylim(1e-2, None)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title(f'GW150914 — {name} — normalised data power')

axes[-1].set_xlabel('Frequency [Hz]')
axes[-1].set_xlim(20, 1024)
fig.tight_layout()
fig.savefig('gw150914_normalised_power.pdf', dpi=150, bbox_inches='tight')
print('Saved gw150914_normalised_power.pdf')

# ---------------------------------------------------------------------------
# Figure 3 & 4: Residuals d(f) - h(f)
# Draw one posterior sample for the template
# ---------------------------------------------------------------------------
waveform = RippleIMRPhenomD(f_ref=20)
post_trigger_duration = 2
epoch = 4 - post_trigger_duration
gps   = 1126259462.4
gmst  = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad

samples = read_chains('blackjaxns_alcs_gw150914_notched.csv')
row = samples.sample(1).iloc[0]
p = {
    'M_c': float(row['M_c']), 'q': float(row['q']),
    's1_z': float(row['s1_z']), 's2_z': float(row['s2_z']),
    'iota': float(row['iota']), 'd_L': float(row['d_L']),
    't_c': float(row['t_c']), 'phase_c': float(row['phase_c']),
    'psi': float(row['psi']), 'ra': float(row['ra']), 'dec': float(row['dec']),
    'gmst': gmst,
    'eta': float(row['q']) / (1 + float(row['q']))**2,
}
print(f"Template params: Mc={p['M_c']:.1f} q={p['q']:.2f} dL={p['d_L']:.0f} Mpc")

freqs_jax  = jnp.array(freqs, dtype=jnp.float64)
align_time = jnp.exp(-1j * 2 * jnp.pi * freqs_jax * (epoch + p['t_c']))
waveform_sky = waveform(freqs_jax, p)

for jimdet, name in zip([H1, L1], ['H1', 'L1']):
    jimdet.frequencies = freqs_jax
    jimdet.data = jnp.array(dets[name]['d'], dtype=jnp.complex128)
    jimdet.psd  = jnp.array(dets[name]['S'], dtype=jnp.float64)
    h = np.array(jimdet.fd_response(freqs_jax, waveform_sky, p) * align_time)
    dets[name]['h'] = h
    dets[name]['r'] = np.array(dets[name]['d'], dtype=complex) - h

# Figure 3: ASD of residuals vs PSD
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, (name, det) in zip(axes, dets.items()):
    d, r, S, mask = det['d'], det['r'], det['S'], det['mask']
    h = det['h']
    c = det['color']

    ax.loglog(freqs, np.sqrt(S), color='k', lw=1.2, alpha=0.6, label='PSD')
    ax.loglog(freqs, np.sqrt(2*df)*np.abs(d), color=c, lw=0.5,
              alpha=0.4, label='data')
    smooth_r = median_filter(np.abs(r)**2, size=50)
    ax.loglog(freqs, np.sqrt(2*df*smooth_r), color=c, lw=1.5,
              label='residual d−h (smoothed)')
    ax.loglog(freqs, np.sqrt(2*df)*np.abs(h), color='green', lw=1.0,
              alpha=0.8, label='template h(f)')

    notched_freqs = freqs[mask == 0]
    ax.scatter(notched_freqs, np.sqrt(S[mask==0]) * 3, marker='v', s=8,
               color='red', zorder=5, label='notched')

    ax.set_ylabel('ASD [strain / √Hz]')
    ax.set_ylim(1e-24, 1e-20)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title(f'GW150914 — {name} — residuals')

axes[-1].set_xlabel('Frequency [Hz]')
axes[-1].set_xlim(20, 1024)
fig.tight_layout()
fig.savefig('gw150914_residuals_asd.pdf', dpi=150, bbox_inches='tight')
print('Saved gw150914_residuals_asd.pdf')

# Figure 4: Normalised residual power 2*df*|d-h|^2/S (signal region drops to ~1)
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, (name, det) in zip(axes, dets.items()):
    r, S, mask = det['r'], det['S'], det['mask']
    c = det['color']

    B_data = 2 * df * np.abs(np.array(det['d'], dtype=complex))**2 / S
    B_res  = 2 * df * np.abs(r)**2 / S

    kept    = mask == 1
    notched = mask == 0

    ax.semilogy(freqs[kept], B_data[kept], '.', ms=1.5, color='grey',
                alpha=0.3, label='data B_k')
    ax.semilogy(freqs[kept], B_res[kept],  '.', ms=1.5, color=c,
                alpha=0.5, label='residual B_k')
    ax.semilogy(freqs[notched], B_res[notched], 'r.', ms=4, label='notched')
    ax.axhline(THRESHOLD, color='k', lw=1, linestyle='--',
               label=f'threshold = {THRESHOLD}')
    ax.axhline(1.0, color='grey', lw=0.8, linestyle=':',
               label='noise floor')

    ax.set_ylabel(r'$2\Delta f\,|{\cdot}|^2 / S_0$')
    ax.set_ylim(1e-2, None)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title(f'GW150914 — {name} — normalised residual power')

axes[-1].set_xlabel('Frequency [Hz]')
axes[-1].set_xlim(20, 1024)
fig.tight_layout()
fig.savefig('gw150914_normalised_residuals.pdf', dpi=150, bbox_inches='tight')
print('Saved gw150914_normalised_residuals.pdf')

plt.show()
