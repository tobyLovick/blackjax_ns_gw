"""
Create a synthetic GW injection for the Welch PSD validation study.

Generates:
  - 4 × 4s off-source noise segments → IFFT to time domain → gwpy Welch PSD
  - 1 × 4s on-source segment: noise + IMRPhenomD signal at true_params

Saves to injection_welch/:
  frequencies.npy      float64  frequency grid (f_min..f_max)
  data_fd.npy          complex128  on-source data h(f) + n(f)
  psd_true.npy         float64  true one-sided PSD on the frequency grid
  psd_welch.npy        float64  gwpy median Welch estimate
  true_params.npy      dict     injection parameters

Usage:
    python create_injection.py [--seed 0] [--snr-target 20] [--outdir injection_welch]
"""

import argparse
import os
import numpy as np
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from astropy.time import Time
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.waveform import RippleIMRPhenomD

parser = argparse.ArgumentParser()
parser.add_argument('--seed',       type=int,   default=0)
parser.add_argument('--snr-target', type=float, default=20.0,
                    help='Target network SNR (scales d_L to match)')
parser.add_argument('--outdir',     default='injection_welch')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------------------
# Configuration matching the real analysis pipeline
# ---------------------------------------------------------------------------
F_MIN        = 20.0
F_MAX        = 1024.0
DURATION     = 4          # on-source seconds
N_WELCH_SEGS = 4          # number of off-source segments for Welch
SAMPLE_RATE  = 4096       # Hz  (Nyquist > 2 * F_MAX)
TUKEY_ALPHA  = 0.2
F_REF        = 20.0

GPS_FIDUCIAL = 1126259462.4   # GW150914 — sets gmst for antenna patterns

# ---------------------------------------------------------------------------
# Frequency grid (matches jimgw: rfftfreq of DURATION*SAMPLE_RATE samples)
# ---------------------------------------------------------------------------
n_samples = DURATION * SAMPLE_RATE
delta_t   = 1.0 / SAMPLE_RATE
delta_f   = 1.0 / DURATION

freq_full = np.fft.rfftfreq(n_samples, d=delta_t)   # 0, df, 2df, ...
mask      = (freq_full >= F_MIN) & (freq_full <= F_MAX)
freq      = freq_full[mask]
n_bins    = len(freq)

print(f"Frequency grid: {n_bins} bins, df={delta_f:.4f} Hz, "
      f"[{freq[0]:.1f}, {freq[-1]:.1f}] Hz")

# ---------------------------------------------------------------------------
# True PSD (interpolate aLIGO O4 ASD onto our frequency grid)
# ---------------------------------------------------------------------------
asd_data  = np.loadtxt('aLIGO_O4_high_asd.txt')
asd_interp = interp1d(asd_data[:, 0], asd_data[:, 1],
                      kind='linear', fill_value='extrapolate')
asd_true  = asd_interp(freq)
psd_true  = asd_true ** 2    # one-sided PSD  [strain^2 / Hz]

print(f"True PSD loaded, ASD range: "
      f"[{asd_true.min():.2e}, {asd_true.max():.2e}] strain/rtHz")

# ---------------------------------------------------------------------------
# True injection parameters
# ---------------------------------------------------------------------------
gmst = Time(GPS_FIDUCIAL, format='gps').sidereal_time('apparent', 'greenwich').rad

true_params = {
    'M_c':     28.0,
    'q':       0.8,
    's1_z':    0.0,
    's2_z':    0.0,
    'iota':    0.4,
    'd_L':     500.0,   # will be rescaled to hit snr_target
    't_c':     0.0,
    'phase_c': 0.0,
    'psi':     0.8,
    'ra':      1.375,
    'dec':     -1.211,
    'gmst':    gmst,
    'eta':     0.8 / (1 + 0.8)**2,
}

# ---------------------------------------------------------------------------
# Compute signal and scale d_L to hit target SNR (network SNR over H1+L1)
# ---------------------------------------------------------------------------
DETECTORS = [H1, L1]

waveform  = RippleIMRPhenomD(f_ref=F_REF)
freq_jnp  = jnp.array(freq)

epoch      = DURATION - 2    # post_trigger_duration = 2s
align_time = jnp.exp(-1j * 2 * jnp.pi * freq_jnp * (epoch + true_params['t_c']))

waveform_sky = waveform(freq_jnp, true_params)
h_fds = {
    det.name: np.array(det.fd_response(freq_jnp, waveform_sky, true_params) * align_time)
    for det in DETECTORS
}

# Network optimal SNR: rho_net^2 = sum_det 4*df*sum(|h_det|^2 / S)
snr_sq_net = sum(
    4.0 * delta_f * np.sum(np.abs(h_fds[det.name])**2 / psd_true)
    for det in DETECTORS
)
snr_ref = np.sqrt(float(snr_sq_net))
scale   = snr_ref / args.snr_target
true_params['d_L'] *= scale
h_fds = {name: h / scale for name, h in h_fds.items()}

print(f"Signal scaled: d_L = {true_params['d_L']:.1f} Mpc → network SNR ≈ {args.snr_target:.1f}")

# ---------------------------------------------------------------------------
# Generate on-source noise and data for each detector
# ---------------------------------------------------------------------------
noise_std = np.sqrt(psd_true / (2.0 * delta_f))

data_fds = {}
for det in DETECTORS:
    n_real = rng.standard_normal(n_bins) * noise_std / np.sqrt(2)
    n_imag = rng.standard_normal(n_bins) * noise_std / np.sqrt(2)
    data_fds[det.name] = h_fds[det.name] + (n_real + 1j * n_imag)

# ---------------------------------------------------------------------------
# Generate off-source segments, IFFT to time domain, gwpy Welch PSD
# ---------------------------------------------------------------------------
off_source_td = []
for seg in range(N_WELCH_SEGS):
    # Generate one full rfft-sized noise vector (all bins)
    n_real_full = rng.standard_normal(len(freq_full))
    n_imag_full = rng.standard_normal(len(freq_full))

    # Build full-spectrum noise — zero outside [F_MIN, F_MAX] to prevent
    # leakage from the seismic wall (ASD at <20 Hz is ~1e6× larger than
    # at 20 Hz; that power bleeds into the analysis band via the Hann window).
    asd_full = interp1d(asd_data[:, 0], asd_data[:, 1],
                        kind='linear', fill_value='extrapolate')(freq_full)
    asd_full = np.where((freq_full >= F_MIN) & (freq_full <= F_MAX), asd_full, 0.0)
    psd_full = asd_full ** 2
    std_full = np.sqrt(psd_full / (2.0 * delta_f))
    n_fd_full = (n_real_full + 1j * n_imag_full) * std_full / np.sqrt(2)

    # IFFT to time domain (irfft convention: divide by dt to get correct units)
    n_td = np.fft.irfft(n_fd_full) / delta_t
    off_source_td.append(n_td)

# Concatenate N_WELCH_SEGS * DURATION seconds of off-source time series
off_source_full = np.concatenate(off_source_td)
ts = TimeSeries(off_source_full, sample_rate=SAMPLE_RATE, t0=0.0)

# Median Welch PSD with fftlength=DURATION (matches jimgw exactly)
psd_gwpy  = ts.psd(fftlength=DURATION, method='median')
psd_welch = np.interp(freq, np.array(psd_gwpy.frequencies.value),
                      np.array(psd_gwpy.value))

print(f"Welch PSD estimated from {N_WELCH_SEGS}×{DURATION}s = "
      f"{N_WELCH_SEGS*DURATION}s off-source data")
print(f"  True PSD median:  {np.median(psd_true):.3e}")
print(f"  Welch PSD median: {np.median(psd_welch):.3e}")
print(f"  Ratio (should be ~1): {np.median(psd_welch/psd_true):.4f}")

# ---------------------------------------------------------------------------
# Save — one file per detector for data; shared PSD and params
# ---------------------------------------------------------------------------
np.save(f'{args.outdir}/frequencies.npy',  freq)
np.save(f'{args.outdir}/psd_true.npy',     psd_true)
np.save(f'{args.outdir}/psd_welch.npy',    psd_welch)
np.save(f'{args.outdir}/true_params.npy',  true_params, allow_pickle=True)
for det in DETECTORS:
    np.save(f'{args.outdir}/data_fd_{det.name}.npy', data_fds[det.name])

print(f"\nSaved to {args.outdir}/")
print(f"  frequencies.npy        {freq.shape}")
for det in DETECTORS:
    print(f"  data_fd_{det.name}.npy       {data_fds[det.name].shape}  (complex)")
print(f"  psd_true.npy           {psd_true.shape}")
print(f"  psd_welch.npy          {psd_welch.shape}")
print(f"  true_params.npy        {list(true_params.keys())}")
