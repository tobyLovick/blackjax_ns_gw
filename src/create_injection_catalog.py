"""
Create a synthetic BBH injection catalog for the Welch PSD population study.

Mass distribution: Power Law + Peak (GWTC-3, Abbott et al. 2023 population paper)
  m1 ~ (1-lambda)*PowerLaw(alpha=3.5, [m_min, m_max]) + lambda*Normal(mu=34, sigma=4)
  q  ~ PowerLaw(beta=1.1, [m_min/m1, 1.0])
Distance: uniform in comoving volume (p(d_L) ~ d_L^2) to d_L_max
Orientation / sky: isotropic
Spins: zero (non-spinning; IMRPhenomD only)

Saves to --out as a .npy list of dicts, each containing injection parameters
and the network optimal SNR (H1+L1).

Usage:
    python create_injection_catalog.py [--n 100] [--snr-min 8] [--seed 42]
"""

import argparse
import numpy as np
from scipy.interpolate import interp1d

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from astropy.time import Time
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.waveform import RippleIMRPhenomD

parser = argparse.ArgumentParser()
parser.add_argument('--n',       type=int,   default=100)
parser.add_argument('--snr-min', type=float, default=8.0)
parser.add_argument('--seed',    type=int,   default=42)
parser.add_argument('--out',     default='injection_catalog.npy')
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------------------
# ASD and frequency grid
# ---------------------------------------------------------------------------
F_MIN       = 20.0
F_MAX       = 1024.0
DURATION    = 4
SAMPLE_RATE = 4096
F_REF       = 20.0

n_samples = DURATION * SAMPLE_RATE
delta_t   = 1.0 / SAMPLE_RATE
delta_f   = 1.0 / DURATION

freq_full = np.fft.rfftfreq(n_samples, d=delta_t)
mask      = (freq_full >= F_MIN) & (freq_full <= F_MAX)
freq      = freq_full[mask]

asd_data  = np.loadtxt('aLIGO_O4_high_asd.txt')
asd_interp = interp1d(asd_data[:, 0], asd_data[:, 1],
                      kind='linear', fill_value='extrapolate')
psd       = asd_interp(freq) ** 2

freq_jnp  = jnp.array(freq)
psd_jnp   = jnp.array(psd)

# ---------------------------------------------------------------------------
# Waveform and detectors
# ---------------------------------------------------------------------------
DETECTORS = [H1, L1]
waveform  = RippleIMRPhenomD(f_ref=F_REF)
epoch     = DURATION - 2   # post_trigger_duration = 2s

# ---------------------------------------------------------------------------
# Power Law + Peak mass model (GWTC-3 population, Abbott et al. 2023)
# ---------------------------------------------------------------------------
ALPHA    = 3.5    # power-law slope for m1
M_MIN    = 5.0    # minimum BH mass [Msun]
M_MAX    = 87.0   # maximum BH mass [Msun]
MU_M     = 34.0   # Gaussian peak location [Msun]
SIGMA_M  = 4.0    # Gaussian peak width [Msun]
LAMBDA   = 0.04   # fraction in Gaussian peak
BETA_Q   = 1.1    # mass-ratio power-law slope
Q_MIN    = 0.2    # global minimum mass ratio

DL_MAX   = 5000.0  # Mpc — upper limit for volume sampling

def sample_m1(n):
    """Sample m1 from Power Law + Peak via rejection sampling."""
    samples = []
    while len(samples) < n:
        batch = 10 * n
        # Choose component
        use_peak = rng.random(batch) < LAMBDA
        # Power-law component: p(m) ~ m^{-alpha}, sampled via inverse CDF
        u = rng.random(batch)
        exp = 1 - ALPHA
        m_pl = (M_MIN**exp + u * (M_MAX**exp - M_MIN**exp)) ** (1/exp)
        # Gaussian component
        m_gauss = rng.normal(MU_M, SIGMA_M, batch)
        m1 = np.where(use_peak, m_gauss, m_pl)
        valid = (m1 >= M_MIN) & (m1 <= M_MAX)
        samples.extend(m1[valid].tolist())
    return np.array(samples[:n])

def sample_q(m1_vals):
    """Sample q ~ q^beta_q on [max(m_min/m1, q_min), 1] per event."""
    qs = np.zeros(len(m1_vals))
    for i, m1 in enumerate(m1_vals):
        q_lo = max(M_MIN / m1, Q_MIN)
        if q_lo >= 1.0:
            qs[i] = 1.0
            continue
        # Inverse CDF of q^beta_q on [q_lo, 1]
        exp = BETA_Q + 1
        u = rng.random()
        qs[i] = (q_lo**exp + u * (1.0**exp - q_lo**exp)) ** (1/exp)
    return qs

def sample_dl(n):
    """Uniform in comoving volume: p(d_L) ~ d_L^2."""
    u = rng.random(n)
    return DL_MAX * u ** (1/3)

def network_snr(params):
    """Compute H1+L1 network optimal SNR at given params."""
    gmst = float(params['gmst'])
    p = dict(params)
    p['eta'] = p['q'] / (1 + p['q'])**2
    align = jnp.exp(-1j * 2 * jnp.pi * freq_jnp * (epoch + p['t_c']))
    ws = waveform(freq_jnp, p)
    snr_sq = 0.0
    for det in DETECTORS:
        h = np.array(det.fd_response(freq_jnp, ws, p) * align)
        snr_sq += float(4.0 * delta_f * np.sum(np.abs(h)**2 / psd))
    return np.sqrt(snr_sq)

# ---------------------------------------------------------------------------
# GPS fiducial (sets gmst — not important for injection study)
# ---------------------------------------------------------------------------
GPS_FIDUCIAL = 1126259462.4

# ---------------------------------------------------------------------------
# Draw events until we have N accepted above SNR cut
# ---------------------------------------------------------------------------
catalog = []
n_tried = 0

print(f"Drawing catalog: target={args.n} events, SNR_min={args.snr_min}")

while len(catalog) < args.n:
    batch = max(10, (args.n - len(catalog)) * 3)
    m1s   = sample_m1(batch)
    qs    = sample_q(m1s)
    m2s   = qs * m1s
    Mcs   = (m1s * m2s)**0.6 / (m1s + m2s)**0.2
    d_Ls  = sample_dl(batch)

    # Isotropic sky and orientation
    ras      = rng.uniform(0, 2*np.pi, batch)
    decs     = np.arcsin(rng.uniform(-1, 1, batch))
    iotas    = np.arccos(rng.uniform(-1, 1, batch))
    psis     = rng.uniform(0, np.pi, batch)
    phase_cs = rng.uniform(0, 2*np.pi, batch)

    n_tried += batch

    for i in range(batch):
        if len(catalog) >= args.n:
            break

        gps  = GPS_FIDUCIAL
        gmst = Time(gps, format='gps').sidereal_time('apparent', 'greenwich').rad

        params = {
            'M_c':     float(Mcs[i]),
            'q':       float(qs[i]),
            's1_z':    0.0,
            's2_z':    0.0,
            'iota':    float(iotas[i]),
            'd_L':     float(d_Ls[i]),
            't_c':     0.0,
            'phase_c': float(phase_cs[i]),
            'psi':     float(psis[i]),
            'ra':      float(ras[i]),
            'dec':     float(decs[i]),
            'gmst':    float(gmst),
            'eta':     float(qs[i]) / (1 + float(qs[i]))**2,
        }

        snr = network_snr(params)
        if snr >= args.snr_min:
            params['snr'] = float(snr)
            params['m1']  = float(m1s[i])
            params['m2']  = float(m2s[i])
            catalog.append(params)
            if len(catalog) % 10 == 0:
                print(f"  Accepted {len(catalog)}/{args.n}  "
                      f"(efficiency {len(catalog)/n_tried*100:.1f}%)")

print(f"\nFinal catalog: {len(catalog)} events from {n_tried} trials "
      f"({len(catalog)/n_tried*100:.1f}% efficiency)")

Mcs_out = np.array([e['M_c'] for e in catalog])
qs_out  = np.array([e['q']   for e in catalog])
dLs_out = np.array([e['d_L'] for e in catalog])
snrs    = np.array([e['snr'] for e in catalog])

print(f"\nMc:  {Mcs_out.min():.1f} – {Mcs_out.max():.1f}  median={np.median(Mcs_out):.1f} Msun")
print(f"q:   {qs_out.min():.2f} – {qs_out.max():.2f}  median={np.median(qs_out):.2f}")
print(f"d_L: {dLs_out.min():.0f} – {dLs_out.max():.0f}  median={np.median(dLs_out):.0f} Mpc")
print(f"SNR: {snrs.min():.1f} – {snrs.max():.1f}  median={np.median(snrs):.1f}")

np.save(args.out, catalog, allow_pickle=True)
print(f"\nSaved {args.out}")
