"""Plot prior draws for the calibration spline (amplitude and phase)."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# --- Spline setup (matches blackjax_gw150914_cal.py) ---
N_NODES   = 5
F_MIN, F_MAX = 20.0, 1024.0
SIGMA_A   = 0.05          # 5%
SIGMA_PHI = np.deg2rad(3.0)  # 3 degrees

log_nodes = np.linspace(np.log(F_MIN), np.log(F_MAX), N_NODES)
node_freqs = np.exp(log_nodes)

# Bilby cubic spline coefficient matrix (LIGO-T2300140)
def build_spline_matrix(n):
    tmp1 = np.zeros((n, n))
    tmp1[0,  0] = -1; tmp1[0,  1] =  2; tmp1[0,  2] = -1
    tmp1[-1,-3] = -1; tmp1[-1,-2] =  2; tmp1[-1,-1] = -1
    tmp2 = np.zeros((n, n))
    for i in range(1, n - 1):
        tmp1[i, i-1] = 1/6; tmp1[i, i] = 2/3; tmp1[i, i+1] = 1/6
        tmp2[i, i-1] = 1.0; tmp2[i, i] = -2.0; tmp2[i, i+1] = 1.0
    return solve(tmp1, tmp2)

SPLINE_COEFF = build_spline_matrix(N_NODES)
spacing = log_nodes[1] - log_nodes[0]

def eval_cubic_spline(log_freq, node_vals):
    m   = SPLINE_COEFF @ node_vals
    idx = np.clip(np.floor((log_freq - log_nodes[0]) / spacing).astype(int),
                  0, N_NODES - 2)
    h   = spacing
    dt  = log_freq - log_nodes[idx]
    y0, y1 = node_vals[idx], node_vals[idx + 1]
    m0, m1 = m[idx], m[idx + 1]
    b = (y1 - y0) / h - h * (2*m0 + m1) / 6
    return y0 + b*dt + (m0/2)*dt**2 + ((m1-m0)/(6*h))*dt**3

# --- Frequency grid ---
freq = np.logspace(np.log10(F_MIN), np.log10(F_MAX), 500)
log_freq = np.log(freq)

# --- Draw prior samples ---
N_SAMPLES = 5000
rng = np.random.default_rng(42)

amp_curves = np.zeros((N_SAMPLES, len(freq)))
phi_curves = np.zeros((N_SAMPLES, len(freq)))

for i in range(N_SAMPLES):
    dA  = rng.normal(0, SIGMA_A,   size=N_NODES)
    dph = rng.normal(0, SIGMA_PHI, size=N_NODES)
    amp_curves[i] = eval_cubic_spline(log_freq, dA)
    phi_curves[i] = eval_cubic_spline(log_freq, dph)

amp_pct = amp_curves * 100          # convert to %
phi_deg = np.rad2deg(phi_curves)    # convert to degrees

# --- Statistics ---
amp_mean = amp_pct.mean(axis=0)
amp_lo   = np.percentile(amp_pct, 5,  axis=0)
amp_hi   = np.percentile(amp_pct, 95, axis=0)

phi_mean = phi_deg.mean(axis=0)
phi_lo   = np.percentile(phi_deg, 5,  axis=0)
phi_hi   = np.percentile(phi_deg, 95, axis=0)

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
fig.subplots_adjust(hspace=0.08)

col = '#555555'

ax1.fill_between(freq, amp_lo, amp_hi, alpha=0.25, color=col, label='90% prior')
ax1.plot(freq, amp_mean, color=col, lw=1.5, label='Prior mean')
for f in node_freqs:
    ax1.axvline(f, color='k', lw=0.8, alpha=0.5)
ax1.axhline(0, color='k', lw=0.5, ls='--')
ax1.set_ylabel('Amplitude [%]')
ax1.set_ylim(-15, 15)
ax1.legend(frameon=False, fontsize=9)
ax1.set_xscale('log')

ax2.fill_between(freq, phi_lo, phi_hi, alpha=0.25, color=col)
ax2.plot(freq, phi_mean, color=col, lw=1.5)
for f in node_freqs:
    ax2.axvline(f, color='k', lw=0.8, alpha=0.5)
ax2.axhline(0, color='k', lw=0.5, ls='--')
ax2.set_ylabel('Phase [deg]')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylim(-10, 10)
ax2.set_xscale('log')

fig.suptitle('Calibration prior  ($N=5$ nodes, $\\sigma_A=5\\%$, $\\sigma_\\phi=3°$)',
             fontsize=10)

out = 'cal_prior.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved {out}')
plt.show()
