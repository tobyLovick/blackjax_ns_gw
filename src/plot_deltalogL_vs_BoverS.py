"""
ΔlogL (ALCS Laplace − norm) as a function of B/S for several τ values.

Setting s0 = 0 (S = 1) is exact — ΔlogL depends only on x = B/S and τ.

At the MAP, σ̃* satisfies (numerically):
    -2 + 2x·exp(-2σ̃*) - σ̃*/τ² = 0

Then:
    Δ = -log(τ) + f(σ̃*) - ½log(H) + x
    f(σ̃) = -2σ̃ - x·exp(-2σ̃) - σ̃²/(2τ²)     [s0=0]
    H     = 4x·exp(-2σ̃*) + 1/τ²

Analytic limits:
    x → 0:  Δ → 2τ²    (ALCS shifts σ* slightly low, small gain)
    x → ∞:  Δ ≈ x      (ALCS absorbs the line, saves the full -x penalty)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

TAU_VALS = [0.05, 0.10, 0.17, 0.27, 0.50, 1.00]
X_MAX    = 25
N_X      = 2000
N_NEWTON = 20
N_GH     = 50

gh_x, gh_w = np.polynomial.hermite.hermgauss(N_GH)


def _map_and_hessian(x, tau):
    s      = np.zeros_like(x)
    tau_sq = tau ** 2
    for _ in range(N_NEWTON):
        e2s = np.exp(-2 * s)
        g   = -2 + 2 * x * e2s - s / tau_sq
        gp  = -4 * x * e2s     - 1 / tau_sq
        s  -= g / gp
    H = 4 * x * np.exp(-2 * s) + 1 / tau_sq
    return s, H


def _f(s, x, tau_sq):
    return -2 * s - x * np.exp(-2 * s) - s ** 2 / (2 * tau_sq)


def compute_delta_laplace(x, tau):
    s, H    = _map_and_hessian(x, tau)
    f_map   = _f(s, x, tau ** 2)
    return -np.log(tau) + f_map - 0.5 * np.log(H) + x


def compute_delta_quad(x, tau):
    """Exact ΔlogL via Gauss-Hermite quadrature (same formula as alcs_is_correction.py)."""
    s, H    = _map_and_hessian(x, tau)
    tau_sq  = tau ** 2
    sigma   = 1.0 / np.sqrt(H)                                         # (n_x,)
    s_pts   = s[:, None] + np.sqrt(2) * sigma[:, None] * gh_x[None, :]  # (n_x, n_gh)
    f_pts   = _f(s_pts,  x[:, None], tau_sq)
    f_map   = _f(s,      x,          tau_sq)
    log_int = f_pts - f_map[:, None] + gh_x[None, :] ** 2
    log_correction = logsumexp(log_int, b=gh_w[None, :], axis=1) - 0.5 * np.log(np.pi)
    delta_lap = -np.log(tau) + f_map - 0.5 * np.log(H) + x
    return delta_lap + log_correction


x = np.linspace(1e-6, X_MAX, N_X)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(TAU_VALS)))

for ax in axes:
    for tau, col in zip(TAU_VALS, colors):
        d_lap  = compute_delta_laplace(x, tau)
        d_quad = compute_delta_quad(x, tau)
        ax.plot(x, d_lap,  color=col, lw=1.8, ls='--', label=f'τ={tau} Laplace')
        ax.plot(x, d_quad, color=col, lw=1.8, ls='-',  label=f'τ={tau} exact')
    ax.axhline(0, color='k', lw=0.8, ls=':', alpha=0.5)
    ax.set_xlabel(r'$B/\hat S$', fontsize=12)
    ax.set_ylabel(r'$\Delta\log L$ (ALCS $-$ norm)', fontsize=12)
    ax.grid(True, alpha=0.25)

# Custom legend: one entry per tau (solid=exact, dashed=Laplace)
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=col, lw=1.8, label=f'τ = {tau}')
           for tau, col in zip(TAU_VALS, colors)]
handles += [Line2D([0], [0], color='k', lw=1.5, ls='-',  label='exact (quadrature)'),
            Line2D([0], [0], color='k', lw=1.5, ls='--', label='Laplace')]
axes[0].legend(handles=handles, fontsize=8, loc='upper left', ncol=2)

axes[0].set_ylim(-1, 20)
axes[0].set_title('Full range')

axes[1].set_xlim(0, 8)
axes[1].set_ylim(-0.15, 2.5)
axes[1].set_title('Zoomed: low B/S')

fig.suptitle(r'Per-bin $\Delta\log L$ vs residual power $B/\hat S$  (solid=exact, dashed=Laplace)', fontsize=11)
fig.tight_layout()
fig.savefig('deltalogL_vs_BoverS.pdf', dpi=150, bbox_inches='tight')
print('Saved deltalogL_vs_BoverS.pdf')

print(f"\n{'τ':>6}  {'max|Lap-exact|':>16}")
for tau in TAU_VALS:
    err = np.max(np.abs(compute_delta_laplace(x, tau) - compute_delta_quad(x, tau)))
    print(f"{tau:>6.2f}  {err:>16.6f}")
