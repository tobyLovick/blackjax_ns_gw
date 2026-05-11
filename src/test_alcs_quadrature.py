"""
Benchmark: MAP-centred Gauss-Hermite quadrature vs Laplace for ALCS.

The integral is:
    log Z_i = log(2df/pi) - 0.5*log(2*pi*tau^2)
              + log( integral exp(g(s)) ds )

where g(s) = -2s - B*exp(-2s) - (s-s0)^2/(2*tau^2).

Strategy: find MAP s_hat via Newton (same as Laplace), then centre GH nodes
there with scale = sqrt(2/H). Factor as:

    integral exp(g(s)) ds = scale * sum_k w_k * exp(g(s_hat + scale*x_k) + x_k^2)

The exp(+x_k^2) re-absorbs the Gaussian weight that GH factors out.
Centring at the MAP means the nodes always span the actual peak regardless
of B/S or tau.
"""

import os
os.chdir(os.path.dirname(__file__))

import numpy as np
import jax
import jax.numpy as jnp
from scipy import integrate
import time

jax.config.update("jax_enable_x64", True)

# Pre-computed GH nodes/weights — compile-time constants
N_GH = 20
_gh_x_np, _gh_w_np = np.polynomial.hermite.hermgauss(N_GH)
GH_X    = jnp.array(_gh_x_np)
GH_LOGW = jnp.log(jnp.array(_gh_w_np))

# ---------------------------------------------------------------------------

def _alcs_laplace(B_i, log_S_i, tau, log_two_df_over_pi):
    tau_sq = tau**2
    s0 = 0.5 * log_S_i
    s  = s0
    for _ in range(5):
        e2s = jnp.exp(-2.0 * s)
        f   = -2.0 + 2.0*B_i*e2s - (s - s0)/tau_sq
        fp  = -4.0 * B_i*e2s - 1.0/tau_sq
        s   = s - f/fp
    log_f = (log_two_df_over_pi - 2.0*s - B_i*jnp.exp(-2.0*s)
             - 0.5*jnp.log(2.0*jnp.pi*tau_sq) - (s - s0)**2/(2.0*tau_sq))
    H = 4.0 + (2.0*(s - s0) + 1.0)/tau_sq
    return log_f + 0.5*jnp.log(2.0*jnp.pi/H)

def _alcs_quad(B_i, log_S_i, tau, log_two_df_over_pi):
    """MAP-centred GH quadrature. Replaces Newton+Laplace with Newton+quadrature."""
    tau_sq = tau**2
    s0 = 0.5 * log_S_i
    # Newton to find MAP
    s = s0
    for _ in range(5):
        e2s = jnp.exp(-2.0 * s)
        f   = -2.0 + 2.0*B_i*e2s - (s - s0)/tau_sq
        fp  = -4.0 * B_i*e2s - 1.0/tau_sq
        s   = s - f/fp
    s_hat = s
    H     = 4.0*B_i*jnp.exp(-2.0*s_hat) + 1.0/tau_sq
    scale = jnp.sqrt(2.0 / H)                        # node spacing

    s_nodes = s_hat + scale * GH_X                    # (N_GH,)
    g_nodes = (-2.0*s_nodes - B_i*jnp.exp(-2.0*s_nodes)
               - (s_nodes - s0)**2 / (2.0*tau_sq))    # g(s) at each node

    log_integral = jnp.log(scale) + jax.scipy.special.logsumexp(
        GH_LOGW + g_nodes + GH_X**2)                  # re-absorb exp(-x^2) weight

    return log_two_df_over_pi - 0.5*jnp.log(2.0*jnp.pi*tau_sq) + log_integral

def _alcs_scipy(B, log_S, tau, log_two_df_over_pi):
    """Ground truth via scipy.quad."""
    s0, tau_sq = 0.5*log_S, tau**2
    def integrand(s):
        return np.exp(-2*s - B*np.exp(-2*s) - (s-s0)**2/(2*tau_sq))
    val, _ = integrate.quad(integrand, s0 - 15*tau, s0 + 15*tau, limit=500)
    return float(log_two_df_over_pi - 0.5*np.log(2*np.pi*tau_sq) + np.log(val))

# ---------------------------------------------------------------------------
S         = 1e-46
df        = 0.25
log_S     = float(np.log(S))
log_2df_pi = float(np.log(2*df/np.pi))

print(f"MAP-centred GH quadrature vs Laplace  (N_GH={N_GH})")
print("Errors relative to scipy.quad ground truth\n")

for tau in [0.17, 0.52, 1.0, 2.0]:
    print(f"tau={tau}")
    print(f"  {'B/S':>8s}  {'err_laplace':>12s}  {'err_quad':>10s}")
    for BoverS in [0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 3000]:
        B = BoverS * S
        truth   = _alcs_scipy(B, log_S, tau, log_2df_pi)
        laplace = float(_alcs_laplace(B, log_S, tau, log_2df_pi))
        quad    = float(_alcs_quad(B, log_S, tau, log_2df_pi))
        print(f"  {BoverS:8.0f}  {laplace-truth:+12.5f}  {quad-truth:+10.6f}")
    print()

# ---------------------------------------------------------------------------
print("JIT compilation + speed benchmark")
print("(vmapped over 4015 bins, 1000 calls)\n")

N_BINS   = 4015
B_arr    = jnp.array(np.random.exponential(1, N_BINS) * S)
logS_arr = jnp.full(N_BINS, log_S)

laplace_v = jax.jit(jax.vmap(lambda b, ls: _alcs_laplace(b, ls, 0.52, log_2df_pi)))
quad_v    = jax.jit(jax.vmap(lambda b, ls: _alcs_quad(b, ls, 0.52, log_2df_pi)))

_ = laplace_v(B_arr, logS_arr).block_until_ready()
_ = quad_v(B_arr, logS_arr).block_until_ready()

N = 1000
t0 = time.perf_counter()
for _ in range(N): laplace_v(B_arr, logS_arr).block_until_ready()
t_lap = (time.perf_counter() - t0) / N * 1e3

t0 = time.perf_counter()
for _ in range(N): quad_v(B_arr, logS_arr).block_until_ready()
t_quad = (time.perf_counter() - t0) / N * 1e3

print(f"  Laplace  : {t_lap:.3f} ms per call")
print(f"  GH quad  : {t_quad:.3f} ms per call")
print(f"  Overhead : {t_quad/t_lap:.2f}x")
