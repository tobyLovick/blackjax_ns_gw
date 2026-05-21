"""
GW150914 parameter estimation with sampled calibration spline parameters.

Runs two models in series and saves both for evidence comparison:
  norm  -- standard fixed-PSD Whittle likelihood (11 GW params)
  cal   -- norm + calibration spline (11 + 20 = 31 params)

Calibration model matches Romero-Shaw et al. 2020 (arXiv:2006.00714) / bilby:
  h_cal(f) = h(f) * (1 + delta_A(f)) * (2 + i*delta_phi(f)) / (2 - i*delta_phi(f))

where delta_A and delta_phi are cubic splines with N_CAL_NODES nodes each,
log-spaced in frequency between f_min and f_max, with Gaussian priors:
  delta_A_j   ~ N(0, SIGMA_A^2)
  delta_phi_j ~ N(0, SIGMA_PHI^2)

Prior widths from arXiv:1708.03023 (O1/O2 LIGO calibration uncertainty paper):
  SIGMA_A   = 0.05  (5% amplitude, typical for H1/L1 O1)
  SIGMA_PHI = 0.052 rad  (~3 degrees phase)

Two detectors: H1, L1  ->  2 * 2 * N_CAL_NODES = 20 calibration parameters.
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
from astropy.time import Time
import tqdm
import pickle

jax.config.update("jax_enable_x64", True)

from jimgw.single_event.detector import H1, L1
from jimgw.single_event.waveform import RippleIMRPhenomD

from custom_kernels import (
    acceptance_walk_sampler,
    create_unit_cube_functions,
    init_unit_cube_particles,
    transform_to_physical,
)

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------
N_CAL_NODES  = 5
F_MIN, F_MAX = 20.0, 1024.0
SIGMA_A      = 0.05               # 5% amplitude (arXiv:1708.03023 O1 typical)
SIGMA_PHI    = jnp.deg2rad(3.0)   # 3 deg phase  (arXiv:1708.03023 O1 typical)

# Log-spaced nodes (matches bilby CubicSpline node placement)
_CAL_LOG_NODES_NP = np.linspace(np.log(F_MIN), np.log(F_MAX), N_CAL_NODES)
CAL_LOG_NODES = jnp.array(_CAL_LOG_NODES_NP)
CAL_SPACING   = CAL_LOG_NODES[1] - CAL_LOG_NODES[0]

# Precompute cubic spline coefficient matrix (bilby LIGO-T2300140 algorithm)
def _build_spline_matrix(n):
    tmp1 = np.zeros((n, n))
    tmp1[0, 0]  = -1; tmp1[0, 1]  =  2; tmp1[0, 2]  = -1
    tmp1[-1,-3] = -1; tmp1[-1,-2] =  2; tmp1[-1,-1] = -1
    tmp2 = np.zeros((n, n))
    for i in range(1, n - 1):
        tmp1[i, i-1] = 1/6;  tmp1[i, i] = 2/3;  tmp1[i, i+1] = 1/6
        tmp2[i, i-1] = 1.0;  tmp2[i, i] = -2.0; tmp2[i, i+1] = 1.0
    return np.linalg.solve(tmp1, tmp2)   # (n, n)

_SPLINE_COEFF_NP = _build_spline_matrix(N_CAL_NODES)
SPLINE_COEFF     = jnp.array(_SPLINE_COEFF_NP)   # nodes -> second derivatives

def _eval_cubic_spline(log_freq, node_values):
    """
    Evaluate cubic spline at log_freq given values at CAL_LOG_NODES.
    Matches bilby CubicSpline._evaluate_spline (LIGO-T2300140 Eq. 1).
    node_values: (N_CAL_NODES,)
    log_freq:    (n_freq,)
    returns:     (n_freq,)
    """
    # Second derivatives at nodes
    m = SPLINE_COEFF @ node_values   # (N_CAL_NODES,)

    # For each frequency find which interval it falls in
    # idx = clipped index of left node
    raw_idx = (log_freq - CAL_LOG_NODES[0]) / CAL_SPACING
    idx     = jnp.clip(jnp.floor(raw_idx).astype(jnp.int32), 0, N_CAL_NODES - 2)

    h  = CAL_SPACING
    t  = (log_freq - CAL_LOG_NODES[idx]) / h        # (n_freq,) in [0,1]

    y0 = node_values[idx];     y1 = node_values[idx + 1]
    m0 = m[idx];               m1 = m[idx + 1]

    # Standard cubic spline formula
    a =  y0
    b = (y1 - y0) / h - h * (2*m0 + m1) / 6
    c =  m0 / 2
    d = (m1 - m0) / (6 * h)
    dt = log_freq - CAL_LOG_NODES[idx]
    return a + b*dt + c*dt**2 + d*dt**3

def apply_calibration(h, log_freq, delta_A_nodes, delta_phi_nodes):
    """
    Apply calibration correction to template h (exact bilby formula).
    delta_A_nodes, delta_phi_nodes: (N_CAL_NODES,)
    returns: h_cal (n_freq,) complex
    """
    dA  = _eval_cubic_spline(log_freq, delta_A_nodes)
    dph = _eval_cubic_spline(log_freq, delta_phi_nodes)
    # Bilby: calibration_factor = (1 + dA) * (2 + i*dph) / (2 - i*dph)
    return h * (1.0 + dA) * (2.0 + 1j * dph) / (2.0 - 1j * dph)

# ---------------------------------------------------------------------------
# Load GW150914 data
# ---------------------------------------------------------------------------
waveform = RippleIMRPhenomD(f_ref=20)

frequencies = jnp.array(np.load('gw150914_frequencies.npy'), dtype=jnp.float64)
log_freq    = jnp.log(frequencies)

detectors = [H1, L1]
for det in detectors:
    det.frequencies = frequencies
    det.data = jnp.array(np.load(f'gw150914_{det.name}_strain.npy'), dtype=jnp.complex128)
    det.psd  = jnp.array(np.load(f'gw150914_{det.name}_psd.npy'),    dtype=jnp.float64)
    det.mask = jnp.ones(len(frequencies), dtype=jnp.float64)

post_trigger_duration = 2
duration = 4
epoch    = duration - post_trigger_duration
gps      = 1126259462.4
gmst     = Time(gps, format='gps').sidereal_time('apparent', 'greenwich').rad

# ---------------------------------------------------------------------------
# Parameter names and helpers
# ---------------------------------------------------------------------------
GW_KEYS = ["M_c","q","s1_z","s2_z","iota","d_L","t_c","psi","ra","dec","phase_c"]

# Calibration parameter names: {det}_{kind}_{node_idx}
CAL_KEYS = [f"{det.name}_{kind}_{j}"
            for det in detectors
            for kind in ("amplitude", "phase")
            for j in range(N_CAL_NODES)]

def get_ravel_order(d):
    test = {k: float(i) for i, k in enumerate(d.keys())}
    flat, _ = jax.flatten_util.ravel_pytree(test)
    order = []
    for v in flat:
        for k, tv in test.items():
            if abs(v - tv) < 1e-10:
                order.append(k); break
    return order

# ---------------------------------------------------------------------------
# Per-bin Normal likelihood kernel
# ---------------------------------------------------------------------------
def _norm_bin(B_i, log_S_i, log_two_df_over_pi):
    return log_two_df_over_pi - log_S_i - B_i * jnp.exp(-log_S_i)

_norm_bins = jax.vmap(_norm_bin, in_axes=(0, 0, None))

# ---------------------------------------------------------------------------
# Run one mode
# ---------------------------------------------------------------------------
def run_mode(mode):
    use_cal = (mode == 'cal')
    sample_keys = GW_KEYS + (CAL_KEYS if use_cal else [])

    test_p      = {k: jax.random.uniform(jax.random.PRNGKey(0), (100,)) for k in sample_keys}
    sample_keys = get_ravel_order(test_p)

    # Param config: GW params + (optionally) Gaussian cal params
    GW_CONFIG = {
        "M_c":     {"min": 25.0,      "max": 50.0,       "type": "uniform",   "wrap": False},
        "q":       {"min": 0.25,      "max": 1.0,        "type": "uniform",   "wrap": False},
        "s1_z":    {"min": -1.0,      "max": 1.0,        "type": "uniform",   "wrap": False},
        "s2_z":    {"min": -1.0,      "max": 1.0,        "type": "uniform",   "wrap": False},
        "iota":    {"min": 0.0,       "max": jnp.pi,     "type": "sine",      "wrap": False},
        "d_L":     {"min": 100.0,     "max": 5000.0,     "type": "powerlaw",  "wrap": False},
        "t_c":     {"min": -0.1,      "max": 0.1,        "type": "uniform",   "wrap": False},
        "phase_c": {"min": 0.0,       "max": 2*jnp.pi,   "type": "uniform",   "wrap": True},
        "psi":     {"min": 0.0,       "max": jnp.pi,     "type": "uniform",   "wrap": True},
        "ra":      {"min": 0.0,       "max": 2*jnp.pi,   "type": "uniform",   "wrap": True},
        "dec":     {"min": -jnp.pi/2, "max": jnp.pi/2,   "type": "cosine",    "wrap": False},
    }
    CAL_CONFIG = {k: {"sigma": SIGMA_A   if "amplitude" in k else SIGMA_PHI,
                      "type": "gaussian", "wrap": False}
                  for k in CAL_KEYS}

    cfg = {**GW_CONFIG, **(CAL_CONFIG if use_cal else {})}
    sc  = {k: cfg[k] for k in sample_keys}

    type_arr  = jnp.array([0 if sc[k]["type"]=="uniform"  else
                            1 if sc[k]["type"]=="sine"     else
                            2 if sc[k]["type"]=="cosine"   else
                            3 if sc[k]["type"]=="powerlaw" else
                            4   # gaussian
                            for k in sample_keys])
    mins      = jnp.array([sc[k].get("min", 0.) for k in sample_keys])
    maxs      = jnp.array([sc[k].get("max", 1.) for k in sample_keys])
    sigmas    = jnp.array([sc[k].get("sigma", 1.) for k in sample_keys])

    @jax.jit
    def prior_transform_fn(u_params):
        u, _ = jax.flatten_util.ravel_pytree(u_params)
        x = jnp.where(type_arr == 0, mins + u*(maxs - mins),
            jnp.where(type_arr == 1, jnp.arccos(1 - 2*u),
            jnp.where(type_arr == 2, jnp.arcsin(2*u - 1),
            jnp.where(type_arr == 3, (mins**3 + u*(maxs**3 - mins**3))**(1/3),
                      sigmas * jax.scipy.special.ndtri(u)))))   # gaussian
        _, unf = jax.flatten_util.ravel_pytree({k: 0. for k in sample_keys})
        return unf(x)

    @jax.jit
    def logprior_fn(params):
        vals, _ = jax.flatten_util.ravel_pytree(params)
        lp = jnp.where(type_arr == 0, jnp.where((vals>=mins)&(vals<=maxs), -jnp.log(maxs-mins), -jnp.inf),
             jnp.where(type_arr == 1, jnp.where((vals>=0)&(vals<=jnp.pi), jnp.log(jnp.sin(vals)/2), -jnp.inf),
             jnp.where(type_arr == 2, jnp.where(jnp.abs(vals)<jnp.pi/2, jnp.log(jnp.cos(vals)/2), -jnp.inf),
             jnp.where(type_arr == 3, 2*jnp.log(vals) - jnp.log(maxs**3 - mins**3),
                       -vals**2/(2*sigmas**2) - jnp.log(sigmas) - 0.5*jnp.log(2*jnp.pi)))))
        return jnp.sum(lp)

    def loglikelihood_fn(params):
        p = dict(params)
        p["gmst"] = gmst
        p["eta"]  = p["q"] / (1 + p["q"])**2
        waveform_sky = waveform(frequencies, p)
        align_time   = jnp.exp(-1j * 2*jnp.pi * frequencies * (epoch + p["t_c"]))
        df                 = frequencies[1] - frequencies[0]
        log_two_df_over_pi = jnp.log(2.0 * df / jnp.pi)
        log_L = 0.0
        for det in detectors:
            h = det.fd_response(frequencies, waveform_sky, p) * align_time
            if use_cal:
                dA  = jnp.array([p[f"{det.name}_amplitude_{j}"] for j in range(N_CAL_NODES)])
                dph = jnp.array([p[f"{det.name}_phase_{j}"]     for j in range(N_CAL_NODES)])
                h   = apply_calibration(h, log_freq, dA, dph)
            B     = 2.0 * df * jnp.abs(det.data - h)**2
            log_L = log_L + jnp.dot(_norm_bins(B, jnp.log(det.psd), log_two_df_over_pi), det.mask)
        return log_L

    # Nested sampling
    n_live   = 2000
    n_delete = 700
    rng_key  = jax.random.PRNGKey(10)
    rng_key, init_key = jax.random.split(rng_key)

    example_params      = {k: 0. for k in sample_keys}
    unit_cube_particles = init_unit_cube_particles(init_key, example_params, n_live)

    periodic_mask = jax.tree_util.tree_map(lambda _: False, example_params)
    for k in sample_keys:
        if sc[k]["wrap"]:
            periodic_mask[k] = True

    unit_cube_fns = create_unit_cube_functions(
        physical_loglikelihood_fn=loglikelihood_fn,
        prior_transform_fn=prior_transform_fn,
        mask_tree=periodic_mask,
    )

    nested_sampler = acceptance_walk_sampler(
        logprior_fn=unit_cube_fns['logprior_fn'],
        loglikelihood_fn=unit_cube_fns['loglikelihood_fn'],
        nlive=n_live, n_target=60, max_mcmc=8000, num_delete=n_delete,
        stepper_fn=unit_cube_fns['stepper_fn'],
    )
    state = nested_sampler.init(unit_cube_particles)

    @jax.jit
    def one_step(carry, _):
        state, k = carry
        k, subk  = jax.random.split(k)
        state, dead = nested_sampler.step(subk, state)
        return (state, k), dead

    def terminate(state):
        dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
        return jnp.isfinite(dlogz) and dlogz < 0.1

    dead = []
    with tqdm.tqdm(desc=f"GW150914 {mode}", unit=" dead points") as pbar:
        while not terminate(state):
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)

    from blackjax.ns.utils import finalise
    from anesthetic import NestedSamples

    final_state      = finalise(state, dead)
    physical_params  = transform_to_physical(final_state.particles, prior_transform_fn)
    logL_birth       = jnp.where(jnp.isnan(final_state.loglikelihood_birth),
                                  -jnp.inf, final_state.loglikelihood_birth)

    labels = {
        "M_c": r"$\mathcal{M}_c$", "q": r"$q$", "d_L": r"$d_L$",
        "iota": r"$\iota$", "ra": r"$\alpha$", "dec": r"$\delta$",
        "s1_z": r"$\chi_1$", "s2_z": r"$\chi_2$", "t_c": r"$t_c$",
        "psi": r"$\psi$", "phase_c": r"$\phi_c$",
        **{k: k for k in CAL_KEYS},
    }

    samples = NestedSamples(
        physical_params, logL=final_state.loglikelihood,
        logL_birth=logL_birth, labels=labels,
        logzero=jnp.nan, dtype=jnp.float64,
    )
    samples.to_csv(f"blackjaxns_cal_gw150914_{mode}.csv")
    with open(f"blackjaxns_cal_gw150914_{mode}_final_state.pkl", "wb") as f:
        pickle.dump(final_state, f)
    print(f"  [{mode}]  logZ = {samples.logZ():.2f}  (n_params = {len(sample_keys)})")
    return samples.logZ()

# ---------------------------------------------------------------------------
# Run both models
# ---------------------------------------------------------------------------
logZ_norm = run_mode('norm')
logZ_cal  = run_mode('cal')

print(f"\nlog Bayes factor  cal vs norm:  {logZ_cal - logZ_norm:.2f} nats")
print(f"  (positive = data prefers calibration marginalisation)")
