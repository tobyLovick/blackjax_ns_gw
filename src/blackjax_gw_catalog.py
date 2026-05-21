"""
Run all four noise models on a GW event loaded from a pre-built catalog .npy.
Downloads detector data at runtime; includes V1 if the event was observed by Virgo.

Usage:
    python blackjax_gw_catalog.py --event GW190521
    python blackjax_gw_catalog.py --idx 7          # uses catalog order (for SLURM arrays)

Build the catalog first:
    python build_catalog.py --pastro 0.99 --out o3_catalog.npy

Outputs (in ./results/{event_lower}/):
    {mode}.csv
    {mode}_final_state.pkl
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import tqdm
from astropy.time import Time
from anesthetic import NestedSamples
from blackjax.ns.utils import finalise

jax.config.update("jax_enable_x64", True)

from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.waveform import RippleIMRPhenomD
from custom_kernels import (
    acceptance_walk_sampler,
    create_unit_cube_functions,
    init_unit_cube_particles,
    transform_to_physical,
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
grp = parser.add_mutually_exclusive_group(required=True)
grp.add_argument('--event', type=str,
                 help='Event name, e.g. GW190521')
grp.add_argument('--idx', type=int,
                 help='Catalog index (for SLURM array jobs)')
parser.add_argument('--catalog', default='o3_catalog.npy',
                    help='Path to catalog .npy (default o3_catalog.npy)')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load catalog
# ---------------------------------------------------------------------------
catalog = np.load(args.catalog, allow_pickle=True).item()
names   = list(catalog.keys())

if args.idx is not None:
    event_name = names[args.idx]
else:
    event_name = args.event
    if event_name not in catalog:
        raise ValueError(f"{event_name} not in catalog. Available: {names}")

cfg  = catalog[event_name]
name = event_name.lower().replace('-', '_')   # safe filename prefix
gps  = cfg['gps']
gmst = Time(gps, format='gps').sidereal_time('apparent', 'greenwich').rad

print(f"\nEvent   : {event_name}")
print(f"GPS     : {gps}")
print(f"Mc range: [{cfg['mc_low']}, {cfg['mc_high']}] Msun")
print(f"Dets    : {cfg['detectors']}")
print(f"p_astro : {cfg['pastro']}")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
outdir = os.path.join('results', name)
os.makedirs(outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# Download and load detector data
# ---------------------------------------------------------------------------
_det_objects = {'H1': H1, 'L1': L1, 'V1': V1}
detectors = []
for det_name in cfg['detectors']:
    if det_name not in _det_objects:
        print(f"  WARNING: unknown detector {det_name}, skipping")
        continue
    det = _det_objects[det_name]
    print(f"  Loading {det_name} data...")
    det.load_data(gps, 2, 2, 20.0, 1024.0, psd_pad=16, tukey_alpha=0.2)
    detectors.append(det)

if not detectors:
    raise RuntimeError("No detectors loaded — cannot proceed.")

# All detectors share the same frequency grid (set by the first detector loaded)
frequencies = jnp.array(detectors[0].frequencies, dtype=jnp.float64)
for det in detectors:
    det.frequencies = frequencies
    det.data = jnp.array(det.data,  dtype=jnp.complex128)
    det.psd  = jnp.array(det.psd,   dtype=jnp.float64)
    det.mask = jnp.ones(len(frequencies), dtype=jnp.float64)

waveform = RippleIMRPhenomD(f_ref=20)
post_trigger_duration = 2
duration  = 4
epoch     = duration - post_trigger_duration

# ---------------------------------------------------------------------------
# GH quadrature constants
# ---------------------------------------------------------------------------
_gh_x_np, _gh_w_np = np.polynomial.hermite.hermgauss(20)
GH_X    = jnp.array(_gh_x_np)
GH_LOGW = jnp.log(jnp.array(_gh_w_np))

# ---------------------------------------------------------------------------
# Calibration spline constants  (Payne et al. 2021, arXiv:2104.14829)
# Sigma values from arXiv:1708.03023 (O1/O2 typical: ~5% amplitude, ~3 deg phase)
# ---------------------------------------------------------------------------
_N_CAL   = 10
_CAL_LOG_NODES = jnp.linspace(jnp.log(20.0), jnp.log(1024.0), _N_CAL)
_CAL_SPACING   = _CAL_LOG_NODES[1] - _CAL_LOG_NODES[0]
_CAL_SIGMA_A   = 0.05
_CAL_SIGMA_PHI = jnp.deg2rad(3.0)

def _cal_basis(log_freq):
    """Tent (linear-interp) basis: shape (n_freq, n_nodes)."""
    return jnp.maximum(
        0., 1. - jnp.abs(log_freq[:, None] - _CAL_LOG_NODES[None, :]) / _CAL_SPACING
    )  # (n_freq, n_nodes)

def calibration_log_marginal(h, d_tilde, psd, mask, log_freq, df):
    """
    Analytically marginalise over calibration spline nodes in the linear approx.
    Returns additive log-likelihood correction from one detector.
    Amplitude and phase blocks are independent (cross-term vanishes).
    """
    w    = mask * 2.0 * df / psd              # (n_freq,) effective weights
    r    = d_tilde - h                         # (n_freq,) complex residual
    Phi  = _cal_basis(log_freq)                # (n_freq, n_nodes)

    h2     = jnp.abs(h) ** 2                  # (n_freq,)
    wh2    = w * h2                            # (n_freq,)
    wh2Phi = wh2[:, None] * Phi               # (n_freq, n_nodes)
    G      = Phi.T @ wh2Phi                   # (n_nodes, n_nodes) — shared Fisher

    A_amp = G + jnp.eye(_N_CAL) / _CAL_SIGMA_A**2
    A_phi = G + jnp.eye(_N_CAL) / _CAL_SIGMA_PHI**2

    rh    = jnp.conj(r) * h                   # (n_freq,) complex
    wPhi  = w[:, None] * Phi                  # (n_freq, n_nodes)
    c_A   =  2.0 * (wPhi.T @ jnp.real(rh))   # (n_nodes,)
    c_phi = -2.0 * (wPhi.T @ jnp.imag(rh))   # (n_nodes,)

    def _log_marginal_block(A, c, sigma):
        sol = jnp.linalg.solve(A, c)
        return (-_N_CAL * jnp.log(sigma)
                - 0.5 * jnp.linalg.slogdet(A)[1]
                + 0.5 * c @ sol)

    return (_log_marginal_block(A_amp, c_A,   _CAL_SIGMA_A)
          + _log_marginal_block(A_phi, c_phi, _CAL_SIGMA_PHI))

DELTA     = 4e-38
LOG_DELTA = jnp.log(DELTA)

# ---------------------------------------------------------------------------
# Per-bin likelihood kernels  (identical to blackjax_gw_all.py)
# ---------------------------------------------------------------------------
def _norm_single_bin(B_i, log_S_i, log_two_df_over_pi):
    return log_two_df_over_pi - log_S_i - B_i * jnp.exp(-log_S_i)

def _alcs_single_bin(B_i, log_S_i, tau, log_two_df_over_pi):
    tau_sq = tau ** 2
    s0 = 0.5 * log_S_i
    s  = s0
    for _ in range(5):
        e2s = jnp.exp(-2.0 * s)
        f   = -2.0 + 2.0*B_i*e2s - (s - s0)/tau_sq
        fp  = -4.0*B_i*e2s - 1.0/tau_sq
        s   = s - f/fp
    H     = 4.0*B_i*jnp.exp(-2.0*s) + 1.0/tau_sq
    scale = jnp.sqrt(2.0 / H)
    s_nodes = s + scale * GH_X
    g_nodes = (-2.0*s_nodes - B_i*jnp.exp(-2.0*s_nodes)
               - (s_nodes - s0)**2 / (2.0*tau_sq))
    log_integral = jnp.log(scale) + jax.scipy.special.logsumexp(GH_LOGW + g_nodes + GH_X**2)
    return log_two_df_over_pi - 0.5*jnp.log(2.0*jnp.pi*tau_sq) + log_integral

def _bad_combine(log_Z_i, p_anom):
    return jnp.logaddexp(jnp.log1p(-p_anom) + log_Z_i, jnp.log(p_anom) - LOG_DELTA)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def get_ravel_order(particles_dict):
    test_dict = {key: float(i) for i, key in enumerate(particles_dict.keys())}
    test_flat, _ = jax.flatten_util.ravel_pytree(test_dict)
    order = []
    for val in test_flat:
        for key, test_val in test_dict.items():
            if abs(val - test_val) < 1e-10:
                order.append(key)
                break
    return order

# ---------------------------------------------------------------------------
# Run one mode
# ---------------------------------------------------------------------------
def run_one_mode(mode):
    use_alcs = 'alcs' in mode
    use_bad  = 'bad'  in mode
    use_cal  = 'cal'  in mode and not use_alcs and not use_bad

    base_keys   = ["M_c","q","s1_z","s2_z","iota","d_L","t_c","psi","ra","dec","phase_c"]
    sample_keys = (base_keys
                   + (["tau"]    if use_alcs else [])
                   + (["p_anom"] if use_bad  else []))

    test_particles = {key: jax.random.uniform(jax.random.PRNGKey(42), (100,))
                      for key in sample_keys}
    sample_keys = get_ravel_order(test_particles)

    param_config = {
        "M_c":     {"min": cfg['mc_low'],  "max": cfg['mc_high'], "prior": "uniform",     "wraparound": False},
        "q":       {"min": 0.125,          "max": 1.0,            "prior": "uniform",     "wraparound": False},
        "s1_z":    {"min": -1.0,           "max": 1.0,            "prior": "uniform",     "wraparound": False},
        "s2_z":    {"min": -1.0,           "max": 1.0,            "prior": "uniform",     "wraparound": False},
        "iota":    {"min": 0.0,            "max": jnp.pi,         "prior": "sine",        "wraparound": False},
        "d_L":     {"min": 100.0,          "max": 10000.0,        "prior": "powerlaw",    "wraparound": False},
        "t_c":     {"min": -0.1,           "max": 0.1,            "prior": "uniform",     "wraparound": False},
        "phase_c": {"min": 0.0,            "max": 2*jnp.pi,       "prior": "uniform",     "wraparound": True},
        "psi":     {"min": 0.0,            "max": jnp.pi,         "prior": "uniform",     "wraparound": True},
        "ra":      {"min": 0.0,            "max": 2*jnp.pi,       "prior": "uniform",     "wraparound": True},
        "dec":     {"min": -jnp.pi/2,      "max": jnp.pi/2,       "prior": "cosine",      "wraparound": False},
        "tau":     {"min": 0.001,          "max": 2.0,            "prior": "log_uniform", "wraparound": False},
        "p_anom":  {"min": 1e-4,           "max": 0.5,            "prior": "log_uniform", "wraparound": False},
    }

    sc = {key: param_config[key] for key in sample_keys}

    param_mins = jnp.array([sc[k]["min"] for k in sample_keys])
    param_maxs = jnp.array([sc[k]["max"] for k in sample_keys])
    param_prior_types = jnp.array([
        0 if sc[k]["prior"] == "uniform"     else
        1 if sc[k]["prior"] == "sine"        else
        2 if sc[k]["prior"] == "cosine"      else
        3 if sc[k]["prior"] == "powerlaw"    else
        4 for k in sample_keys
    ])

    @jax.jit
    def prior_transform_fn(u_params):
        u, _ = jax.flatten_util.ravel_pytree(u_params)
        x = jnp.where(param_prior_types == 0, param_mins + u*(param_maxs-param_mins),
            jnp.where(param_prior_types == 1, jnp.arccos(1 - 2*u),
            jnp.where(param_prior_types == 2, jnp.arcsin(2*u - 1),
            jnp.where(param_prior_types == 3,
                      (param_mins**3 + u*(param_maxs**3 - param_mins**3))**(1/3),
                      param_mins*(param_maxs/param_mins)**u))))
        _, unflatten = jax.flatten_util.ravel_pytree({k: 0.0 for k in sample_keys})
        return unflatten(x)

    def loglikelihood_fn(params):
        p = dict(params)
        p["gmst"] = gmst
        p["eta"]  = p["q"] / (1 + p["q"])**2
        waveform_sky = waveform(frequencies, p)
        align_time   = jnp.exp(-1j*2*jnp.pi*frequencies*(epoch+p["t_c"]))
        df = frequencies[1] - frequencies[0]
        log_two_df_over_pi = jnp.log(2.0*df/jnp.pi)
        log_freq = jnp.log(frequencies)
        log_L = 0.0
        for det in detectors:
            h    = det.fd_response(frequencies, waveform_sky, p) * align_time
            B    = 2.0*df * jnp.abs(det.data - h)**2
            logS = jnp.log(det.psd)
            if use_alcs and use_bad:
                bins = jax.vmap(lambda b, ls: _bad_combine(
                    _alcs_single_bin(b, ls, p["tau"], log_two_df_over_pi),
                    p["p_anom"]))(B, logS)
            elif use_alcs:
                bins = jax.vmap(lambda b, ls: _alcs_single_bin(
                    b, ls, p["tau"], log_two_df_over_pi))(B, logS)
            elif use_bad:
                bins = jax.vmap(lambda b, ls: _bad_combine(
                    _norm_single_bin(b, ls, log_two_df_over_pi),
                    p["p_anom"]))(B, logS)
            elif use_cal:
                bins = jax.vmap(lambda b, ls: _norm_single_bin(
                    b, ls, log_two_df_over_pi))(B, logS)
                log_L = (log_L
                         + jnp.sum(bins * det.mask)
                         + calibration_log_marginal(
                             h, det.data, det.psd, det.mask, log_freq, df))
                continue
            else:
                bins = jax.vmap(lambda b, ls: _norm_single_bin(
                    b, ls, log_two_df_over_pi))(B, logS)
            log_L = log_L + jnp.sum(bins * det.mask)
        return log_L

    n_live   = 1400
    n_delete = 700
    rng_key  = jax.random.PRNGKey(10)
    rng_key, init_key = jax.random.split(rng_key)

    example_params      = {key: 0.0 for key in sample_keys}
    unit_cube_particles = init_unit_cube_particles(init_key, example_params, n_live)

    periodic_mask = jax.tree_util.tree_map(lambda _: False, example_params)
    for key in sample_keys:
        if sc[key]["wraparound"]:
            periodic_mask[key] = True

    unit_cube_fns = create_unit_cube_functions(
        physical_loglikelihood_fn=loglikelihood_fn,
        prior_transform_fn=prior_transform_fn,
        mask_tree=periodic_mask,
    )

    nested_sampler = acceptance_walk_sampler(
        logprior_fn=unit_cube_fns['logprior_fn'],
        loglikelihood_fn=unit_cube_fns['loglikelihood_fn'],
        nlive=n_live, n_target=60, max_mcmc=5000, num_delete=n_delete,
        stepper_fn=unit_cube_fns['stepper_fn'],
    )
    state = nested_sampler.init(unit_cube_particles)

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k)
        state, dead_point = nested_sampler.step(subk, state)
        return (state, k), dead_point

    def terminate(state):
        dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
        return jnp.isfinite(dlogz) and dlogz < 0.1

    dead = []
    with tqdm.tqdm(desc=f"{event_name} {mode}", unit=" dead points") as pbar:
        while not terminate(state):
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)

    final_state = finalise(state, dead)
    physical_particles = transform_to_physical(final_state.particles, prior_transform_fn)
    logL_birth = jnp.where(jnp.isnan(final_state.loglikelihood_birth),
                            -jnp.inf, final_state.loglikelihood_birth)

    column_to_label = {
        "M_c": r"$\mathcal{M}_c$", "q": r"$q$", "d_L": r"$d_L$",
        "iota": r"$\iota$", "ra": r"$\alpha$", "dec": r"$\delta$",
        "s1_z": r"$\chi_1$", "s2_z": r"$\chi_2$", "t_c": r"$t_c$",
        "psi": r"$\psi$", "phase_c": r"$\phi_c$",
        "tau": r"$\tau$", "p_anom": r"$p$",
    }

    samples = NestedSamples(
        physical_particles, logL=final_state.loglikelihood,
        logL_birth=logL_birth, labels=column_to_label,
        logzero=jnp.nan, dtype=jnp.float64,
    )

    out = os.path.join(outdir, mode)
    samples.to_csv(f"{out}.csv")
    with open(f"{out}_final_state.pkl", "wb") as f:
        pickle.dump(final_state, f)
    print(f"  Saved {out}.csv   logZ = {samples.logZ():.2f}")

# ---------------------------------------------------------------------------
# Run all four modes in series
# ---------------------------------------------------------------------------
for mode in ['norm', 'bad', 'alcs_quad', 'alcs_bad_quad', 'cal']:
    print(f"\n{'='*60}")
    print(f"  {event_name}  |  mode = {mode}  |  dets = {[d.name for d in detectors]}")
    print(f"{'='*60}")
    run_one_mode(mode)
