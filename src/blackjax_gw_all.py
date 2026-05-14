"""
Run all four noise models on a named GW event in series:
  norm        : fixed-PSD Whittle likelihood
  bad         : fixed-PSD + BAD anomaly floor
  alcs_quad   : ALCS (GH quadrature) — no anomaly
  alcs_bad_quad: ALCS (GH quadrature) + BAD

Usage:
  python blackjax_gw_all.py --event GW170814
  python blackjax_gw_all.py --event GW151226
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

from jimgw.single_event.detector import H1, L1
from jimgw.single_event.waveform import RippleIMRPhenomD
from custom_kernels import (
    acceptance_walk_sampler,
    create_unit_cube_functions,
    init_unit_cube_particles,
    transform_to_physical,
)

# ---------------------------------------------------------------------------
# Event catalogue
# ---------------------------------------------------------------------------
EVENTS = {
    'GW150914': {'gps': 1126259462.4, 'Mc_min': 20.0, 'Mc_max': 45.0},
    'GW151226': {'gps': 1135136350.6, 'Mc_min':  5.0, 'Mc_max': 15.0},
    'GW170104': {'gps': 1167559936.6, 'Mc_min': 15.0, 'Mc_max': 40.0},
    'GW170608': {'gps': 1180922494.5, 'Mc_min':  5.0, 'Mc_max': 15.0},
    'GW170809': {'gps': 1186302519.8, 'Mc_min': 15.0, 'Mc_max': 40.0},
    'GW170814': {'gps': 1186741861.5, 'Mc_min': 15.0, 'Mc_max': 40.0},
    'GW170823': {'gps': 1187529256.5, 'Mc_min': 20.0, 'Mc_max': 50.0},
}

parser = argparse.ArgumentParser()
parser.add_argument('--event', required=True, choices=list(EVENTS.keys()))
args = parser.parse_args()

cfg  = EVENTS[args.event]
name = args.event.lower()
gps  = cfg['gps']
gmst = Time(gps, format='gps').sidereal_time('apparent', 'greenwich').rad

# ---------------------------------------------------------------------------
# GH quadrature constants
# ---------------------------------------------------------------------------
_gh_x_np, _gh_w_np = np.polynomial.hermite.hermgauss(20)
GH_X    = jnp.array(_gh_x_np)
GH_LOGW = jnp.log(jnp.array(_gh_w_np))

# Physical anomaly range
DELTA     = 4e-38
LOG_DELTA = jnp.log(DELTA)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
frequencies = jnp.array(np.load(f'{name}_frequencies.npy'), dtype=jnp.float64)
detectors = [H1, L1]
for det in detectors:
    det.frequencies = frequencies
    det.data = jnp.array(np.load(f'{name}_{det.name}_strain.npy'), dtype=jnp.complex128)
    det.psd  = jnp.array(np.load(f'{name}_{det.name}_psd.npy'),    dtype=jnp.float64)
    det.mask = jnp.ones(len(frequencies), dtype=jnp.float64)

waveform = RippleIMRPhenomD(f_ref=20)
post_trigger_duration = 2
duration = 4
epoch = duration - post_trigger_duration

# ---------------------------------------------------------------------------
# Per-bin likelihood functions
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
        fp  = -4.0 * B_i*e2s     - 1.0/tau_sq
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

    base_keys   = ["M_c","q","s1_z","s2_z","iota","d_L","t_c","psi","ra","dec","phase_c"]
    sample_keys = (base_keys
                   + (["tau"]    if use_alcs else [])
                   + (["p_anom"] if use_bad  else []))

    test_particles = {key: jax.random.uniform(jax.random.PRNGKey(42), (100,)) for key in sample_keys}
    sample_keys = get_ravel_order(test_particles)

    param_config = {
        "M_c":     {"min": cfg['Mc_min'], "max": cfg['Mc_max'], "prior": "uniform",     "wraparound": False},
        "q":       {"min": 0.25,          "max": 1.0,           "prior": "uniform",     "wraparound": False},
        "s1_z":    {"min": -1.0,          "max": 1.0,           "prior": "uniform",     "wraparound": False},
        "s2_z":    {"min": -1.0,          "max": 1.0,           "prior": "uniform",     "wraparound": False},
        "iota":    {"min": 0.0,           "max": jnp.pi,        "prior": "sine",        "wraparound": False},
        "d_L":     {"min": 100.0,         "max": 5000.0,        "prior": "powerlaw",    "wraparound": False},
        "t_c":     {"min": -0.1,          "max": 0.1,           "prior": "uniform",     "wraparound": False},
        "phase_c": {"min": 0.0,           "max": 2*jnp.pi,      "prior": "uniform",     "wraparound": True},
        "psi":     {"min": 0.0,           "max": jnp.pi,        "prior": "uniform",     "wraparound": True},
        "ra":      {"min": 0.0,           "max": 2*jnp.pi,      "prior": "uniform",     "wraparound": True},
        "dec":     {"min": -jnp.pi/2,     "max": jnp.pi/2,      "prior": "cosine",      "wraparound": False},
        "tau":     {"min": 0.001,         "max": 2.0,           "prior": "log_uniform", "wraparound": False},
        "p_anom":  {"min": 1e-4,          "max": 0.5,           "prior": "log_uniform", "wraparound": False},
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
                      (param_mins**(1+2) + u*(param_maxs**(1+2)-param_mins**(1+2)))**(1/3),
                      param_mins*(param_maxs/param_mins)**u))))
        _, unflatten = jax.flatten_util.ravel_pytree({k: 0.0 for k in sample_keys})
        return unflatten(x)

    @jax.jit
    def logprior_fn(params):
        x, _ = jax.flatten_util.ravel_pytree(params)
        lp = jnp.where(param_prior_types == 0,
                        jnp.where((x>=param_mins)&(x<=param_maxs), -jnp.log(param_maxs-param_mins), -jnp.inf),
             jnp.where(param_prior_types == 1,
                        jnp.where((x>=0)&(x<=jnp.pi), jnp.log(jnp.sin(x)/2), -jnp.inf),
             jnp.where(param_prior_types == 2,
                        jnp.where(jnp.abs(x)<jnp.pi/2, jnp.log(jnp.cos(x)/2), -jnp.inf),
             jnp.where(param_prior_types == 3,
                        jnp.where((x>=param_mins)&(x<=param_maxs),
                                  2*jnp.log(x)+jnp.log(3)-jnp.log(param_maxs**3-param_mins**3), -jnp.inf),
                        jnp.where((x>=param_mins)&(x<=param_maxs),
                                  -jnp.log(x)-jnp.log(jnp.log(param_maxs/param_mins)), -jnp.inf)))))
        return jnp.sum(lp)

    if use_alcs and use_bad:
        _per_bin_v = jax.vmap(
            lambda B, lS, tau, p: _bad_combine(_alcs_single_bin(B, lS, tau, None), p),
            in_axes=(0,0,None,None))
        # fix: pass log_two_df_over_pi via closure below
    elif use_alcs:
        pass
    elif use_bad:
        pass

    def loglikelihood_fn(params):
        p = dict(params)
        p["gmst"] = gmst
        p["eta"]  = p["q"] / (1 + p["q"])**2
        waveform_sky = waveform(frequencies, p)
        align_time   = jnp.exp(-1j*2*jnp.pi*frequencies*(epoch+p["t_c"]))
        df = frequencies[1] - frequencies[0]
        log_two_df_over_pi = jnp.log(2.0*df/jnp.pi)
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
    with tqdm.tqdm(desc=f"{args.event} {mode}", unit=" dead points") as pbar:
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

    out = f"{name}_{mode}"
    samples.to_csv(f"{out}.csv")
    with open(f"{out}_final_state.pkl", "wb") as f:
        pickle.dump(final_state, f)
    print(f"Saved {out}.csv   logZ = {samples.logZ():.2f}")

# ---------------------------------------------------------------------------
# Run all four modes in series
# ---------------------------------------------------------------------------
for mode in ['norm', 'bad', 'alcs_quad', 'alcs_bad_quad']:
    print(f"\n{'='*60}")
    print(f"  {args.event}  |  mode = {mode}")
    print(f"{'='*60}")
    run_one_mode(mode)
