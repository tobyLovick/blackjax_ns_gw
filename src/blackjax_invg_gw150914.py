# Memory configuration
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
    transform_to_physical
)

waveform = RippleIMRPhenomD(f_ref=20)

# ---------------------------------------------------------------------------
# Load pre-fetched GW150914 data (run fetch_gw150914.py once to generate)
# ---------------------------------------------------------------------------
frequencies = jnp.array(np.load('gw150914_frequencies.npy'), dtype=jnp.float64)

detectors = [H1, L1]
for det in detectors:
    det.frequencies = frequencies
    det.data = jnp.array(np.load(f'gw150914_{det.name}_strain.npy'), dtype=jnp.complex128)
    det.psd  = jnp.array(np.load(f'gw150914_{det.name}_psd.npy'),    dtype=jnp.float64)

# ---------------------------------------------------------------------------
# Helper: determine ravel order of parameter dict
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
# Parameters
# ---------------------------------------------------------------------------
sample_keys = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "psi", "ra", "dec", "phase_c", "alpha_0"]

test_particles = {key: jax.random.uniform(jax.random.PRNGKey(42), (100,)) for key in sample_keys}
sample_keys = get_ravel_order(test_particles)

param_config = {
    "M_c":     {"min": 25.0,         "max": 50.0,        "prior": "uniform",     "wraparound": False, "angle": 1.0},
    "q":       {"min": 0.25,         "max": 1.0,         "prior": "uniform",     "wraparound": False, "angle": 1.0},
    "s1_z":    {"min": -1.0,         "max": 1.0,         "prior": "uniform",     "wraparound": False, "angle": 1.0},
    "s2_z":    {"min": -1.0,         "max": 1.0,         "prior": "uniform",     "wraparound": False, "angle": 1.0},
    "iota":    {"min": 0.0,          "max": jnp.pi,      "prior": "sine",        "wraparound": False, "angle": 1.0},
    "d_L":     {"min": 100.0,        "max": 5000.0,      "prior": "powerlaw",    "wraparound": False, "angle": 1.0},
    "t_c":     {"min": -0.1,         "max": 0.1,         "prior": "uniform",     "wraparound": False, "angle": 1.0},
    "phase_c": {"min": 0.0,          "max": 2*jnp.pi,    "prior": "uniform",     "wraparound": True,  "angle": 2*jnp.pi},
    "psi":     {"min": 0.0,          "max": jnp.pi,      "prior": "uniform",     "wraparound": True,  "angle": jnp.pi},
    "ra":      {"min": 0.0,          "max": 2*jnp.pi,    "prior": "uniform",     "wraparound": True,  "angle": 2*jnp.pi},
    "dec":     {"min": -jnp.pi/2,    "max": jnp.pi/2,    "prior": "cosine",      "wraparound": False, "angle": 1.0},
    "alpha_0": {"min": 3.0,          "max": 1000.0,      "prior": "log_uniform", "wraparound": False, "angle": 1.0},
}

sampled_config = {key: param_config[key] for key in sample_keys}
n_dims = len(sample_keys)

param_mins = jnp.array([sampled_config[key]["min"] for key in sample_keys])
param_maxs = jnp.array([sampled_config[key]["max"] for key in sample_keys])
param_prior_types = jnp.array([
    0 if sampled_config[key]["prior"] == "uniform"     else
    1 if sampled_config[key]["prior"] == "sine"        else
    2 if sampled_config[key]["prior"] == "cosine"      else
    3 if sampled_config[key]["prior"] == "powerlaw"    else
    4 for key in sample_keys
])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
post_trigger_duration = 2
duration = 4
epoch = duration - post_trigger_duration
gps = 1126259462.4
gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad

# ---------------------------------------------------------------------------
# Prior transforms
# ---------------------------------------------------------------------------
@jax.jit
def uniform_transform(u, a, b):
    return a + u * (b - a)

@jax.jit
def sine_transform(u):
    return jnp.arccos(1 - 2 * u)

@jax.jit
def cosine_transform(u):
    return jnp.arcsin(2 * u - 1)

@jax.jit
def powerlaw_transform(u, alpha, min, max):
    return (min**(1+alpha) + u * (max**(1+alpha) - min**(1+alpha)))**(1/(1+alpha))

@jax.jit
def log_uniform_transform_alpha0(u):
    return 3.0 * (1000.0 / 3.0) ** u

@jax.jit
def prior_transform_fn(u_params):
    u_values, _ = jax.flatten_util.ravel_pytree(u_params)
    uniform_vals     = uniform_transform(u_values, param_mins, param_maxs)
    sine_vals        = sine_transform(u_values)
    cosine_vals      = cosine_transform(u_values)
    powerlaw_vals    = powerlaw_transform(u_values, 2, param_mins, param_maxs)
    log_uniform_vals = log_uniform_transform_alpha0(u_values)
    x_values = jnp.where(
        param_prior_types == 0, uniform_vals,
        jnp.where(param_prior_types == 1, sine_vals,
        jnp.where(param_prior_types == 2, cosine_vals,
        jnp.where(param_prior_types == 3, powerlaw_vals,
                  log_uniform_vals))))
    example_params = {key: 0.0 for key in sample_keys}
    _, unflatten_fn = jax.flatten_util.ravel_pytree(example_params)
    return unflatten_fn(x_values)

# ---------------------------------------------------------------------------
# InvGamma per-bin analytic marginalisation
# ---------------------------------------------------------------------------
def _invg_single_bin(B_i, S0_i, alpha_0, log_two_df_over_pi):
    beta_0 = S0_i * (alpha_0 - 1.0)
    return (log_two_df_over_pi
            + jnp.log(alpha_0)
            + alpha_0 * jnp.log(beta_0)
            - (alpha_0 + 1.0) * jnp.log(beta_0 + B_i))

_invg_bins = jax.vmap(_invg_single_bin, in_axes=(0, 0, None, None))

# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------
def loglikelihood_fn(params):
    p = dict(params)
    p["gmst"] = gmst
    p["eta"]  = p["q"] / (1 + p["q"]) ** 2
    waveform_sky       = waveform(frequencies, p)
    align_time         = jnp.exp(-1j * 2 * jnp.pi * frequencies * (epoch + p["t_c"]))
    df                 = frequencies[1] - frequencies[0]
    log_two_df_over_pi = jnp.log(2.0 * df / jnp.pi)
    log_L = 0.0
    for det in detectors:
        h_dec = det.fd_response(frequencies, waveform_sky, p) * align_time
        B     = 2.0 * df * jnp.abs(det.data - h_dec) ** 2
        log_L = log_L + jnp.sum(_invg_bins(B, det.psd, p["alpha_0"], log_two_df_over_pi))
    return log_L

# ---------------------------------------------------------------------------
# Prior log-probabilities
# ---------------------------------------------------------------------------
@jax.jit
def uniform_logprob(x, a, b):
    return jnp.where((x >= a) & (x <= b), -jnp.log(b - a), -jnp.inf)

@jax.jit
def sine_logprob(x):
    return jnp.where((x >= 0.0) & (x <= jnp.pi), jnp.log(jnp.sin(x) / 2.0), -jnp.inf)

@jax.jit
def cosine_logprob(x):
    return jnp.where(jnp.abs(x) < jnp.pi / 2, jnp.log(jnp.cos(x) / 2.0), -jnp.inf)

@jax.jit
def powerlaw_logprob(x, alpha, min, max):
    logpdf = alpha*jnp.log(x) + jnp.log(1+alpha) - jnp.log(max**(1+alpha) - min**(1+alpha))
    return jnp.where((x >= min) & (x <= max), logpdf, -jnp.inf)

@jax.jit
def log_uniform_logprob_alpha0(x):
    return jnp.where((x >= 3.0) & (x <= 1000.0),
                     -jnp.log(x) - jnp.log(1000.0 / 3.0),
                     -jnp.inf)

@jax.jit
def logprior_fn(params):
    param_values, _ = jax.flatten_util.ravel_pytree(params)
    uniform_priors     = uniform_logprob(param_values, param_mins, param_maxs)
    sine_priors        = sine_logprob(param_values)
    cosine_priors      = cosine_logprob(param_values)
    powerlaw_priors    = powerlaw_logprob(param_values, 2, param_mins, param_maxs)
    log_uniform_priors = log_uniform_logprob_alpha0(param_values)
    priors = jnp.where(
        param_prior_types == 0, uniform_priors,
        jnp.where(param_prior_types == 1, sine_priors,
        jnp.where(param_prior_types == 2, cosine_priors,
        jnp.where(param_prior_types == 3, powerlaw_priors,
                  log_uniform_priors))))
    return jnp.sum(priors)

# ---------------------------------------------------------------------------
# Nested sampling
# ---------------------------------------------------------------------------
n_live   = 1400
n_delete = 700

rng_key = jax.random.PRNGKey(10)
rng_key, init_key = jax.random.split(rng_key)

example_params      = {key: 0.0 for key in sample_keys}
unit_cube_particles = init_unit_cube_particles(init_key, example_params, n_live)

periodic_mask = jax.tree_util.tree_map(lambda _: False, example_params)
for key in sample_keys:
    if sampled_config[key]["wraparound"]:
        periodic_mask[key] = True

unit_cube_fns = create_unit_cube_functions(
    physical_loglikelihood_fn=loglikelihood_fn,
    prior_transform_fn=prior_transform_fn,
    mask_tree=periodic_mask,
)

nested_sampler = acceptance_walk_sampler(
    logprior_fn=unit_cube_fns['logprior_fn'],
    loglikelihood_fn=unit_cube_fns['loglikelihood_fn'],
    nlive=n_live,
    n_target=60,
    max_mcmc=5000,
    num_delete=n_delete,
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
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not terminate(state):
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(n_delete)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
from blackjax.ns.utils import finalise
from anesthetic import NestedSamples

column_to_label = {
    "M_c": r"$\mathcal{M}_c\ [M_\odot]$", "q": r"$q$",
    "d_L": r"$d_L\ [\mathrm{Mpc}]$",       "iota": r"$\iota\ [\mathrm{rad}]$",
    "ra":  r"$\alpha$",                      "dec":  r"$\delta$",
    "s1_z": r"$\chi_1$",                    "s2_z": r"$\chi_2$",
    "t_c":  r"$t_c$",                        "psi":  r"$\psi$",
    "phase_c": r"$\phi_c$",                  "alpha_0": r"$\alpha_0$",
}

final_state = finalise(state, dead)

with open('blackjaxns_invg_gw150914_final_state.pkl', 'wb') as f:
    pickle.dump(final_state, f)

physical_particles = transform_to_physical(final_state.particles, prior_transform_fn)

logL_birth = final_state.loglikelihood_birth.copy()
logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)
samples = NestedSamples(
    physical_particles,
    logL=final_state.loglikelihood,
    logL_birth=logL_birth,
    labels=column_to_label,
    logzero=jnp.nan,
    dtype=jnp.float64,
)

samples.to_csv("blackjaxns_invg_gw150914.csv")
with open('blackjaxns_invg_gw150914_timings.pkl', 'wb') as f:
    pickle.dump(pbar.format_dict, f)
