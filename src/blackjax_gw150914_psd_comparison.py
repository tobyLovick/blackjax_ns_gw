"""
GW150914 PSD comparison — norm vs ALCS+BAD for different PSD estimators.

Runs two models in series:
  norm      -- standard fixed-PSD Whittle likelihood (11 GW params)
  alcs_bad  -- ALCS + BAD (+tau, +p_anom)

Use --psd-suffix to select PSD estimator, e.g.:
  python blackjax_gw150914_psd_comparison.py               # Welch (default)
  python blackjax_gw150914_psd_comparison.py --psd-suffix _mesa

Output CSVs: blackjaxns_psdcomp_gw150914_{mode}{psd_suffix}.csv
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--psd-suffix", type=str, default="",
                    help="Suffix on PSD filename, e.g. _mesa (default: Welch)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load GW150914 data
# ---------------------------------------------------------------------------
waveform = RippleIMRPhenomD(f_ref=20)

frequencies = jnp.array(np.load('gw150914_frequencies.npy'), dtype=jnp.float64)

detectors = [H1, L1]
for det in detectors:
    det.frequencies = frequencies
    det.data = jnp.array(np.load(f'gw150914_{det.name}_strain.npy'), dtype=jnp.complex128)
    det.psd  = jnp.array(np.load(f'gw150914_{det.name}_psd{args.psd_suffix}.npy'), dtype=jnp.float64)
    det.mask = jnp.ones(len(frequencies), dtype=jnp.float64)

post_trigger_duration = 2
duration = 4
epoch    = duration - post_trigger_duration
gps      = 1126259462.4
gmst     = Time(gps, format='gps').sidereal_time('apparent', 'greenwich').rad

# ---------------------------------------------------------------------------
# GH quadrature constants (20-point, for ALCS)
# ---------------------------------------------------------------------------
_gh_x_np, _gh_w_np = np.polynomial.hermite.hermgauss(20)
GH_X    = jnp.array(_gh_x_np)
GH_LOGW = jnp.log(jnp.array(_gh_w_np))

# BAD anomaly floor
LOG_DELTA = jnp.log(4e-38)

# ---------------------------------------------------------------------------
# Per-bin likelihood kernels
# ---------------------------------------------------------------------------
def _norm_bin(B_i, log_S_i, log_two_df_over_pi):
    return log_two_df_over_pi - log_S_i - B_i * jnp.exp(-log_S_i)

def _alcs_bin(B_i, log_S_i, tau, log_two_df_over_pi):
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
    log_int = jnp.log(scale) + jax.scipy.special.logsumexp(GH_LOGW + g_nodes + GH_X**2)
    return log_two_df_over_pi - 0.5*jnp.log(2.0*jnp.pi*tau_sq) + log_int

def _bad_combine(log_Z_i, p_anom):
    return jnp.logaddexp(jnp.log1p(-p_anom) + log_Z_i, jnp.log(p_anom) - LOG_DELTA)

# ---------------------------------------------------------------------------
# Helper: determine ravel order of parameter dict
# ---------------------------------------------------------------------------
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
# Run one mode: 'norm' or 'alcs_bad'
# ---------------------------------------------------------------------------
def run_mode(mode):
    use_alcs = 'alcs' in mode
    use_bad  = 'bad'  in mode

    sample_keys = (["M_c","q","s1_z","s2_z","iota","d_L","t_c","psi","ra","dec","phase_c"]
                   + (["tau"]    if use_alcs else [])
                   + (["p_anom"] if use_bad  else []))

    test_p      = {k: jax.random.uniform(jax.random.PRNGKey(0), (100,)) for k in sample_keys}
    sample_keys = get_ravel_order(test_p)

    GW_CONFIG = {
        "M_c":     {"min": 25.0,      "max": 50.0,       "type": "uniform",     "wrap": False},
        "q":       {"min": 0.25,      "max": 1.0,        "type": "uniform",     "wrap": False},
        "s1_z":    {"min": -1.0,      "max": 1.0,        "type": "uniform",     "wrap": False},
        "s2_z":    {"min": -1.0,      "max": 1.0,        "type": "uniform",     "wrap": False},
        "iota":    {"min": 0.0,       "max": jnp.pi,     "type": "sine",        "wrap": False},
        "d_L":     {"min": 100.0,     "max": 5000.0,     "type": "powerlaw",    "wrap": False},
        "t_c":     {"min": -0.1,      "max": 0.1,        "type": "uniform",     "wrap": False},
        "phase_c": {"min": 0.0,       "max": 2*jnp.pi,   "type": "uniform",     "wrap": True},
        "psi":     {"min": 0.0,       "max": jnp.pi,     "type": "uniform",     "wrap": True},
        "ra":      {"min": 0.0,       "max": 2*jnp.pi,   "type": "uniform",     "wrap": True},
        "dec":     {"min": -jnp.pi/2, "max": jnp.pi/2,   "type": "cosine",      "wrap": False},
        "tau":     {"min": 0.001,     "max": 2.0,        "type": "log_uniform", "wrap": False},
        "p_anom":  {"min": 1e-4,      "max": 0.5,        "type": "log_uniform", "wrap": False},
    }
    sc = {k: GW_CONFIG[k] for k in sample_keys}

    # type encoding: 0=uniform, 1=sine, 2=cosine, 3=powerlaw, 4=gaussian, 5=log_uniform
    type_arr = jnp.array([0 if sc[k]["type"]=="uniform"     else
                           1 if sc[k]["type"]=="sine"        else
                           2 if sc[k]["type"]=="cosine"      else
                           3 if sc[k]["type"]=="powerlaw"    else
                           5   # log_uniform
                           for k in sample_keys])
    mins = jnp.array([sc[k].get("min", 0.) for k in sample_keys])
    maxs = jnp.array([sc[k].get("max", 1.) for k in sample_keys])

    @jax.jit
    def prior_transform_fn(u_params):
        u, _ = jax.flatten_util.ravel_pytree(u_params)
        x = jnp.where(type_arr == 0, mins + u*(maxs - mins),
            jnp.where(type_arr == 1, jnp.arccos(1 - 2*u),
            jnp.where(type_arr == 2, jnp.arcsin(2*u - 1),
            jnp.where(type_arr == 3, (mins**3 + u*(maxs**3 - mins**3))**(1/3),
                      mins * (maxs/mins)**u))))   # log_uniform
        _, unf = jax.flatten_util.ravel_pytree({k: 0. for k in sample_keys})
        return unf(x)

    @jax.jit
    def logprior_fn(params):
        vals, _ = jax.flatten_util.ravel_pytree(params)
        lp = jnp.where(type_arr == 0,
                       jnp.where((vals>=mins)&(vals<=maxs), -jnp.log(maxs-mins), -jnp.inf),
             jnp.where(type_arr == 1,
                       jnp.where((vals>=0)&(vals<=jnp.pi), jnp.log(jnp.sin(vals)/2), -jnp.inf),
             jnp.where(type_arr == 2,
                       jnp.where(jnp.abs(vals)<jnp.pi/2, jnp.log(jnp.cos(vals)/2), -jnp.inf),
             jnp.where(type_arr == 3,
                       2*jnp.log(vals) - jnp.log(maxs**3 - mins**3),
                       -jnp.log(vals) - jnp.log(jnp.log(maxs/mins))))))  # log_uniform
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
            h    = det.fd_response(frequencies, waveform_sky, p) * align_time
            B    = 2.0 * df * jnp.abs(det.data - h)**2
            logS = jnp.log(det.psd)
            if use_alcs and use_bad:
                bins = jax.vmap(lambda b, ls: _bad_combine(
                    _alcs_bin(b, ls, p["tau"], log_two_df_over_pi),
                    p["p_anom"]))(B, logS)
            elif use_alcs:
                bins = jax.vmap(lambda b, ls: _alcs_bin(
                    b, ls, p["tau"], log_two_df_over_pi))(B, logS)
            elif use_bad:
                bins = jax.vmap(lambda b, ls: _bad_combine(
                    _norm_bin(b, ls, log_two_df_over_pi),
                    p["p_anom"]))(B, logS)
            else:
                bins = jax.vmap(lambda b, ls: _norm_bin(
                    b, ls, log_two_df_over_pi))(B, logS)
            log_L = log_L + jnp.dot(bins, det.mask)
        return log_L

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

    from blackjax.ns.utils import finalise
    from anesthetic import NestedSamples, read_chains

    dead = []
    with tqdm.tqdm(desc=f"GW150914 {mode}{args.psd_suffix}", unit=" dead points") as pbar:
        while not terminate(state):
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)

    final_state     = finalise(state, dead)
    physical_params = transform_to_physical(final_state.particles, prior_transform_fn)
    logL_birth      = jnp.where(jnp.isnan(final_state.loglikelihood_birth),
                                 -jnp.inf, final_state.loglikelihood_birth)

    labels = {
        "M_c": r"$\mathcal{M}_c$", "q": r"$q$", "d_L": r"$d_L$",
        "iota": r"$\iota$", "ra": r"$\alpha$", "dec": r"$\delta$",
        "s1_z": r"$\chi_1$", "s2_z": r"$\chi_2$", "t_c": r"$t_c$",
        "psi": r"$\psi$", "phase_c": r"$\phi_c$",
        "tau": r"$\tau$", "p_anom": r"$p_{\rm anom}$",
    }

    samples = NestedSamples(
        physical_params, logL=final_state.loglikelihood,
        logL_birth=logL_birth, labels=labels,
        logzero=jnp.nan, dtype=jnp.float64,
    )
    csv_path = f"blackjaxns_psdcomp_gw150914_{mode}{args.psd_suffix}.csv"
    samples.to_csv(csv_path)
    with open(f"blackjaxns_psdcomp_gw150914_{mode}{args.psd_suffix}_final_state.pkl", "wb") as f:
        pickle.dump(final_state, f)
    logZ_val = read_chains(csv_path).logZ()
    print(f"  [{mode:10s}{args.psd_suffix}]  logZ = {logZ_val:.2f}  (n_params = {len(sample_keys)})")
    return logZ_val

# ---------------------------------------------------------------------------
# Run norm then alcs_bad, skipping if CSV already exists
# ---------------------------------------------------------------------------
from anesthetic import read_chains

results = {}
for mode in ['norm', 'alcs_bad']:
    csv_path = f"blackjaxns_psdcomp_gw150914_{mode}{args.psd_suffix}.csv"
    if os.path.exists(csv_path):
        results[mode] = read_chains(csv_path).logZ()
        print(f"  [{mode:10s}{args.psd_suffix}]  logZ = {results[mode]:.2f}  (loaded from {csv_path})")
    else:
        print(f"\n{'='*60}")
        print(f"  GW150914  |  mode = {mode}  |  PSD = {args.psd_suffix or 'welch'}")
        print(f"{'='*60}")
        results[mode] = run_mode(mode)

print(f"\nlog Bayes factor  alcs_bad vs norm  [{args.psd_suffix or 'welch'}]:  "
      f"{results['alcs_bad'] - results['norm']:+.2f} nats")
