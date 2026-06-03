"""
Per-job PP test script for the 4-mode Welch PSD study.

Usage:
    python blackjax_pp_injection.py --idx N [--catalog pp_catalog.npy]
                                            [--outdir pp_results]
                                            [--asd aLIGO_O4_high_asd.txt]

Generates synthetic H1+L1 data for event N using seed=N, estimates a Welch PSD
from 4 off-source noise segments, then runs 4 nested sampling chains:
    norm_true, norm_welch, alcs_true, alcs_welch

Saves pp_results/N/{mode}.csv for post-processing by pp_summarise.py.
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from anesthetic import NestedSamples
from blackjax.ns.utils import finalise
from scipy.interpolate import interp1d

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
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--idx',     type=int, required=True,
                    help='Injection index 0-499')
parser.add_argument('--catalog', default='pp_catalog.npy')
parser.add_argument('--outdir',  default='pp_results')
parser.add_argument('--asd',     default='aLIGO_O4_high_asd.txt')
args = parser.parse_args()

IDX    = args.idx
outdir = os.path.join(args.outdir, str(IDX))
os.makedirs(outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# True parameters from PP catalog
# ---------------------------------------------------------------------------
catalog    = np.load(args.catalog, allow_pickle=True)
true_p     = {k: float(v) for k, v in catalog[IDX].items()}
gmst       = true_p['gmst']

print(f"[idx={IDX}]  Mc={true_p['M_c']:.2f}  q={true_p['q']:.3f}  "
      f"d_L={true_p['d_L']:.0f} Mpc")

# ---------------------------------------------------------------------------
# Frequency grid and PSD
# ---------------------------------------------------------------------------
F_MIN        = 20.0
F_MAX        = 1024.0
DURATION     = 4
SAMPLE_RATE  = 4096
F_REF        = 20.0
N_WELCH_SEGS = 4

n_samples = DURATION * SAMPLE_RATE
delta_t   = 1.0 / SAMPLE_RATE
delta_f   = 1.0 / DURATION

freq_full = np.fft.rfftfreq(n_samples, d=delta_t)
band_mask = (freq_full >= F_MIN) & (freq_full <= F_MAX)
freq      = freq_full[band_mask]
n_freq    = len(freq)

asd_data   = np.loadtxt(args.asd)
asd_interp = interp1d(asd_data[:, 0], asd_data[:, 1],
                      kind='linear', fill_value='extrapolate')
psd_true   = asd_interp(freq) ** 2
noise_std  = np.sqrt(psd_true / (2.0 * delta_f))

# ---------------------------------------------------------------------------
# Waveform and detectors
# ---------------------------------------------------------------------------
waveform  = RippleIMRPhenomD(f_ref=F_REF)
epoch     = DURATION - 2
DETECTORS = [H1, L1]

GH_X, GH_LOGW = (lambda x, w: (jnp.array(x), jnp.log(jnp.array(w))))(
    *np.polynomial.hermite.hermgauss(20))

# ---------------------------------------------------------------------------
# Generate synthetic data (seed = IDX → reproducible per job)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(IDX)

def _noise():
    """One band-limited FD noise realisation on the analysis band."""
    nr = rng.standard_normal(n_freq)
    ni = rng.standard_normal(n_freq)
    return (nr + 1j * ni) / np.sqrt(2.0) * noise_std

freq_jnp = jnp.array(freq, dtype=jnp.float64)
true_p['eta'] = true_p['q'] / (1 + true_p['q'])**2
ws    = waveform(freq_jnp, true_p)
align = jnp.exp(-1j * 2 * jnp.pi * freq_jnp * (epoch + true_p['t_c']))

data_H1 = np.array(H1.fd_response(freq_jnp, ws, true_p) * align) + _noise()
data_L1 = np.array(L1.fd_response(freq_jnp, ws, true_p) * align) + _noise()

# Welch PSD: average M=N_WELCH_SEGS periodograms
# S_welch[k] = (2*delta_f / M) * sum_j |n_fd_j[k]|^2  →  E[S_welch] = psd_true
welch_sum = np.zeros(n_freq, dtype=float)
for _ in range(N_WELCH_SEGS):
    welch_sum += np.abs(_noise())**2
psd_welch = 2.0 * delta_f * welch_sum / N_WELCH_SEGS

# JAX arrays
psd_true_jnp  = jnp.array(psd_true,  dtype=jnp.float64)
psd_welch_jnp = jnp.array(psd_welch, dtype=jnp.float64)
data_H1_jnp   = jnp.array(data_H1,   dtype=jnp.complex128)
data_L1_jnp   = jnp.array(data_L1,   dtype=jnp.complex128)
mask_jnp      = jnp.ones(n_freq,      dtype=jnp.float64)

# ---------------------------------------------------------------------------
# Likelihood kernels
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

# ---------------------------------------------------------------------------
# Prior — must exactly match create_pp_catalog.py
# ---------------------------------------------------------------------------
# s1_z, s2_z fixed to 0 (non-spinning injections; excluded from sampling)
PARAM_CONFIG = {
    "M_c":     {"min":  8.0,          "max": 60.0,        "prior": "uniform",     "wrap": False},
    "q":       {"min":  0.2,          "max":  1.0,        "prior": "uniform",     "wrap": False},
    "iota":    {"min":  0.0,          "max": float(jnp.pi), "prior": "sine",      "wrap": False},
    "d_L":     {"min": 100.0,         "max": 2500.0,      "prior": "powerlaw",    "wrap": False},
    "t_c":     {"min": -0.05,         "max":  0.05,       "prior": "uniform",     "wrap": False},
    "phase_c": {"min":  0.0,          "max": float(2*jnp.pi), "prior": "uniform", "wrap": True},
    "psi":     {"min":  0.0,          "max": float(jnp.pi),   "prior": "uniform", "wrap": True},
    "ra":      {"min":  0.0,          "max": float(2*jnp.pi), "prior": "uniform", "wrap": True},
    "dec":     {"min": -float(jnp.pi/2), "max": float(jnp.pi/2), "prior": "cosine", "wrap": False},
    "tau":     {"min":  0.001,        "max":  2.0,        "prior": "log_uniform", "wrap": False},
}

def get_ravel_order(keys):
    test = {k: float(i) for i, k in enumerate(keys)}
    flat, _ = jax.flatten_util.ravel_pytree(test)
    order = []
    for val in flat:
        for k, v in test.items():
            if abs(val - v) < 1e-10:
                order.append(k)
                break
    return order

# ---------------------------------------------------------------------------
# Run one mode
# ---------------------------------------------------------------------------
def run_one_mode(likelihood_model, psd_label):
    mode_name = f"{likelihood_model}_{psd_label}"
    out_path  = os.path.join(outdir, mode_name)
    if os.path.exists(f"{out_path}.csv"):
        print(f"  [skip] {mode_name} already exists")
        return

    use_alcs = (likelihood_model == 'alcs')
    psd_jnp  = psd_true_jnp if psd_label == 'true' else psd_welch_jnp

    base_keys   = ["M_c","q","iota","d_L","t_c","phase_c","psi","ra","dec"]
    sample_keys = base_keys + (["tau"] if use_alcs else [])
    sample_keys = get_ravel_order(sample_keys)

    sc          = {k: PARAM_CONFIG[k] for k in sample_keys}
    param_mins  = jnp.array([sc[k]["min"] for k in sample_keys])
    param_maxs  = jnp.array([sc[k]["max"] for k in sample_keys])
    prior_types = jnp.array([
        0 if sc[k]["prior"] == "uniform"     else
        1 if sc[k]["prior"] == "sine"        else
        2 if sc[k]["prior"] == "cosine"      else
        3 if sc[k]["prior"] == "powerlaw"    else
        4 for k in sample_keys
    ])

    @jax.jit
    def prior_transform_fn(u_params):
        u, _ = jax.flatten_util.ravel_pytree(u_params)
        x = jnp.where(prior_types == 0, param_mins + u*(param_maxs - param_mins),
            jnp.where(prior_types == 1, jnp.arccos(1.0 - 2.0*u),
            jnp.where(prior_types == 2, jnp.arcsin(2.0*u - 1.0),
            jnp.where(prior_types == 3,
                      (param_mins**3 + u*(param_maxs**3 - param_mins**3))**(1.0/3.0),
                      param_mins*(param_maxs/param_mins)**u))))
        _, unflatten = jax.flatten_util.ravel_pytree({k: 0.0 for k in sample_keys})
        return unflatten(x)

    det_data = [
        {'data': data_H1_jnp, 'psd': psd_jnp, 'fd_response': H1.fd_response},
        {'data': data_L1_jnp, 'psd': psd_jnp, 'fd_response': L1.fd_response},
    ]

    def loglikelihood_fn(params):
        p = dict(params)
        p["gmst"] = gmst
        p["s1_z"] = 0.0
        p["s2_z"] = 0.0
        p["eta"]  = p["q"] / (1.0 + p["q"])**2
        ws_sky         = waveform(freq_jnp, p)
        align_t        = jnp.exp(-1j*2*jnp.pi*freq_jnp*(epoch + p["t_c"]))
        df             = freq_jnp[1] - freq_jnp[0]
        log_two_df_pi  = jnp.log(2.0 * df / jnp.pi)
        log_L = 0.0
        for det in det_data:
            h    = det['fd_response'](freq_jnp, ws_sky, p) * align_t
            B    = 2.0 * df * jnp.abs(det['data'] - h)**2
            logS = jnp.log(det['psd'])
            if use_alcs:
                bins = jax.vmap(
                    lambda b, ls: _alcs_bin(b, ls, p["tau"], log_two_df_pi)
                )(B, logS)
            else:
                bins = jax.vmap(
                    lambda b, ls: _norm_bin(b, ls, log_two_df_pi)
                )(B, logS)
            log_L = log_L + jnp.sum(bins * mask_jnp)
        return log_L

    n_live   = 1400
    n_delete = 700
    rng_key  = jax.random.PRNGKey(IDX * 10 + hash(mode_name) % 7)
    rng_key, init_key = jax.random.split(rng_key)

    example_params      = {k: 0.0 for k in sample_keys}
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
        nlive=n_live, n_target=60, max_mcmc=5000, num_delete=n_delete,
        stepper_fn=unit_cube_fns['stepper_fn'],
    )
    state = nested_sampler.init(unit_cube_particles)

    @jax.jit
    def one_step(carry, _):
        state, k = carry
        k, subk = jax.random.split(k)
        state, dead_point = nested_sampler.step(subk, state)
        return (state, k), dead_point

    def terminate(state):
        dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
        return jnp.isfinite(dlogz) and dlogz < 0.1

    dead = []
    with tqdm.tqdm(desc=f"[{IDX}] {mode_name}", unit=" dead") as pbar:
        while not terminate(state):
            (state, rng_key), dp = one_step((state, rng_key), None)
            dead.append(dp)
            pbar.update(n_delete)

    final_state      = finalise(state, dead)
    phys_particles   = transform_to_physical(final_state.particles, prior_transform_fn)
    logL_birth       = jnp.where(jnp.isnan(final_state.loglikelihood_birth),
                                  -jnp.inf, final_state.loglikelihood_birth)

    col_labels = {
        "M_c": r"$\mathcal{M}_c$", "q": r"$q$", "iota": r"$\iota$",
        "d_L": r"$d_L$", "t_c": r"$t_c$", "phase_c": r"$\phi_c$",
        "psi": r"$\psi$", "ra": r"$\alpha$", "dec": r"$\delta$",
        "tau": r"$\tau$",
    }
    samples = NestedSamples(
        phys_particles, logL=final_state.loglikelihood,
        logL_birth=logL_birth, labels=col_labels,
        logzero=jnp.nan, dtype=jnp.float64,
    )
    samples.to_csv(f"{out_path}.csv")
    logZ = samples.logZ(1000).mean()
    print(f"  [idx={IDX}] {mode_name}  logZ={logZ:.2f}  →  {out_path}.csv")

# ---------------------------------------------------------------------------
# Run all four modes
# ---------------------------------------------------------------------------
MODES = [('norm', 'true'), ('norm', 'welch'), ('alcs', 'true'), ('alcs', 'welch')]

for lm, pl in MODES:
    print(f"\n{'='*55}")
    print(f"  [idx={IDX}]  likelihood={lm}  PSD={pl}")
    print(f"{'='*55}")
    run_one_mode(lm, pl)
