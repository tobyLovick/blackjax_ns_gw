"""
4-way ALCS analysis for O4 events: norm_welch / alcs_welch / norm_bw / alcs_bw.

Usage:
  python blackjaxns_o4_analysis.py --event-idx 0 --event-list o4_event_list.json

Downloads strain + PSDs on demand (if not already cached in o4_data/{event_name}/).
Writes results to o4_results/{event_name}/{mode}.csv

Skips any mode whose CSV already exists.
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse, json, pickle, sys, time, random
import numpy as np
import jax
import jax.numpy as jnp
import tqdm
from astropy.time import Time
from anesthetic import NestedSamples, read_chains
from blackjax.ns.utils import finalise

jax.config.update("jax_enable_x64", True)

from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.waveform import RippleIMRPhenomXPHM
from custom_kernels import (
    acceptance_walk_sampler,
    create_unit_cube_functions,
    init_unit_cube_particles,
    transform_to_physical,
)

DETECTOR_OBJECTS = {'H1': H1, 'L1': L1, 'V1': V1}

MODES = ['norm_welch', 'norm_bw', 'alcs_welch', 'alcs_bw']

_gh_x_np, _gh_w_np = np.polynomial.hermite.hermgauss(20)
GH_X    = jnp.array(_gh_x_np)
GH_LOGW = jnp.log(jnp.array(_gh_w_np))
LOG_DELTA = jnp.log(4e-38)

# ---------------------------------------------------------------------------
# Per-bin likelihood kernels (identical to existing scripts)
# ---------------------------------------------------------------------------
def _norm_bin(B_i, log_S_i, log_two_df_over_pi):
    return log_two_df_over_pi - log_S_i - B_i * jnp.exp(-log_S_i)

def _alcs_bin(B_i, log_S_i, tau, log_two_df_over_pi):
    tau_sq = tau ** 2
    s0 = 0.5 * log_S_i
    s  = s0
    for _ in range(5):
        e2s = jnp.exp(-2.0 * s)
        s   = s - (-2.0 + 2.0*B_i*e2s - (s - s0)/tau_sq) / (-4.0*B_i*e2s - 1.0/tau_sq)
    H     = 4.0*B_i*jnp.exp(-2.0*s) + 1.0/tau_sq
    scale = jnp.sqrt(2.0 / H)
    s_nodes = s + scale * GH_X
    g_nodes = (-2.0*s_nodes - B_i*jnp.exp(-2.0*s_nodes) - (s_nodes - s0)**2 / (2.0*tau_sq))
    log_int = jnp.log(scale) + jax.scipy.special.logsumexp(GH_LOGW + g_nodes + GH_X**2)
    return log_two_df_over_pi - 0.5*jnp.log(2.0*jnp.pi*tau_sq) + log_int

def _bad_combine(log_Z_i, p_anom):
    return jnp.logaddexp(jnp.log1p(-p_anom) + log_Z_i, jnp.log(p_anom) - LOG_DELTA)

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
# Data preparation (inline, so nodes don't need a separate prep step)
# ---------------------------------------------------------------------------
def prepare_data_if_needed(event_cfg):
    """Download strain + PSDs for this event if not already cached."""
    import o4_data_prep as prep

    name     = event_cfg['name']
    data_dir = os.path.join('o4_data', name)

    if os.path.exists(os.path.join(data_dir, 'detectors.json')):
        print(f"  Data already cached for {name}")
        return
    if os.path.exists(os.path.join(data_dir, 'INSUFFICIENT_DETECTORS')):
        return

    # Stagger 167 simultaneous job starts so GWOSC/Zenodo aren't hammered
    delay = random.uniform(0, 120)
    print(f"  Staggering download by {delay:.0f}s ...", flush=True)
    time.sleep(delay)

    os.makedirs(data_dir, exist_ok=True)
    freqs_analysis, freqs_mask = prep.make_analysis_freqs()

    freq_path = os.path.join(data_dir, 'frequencies.npy')
    if not os.path.exists(freq_path):
        np.save(freq_path, freqs_analysis)

    gps       = event_cfg['gps']
    gps_start = gps - prep.DOWNLOAD_DURATION / 2
    gps_end   = gps + prep.DOWNLOAD_DURATION / 2
    active_dets = []

    for det in prep.DETECTORS:
        strain_path = os.path.join(data_dir, f'{det}_strain.npy')
        welch_path  = os.path.join(data_dir, f'{det}_psd_welch.npy')
        if os.path.exists(strain_path) and os.path.exists(welch_path):
            active_dets.append(det)
            continue
        ts = prep.download_strain(det, gps_start, gps_end)
        if ts is None:
            continue
        np.save(strain_path, prep.extract_on_source_fft(ts, gps, freqs_mask))
        np.save(welch_path,  prep.compute_welch_psd(ts, gps, freqs_analysis))
        active_dets.append(det)
        print(f"  {det}: strain + Welch saved", flush=True)

    if len(active_dets) < 2:
        print(f"  SKIP {name}: only {len(active_dets)} detector(s)")
        open(os.path.join(data_dir, 'INSUFFICIENT_DETECTORS'), 'w').close()
        return

    missing_bw = [d for d in active_dets
                  if not os.path.exists(os.path.join(data_dir, f'{d}_psd_bw.npy'))]
    if missing_bw:
        bw_psds, _ = prep.download_bw_psd(
            event_cfg['hdf5_url'], name, active_dets, freqs_analysis)
        for det, psd in bw_psds.items():
            np.save(os.path.join(data_dir, f'{det}_psd_bw.npy'), psd)
            print(f"  {det}: BayesWave PSD saved", flush=True)
        active_dets = [d for d in active_dets if d in bw_psds]

    with open(os.path.join(data_dir, 'detectors.json'), 'w') as f:
        json.dump(active_dets, f)
    print(f"  Data prep complete: {name}  detectors={active_dets}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--event-idx',  type=int,  required=True)
    p.add_argument('--event-list', type=str,  default='o4_event_list.json')
    p.add_argument('--n-live',     type=int,  default=1400)
    p.add_argument('--n-delete',   type=int,  default=700)
    p.add_argument('--modes',      type=str,  default=None,
                   help='Comma-separated subset of modes to run, e.g. norm_welch or norm_bw,alcs_bw')
    return p.parse_args()


def run_mode(mode, event_cfg, detectors, frequencies, gps, out_dir, n_live, n_delete):
    use_alcs = 'alcs' in mode
    psd_key  = 'bw' if 'bw' in mode else 'welch'

    # Load PSD for this mode
    for det in detectors:
        psd = np.load(os.path.join('o4_data', event_cfg['name'], f'{det.name}_psd_{psd_key}.npy'))
        det.psd = jnp.array(psd, dtype=jnp.float64)

    gmst  = Time(gps, format='gps').sidereal_time('apparent', 'greenwich').rad
    epoch = 2   # duration(4) - post_trigger(2)

    prior = event_cfg['prior']
    sample_keys = (["M_c","q","a_1","cos_tilt_1","phi_1","a_2","cos_tilt_2","phi_2",
                    "iota","d_L","t_c","psi","ra","dec","phase_c"]
                   + (["tau","p_anom"] if use_alcs else []))

    test_p = {k: jax.random.uniform(jax.random.PRNGKey(0), (100,)) for k in sample_keys}
    sample_keys = get_ravel_order(test_p)

    cfg_prior = {
        "M_c":     {"min": prior['Mc_lo'],  "max": prior['Mc_hi'],  "type": "uniform",     "wrap": False},
        "q":       {"min": 0.125,           "max": 1.0,             "type": "uniform",     "wrap": False},
        "a_1":       {"min": 0.0,  "max": 0.99,       "type": "uniform", "wrap": False},
        "cos_tilt_1":{"min": -1.0, "max": 1.0,        "type": "uniform", "wrap": False},
        "phi_1":     {"min": 0.0,  "max": 2*jnp.pi,   "type": "uniform", "wrap": True},
        "a_2":       {"min": 0.0,  "max": 0.99,       "type": "uniform", "wrap": False},
        "cos_tilt_2":{"min": -1.0, "max": 1.0,        "type": "uniform", "wrap": False},
        "phi_2":     {"min": 0.0,  "max": 2*jnp.pi,   "type": "uniform", "wrap": True},
        "iota":    {"min": 0.0,             "max": jnp.pi,          "type": "sine",        "wrap": False},
        "d_L":     {"min": prior['dL_lo'],  "max": prior['dL_hi'],  "type": "powerlaw",    "wrap": False},
        "t_c":     {"min": -0.1,            "max": 0.1,             "type": "uniform",     "wrap": False},
        "phase_c": {"min": 0.0,             "max": 2*jnp.pi,        "type": "uniform",     "wrap": True},
        "psi":     {"min": 0.0,             "max": jnp.pi,          "type": "uniform",     "wrap": True},
        "ra":      {"min": 0.0,             "max": 2*jnp.pi,        "type": "uniform",     "wrap": True},
        "dec":     {"min": -jnp.pi/2,       "max": jnp.pi/2,        "type": "cosine",      "wrap": False},
        "tau":     {"min": 0.001,           "max": 2.0,             "type": "log_uniform", "wrap": False},
        "p_anom":  {"min": 1e-4,            "max": 0.5,             "type": "log_uniform", "wrap": False},
    }
    sc = {k: cfg_prior[k] for k in sample_keys}

    type_arr = jnp.array([0 if sc[k]["type"]=="uniform"     else
                           1 if sc[k]["type"]=="sine"        else
                           2 if sc[k]["type"]=="cosine"      else
                           3 if sc[k]["type"]=="powerlaw"    else
                           5 for k in sample_keys])
    mins = jnp.array([sc[k]["min"] for k in sample_keys])
    maxs = jnp.array([sc[k]["max"] for k in sample_keys])

    @jax.jit
    def prior_transform_fn(u_params):
        u, _ = jax.flatten_util.ravel_pytree(u_params)
        x = jnp.where(type_arr == 0, mins + u*(maxs - mins),
            jnp.where(type_arr == 1, jnp.arccos(1 - 2*u),
            jnp.where(type_arr == 2, jnp.arcsin(2*u - 1),
            jnp.where(type_arr == 3, (mins**3 + u*(maxs**3 - mins**3))**(1/3),
                      mins * (maxs/mins)**u))))
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
                        -jnp.log(vals) - jnp.log(jnp.log(maxs/mins))))))
        return jnp.sum(lp)

    waveform = RippleIMRPhenomXPHM(f_ref=20)

    def loglikelihood_fn(params):
        p = dict(params)
        p["gmst"] = gmst
        p["eta"]  = p["q"] / (1 + p["q"])**2
        waveform_sky = waveform(frequencies, p)
        align_time   = jnp.exp(-1j*2*jnp.pi*frequencies*(epoch + p["t_c"]))
        df = frequencies[1] - frequencies[0]
        log_two_df_over_pi = jnp.log(2.0*df/jnp.pi)
        log_L = 0.0
        for det in detectors:
            h    = det.fd_response(frequencies, waveform_sky, p) * align_time
            B    = 2.0*df * jnp.abs(det.data - h)**2
            logS = jnp.log(det.psd)
            if use_alcs:
                bins = jax.vmap(lambda b, ls: _bad_combine(
                    _alcs_bin(b, ls, p["tau"], log_two_df_over_pi),
                    p["p_anom"]))(B, logS)
            else:
                bins = jax.vmap(lambda b, ls: _norm_bin(b, ls, log_two_df_over_pi))(B, logS)
            log_L = log_L + jnp.sum(bins)
        return log_L

    example_params      = {k: 0. for k in sample_keys}
    rng_key             = jax.random.PRNGKey(10)
    rng_key, init_key   = jax.random.split(rng_key)
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
        k, subk  = jax.random.split(k)
        state, dead = nested_sampler.step(subk, state)
        return (state, k), dead

    def terminate(state):
        dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
        return jnp.isfinite(dlogz) and dlogz < 0.1

    dead = []
    with tqdm.tqdm(desc=f"{event_cfg['name']} {mode}", unit=" dead") as pbar:
        while not terminate(state):
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)

    final_state = finalise(state, dead)
    physical    = transform_to_physical(final_state.particles, prior_transform_fn)
    logL_birth  = jnp.where(jnp.isnan(final_state.loglikelihood_birth),
                             -jnp.inf, final_state.loglikelihood_birth)

    labels = {
        "M_c":"$\\mathcal{M}_c$","q":"$q$","d_L":"$d_L$","iota":"$\\iota$",
        "ra":"$\\alpha$","dec":"$\\delta$",
        "a_1":"$a_1$","cos_tilt_1":"$\\cos\\theta_1$","phi_1":"$\\phi_1$",
        "a_2":"$a_2$","cos_tilt_2":"$\\cos\\theta_2$","phi_2":"$\\phi_2$",
        "t_c":"$t_c$","psi":"$\\psi$","phase_c":"$\\phi_c$",
        "tau":"$\\tau$","p_anom":"$p_{\\rm anom}$",
    }
    samples = NestedSamples(
        physical, logL=final_state.loglikelihood,
        logL_birth=logL_birth, labels=labels,
        logzero=jnp.nan, dtype=jnp.float64,
    )
    csv_path = os.path.join(out_dir, f"{mode}.csv")
    samples.to_csv(csv_path)
    with open(os.path.join(out_dir, f"{mode}_state.pkl"), "wb") as f:
        pickle.dump(final_state, f)
    logZ = read_chains(csv_path).logZ()
    print(f"  [{mode}]  logZ = {logZ:.2f}")
    return logZ


def main():
    args = parse_args()
    with open(args.event_list) as f:
        events = json.load(f)
    event_cfg = events[args.event_idx]
    name      = event_cfg['name']
    gps       = event_cfg['gps']
    data_dir  = os.path.join('o4_data', name)
    out_dir   = os.path.join('o4_results_xphm', name)
    os.makedirs(out_dir, exist_ok=True)

    # Download strain + PSDs if not already cached
    prepare_data_if_needed(event_cfg)
    if os.path.exists(os.path.join(data_dir, 'INSUFFICIENT_DETECTORS')):
        print(f"SKIP {name}: insufficient detectors"); sys.exit(0)

    with open(os.path.join(data_dir, 'detectors.json')) as f:
        det_names = json.load(f)

    frequencies = jnp.array(np.load(os.path.join(data_dir, 'frequencies.npy')), dtype=jnp.float64)

    detectors = []
    for dname in det_names:
        det = DETECTOR_OBJECTS[dname]
        det.frequencies = frequencies
        det.data = jnp.array(
            np.load(os.path.join(data_dir, f'{dname}_strain.npy')), dtype=jnp.complex128)
        detectors.append(det)

    print(f"\n{'='*60}")
    print(f"  {name}  |  detectors: {det_names}  |  n_bins: {len(frequencies)}")
    print(f"  Mc_prior: [{event_cfg['prior']['Mc_lo']:.1f}, {event_cfg['prior']['Mc_hi']:.1f}]  "
          f"dL_prior: [{event_cfg['prior']['dL_lo']:.0f}, {event_cfg['prior']['dL_hi']:.0f}]")
    print(f"{'='*60}")

    modes_to_run = MODES
    if args.modes:
        requested = [m.strip() for m in args.modes.split(',')]
        modes_to_run = [m for m in MODES if m in requested]

    logZ = {}
    for mode in modes_to_run:
        csv_path = os.path.join(out_dir, f"{mode}.csv")
        if os.path.exists(csv_path):
            logZ[mode] = read_chains(csv_path).logZ()
            print(f"  [{mode}]  logZ = {logZ[mode]:.2f}  (cached)")
            continue

        # Check required data files exist for this mode
        psd_key = 'bw' if 'bw' in mode else 'welch'
        missing = [d for d in det_names
                   if not os.path.exists(os.path.join(data_dir, f'{d}_psd_{psd_key}.npy'))]
        if missing:
            print(f"  SKIP {mode}: missing PSD for {missing}")
            continue

        logZ[mode] = run_mode(mode, event_cfg, detectors, frequencies, gps, out_dir,
                              args.n_live, args.n_delete)

    # Summary
    if 'norm_welch' in logZ:
        ref = logZ['norm_welch']
        print(f"\n  Bayes factors vs norm_welch:")
        for m, z in logZ.items():
            if m != 'norm_welch':
                print(f"    {m:15s}  {z - ref:+.1f} nats")


if __name__ == '__main__':
    main()
