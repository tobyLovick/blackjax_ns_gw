"""
Compute PP-test quantiles from completed NS chains on Isambard.

Run after all array jobs complete:
    python pp_summarise.py [--results pp_results] [--catalog pp_catalog.npy]
                           [--out pp_summary.npy]

For each accepted event and each of the 4 modes, computes the quantile of
the true parameter value in the posterior (fraction of posterior mass below
the truth). Under a calibrated posterior these quantiles are uniform on [0,1].

Output:
    pp_summary.npy  — dict with keys:
        'indices'   : (N_done,) int   — which catalog events were processed
        'quantiles' : (N_done, 4, 7) float  — quantile[event, mode, param]
        'tau_median': (N_done, 2) float     — median tau for alcs_true, alcs_welch
        'logZ'      : (N_done, 4) float     — logZ for each mode
        'modes'     : list of str           — mode names (length 4)
        'params'    : list of str           — parameter names (length 7)
"""

import argparse
import os
import numpy as np
from anesthetic import read_chains

parser = argparse.ArgumentParser()
parser.add_argument('--results', default='pp_results')
parser.add_argument('--catalog', default='pp_catalog.npy')
parser.add_argument('--out',     default='pp_summary.npy')
parser.add_argument('--n',       type=int, default=500,
                    help='Number of catalog events')
args = parser.parse_args()

MODES      = ['norm_true', 'norm_welch', 'alcs_true', 'alcs_welch']
PP_PARAMS  = ['M_c', 'q', 'iota', 'd_L', 't_c', 'phase_c', 'psi']

catalog = np.load(args.catalog, allow_pickle=True)

def get_quantile(samples, weights, true_val):
    """Fraction of posterior mass strictly below true_val."""
    idx = np.argsort(samples)
    cdf = np.cumsum(weights[idx])
    cdf /= cdf[-1]
    return float(np.interp(true_val, samples[idx], cdf))

indices    = []
quantiles  = []
tau_median = []
logZ_all   = []

for i in range(args.n):
    event_dir = os.path.join(args.results, str(i))
    # Check all 4 chains exist
    if not all(os.path.exists(os.path.join(event_dir, f"{m}.csv"))
               for m in MODES):
        continue

    try:
        q_event  = np.zeros((4, len(PP_PARAMS)))
        tau_ev   = np.zeros(2)
        logZ_ev  = np.zeros(4)

        true_vals = {k: float(catalog[i][k]) for k in PP_PARAMS}

        for mi, mode in enumerate(MODES):
            path = os.path.join(event_dir, mode)
            s    = read_chains(path)
            w    = s.get_weights()

            for pi, param in enumerate(PP_PARAMS):
                q_event[mi, pi] = get_quantile(
                    s[param].values, w, true_vals[param])

            if 'alcs' in mode:   # tau only sampled in alcs modes
                tau_idx = MODES.index(mode) - 2   # 0=alcs_true, 1=alcs_welch
                tau_ev[tau_idx] = float(np.average(s['tau'].values, weights=w))

            logZ_ev[mi] = float(s.logZ(1000).mean())

        indices.append(i)
        quantiles.append(q_event)
        tau_median.append(tau_ev)
        logZ_all.append(logZ_ev)

        if (len(indices) % 50) == 0:
            print(f"  Processed {len(indices)} events  (last: idx={i})")

    except Exception as e:
        print(f"  [warn] idx={i} failed: {e}")
        continue

N_done = len(indices)
print(f"\nDone: {N_done}/{args.n} events processed")

summary = {
    'indices':    np.array(indices,              dtype=int),
    'quantiles':  np.array(quantiles,            dtype=float),  # (N, 4, 7)
    'tau_median': np.array(tau_median,           dtype=float),  # (N, 2)
    'logZ':       np.array(logZ_all,             dtype=float),  # (N, 4)
    'modes':      MODES,
    'params':     PP_PARAMS,
}
np.save(args.out, summary, allow_pickle=True)
print(f"Saved {args.out}  ({N_done} events, shape {summary['quantiles'].shape})")
