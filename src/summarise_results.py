"""
Summarise nested sampling results into a compact .npy table.

Reads results/{event}/{mode}.csv for every event in the catalog and every mode,
extracts the statistics needed for the paper tables, and saves a single summary.

Usage:
    python summarise_results.py [--catalog o3_catalog.npy] [--out summary.npy]

Output .npy is a dict keyed by event name, each value a dict keyed by mode,
each containing:
    logZ        float   log evidence (mean over bootstrap)
    logZ_err    float   log evidence uncertainty (std over bootstrap)
    tau_mean    float   posterior mean of tau         (ALCS/ALCS+BAD only)
    tau_std     float   posterior std of tau          (ALCS/ALCS+BAD only)
    panom_mean  float   posterior mean of p_anom      (BAD/ALCS+BAD only)
    panom_std   float   posterior std of p_anom       (BAD/ALCS+BAD only)
    Mc_mean     float   posterior mean of chirp mass
    Mc_std      float   posterior std of chirp mass
    dL_mean     float   posterior mean of luminosity distance
    dL_std      float   posterior std of luminosity distance

Missing runs are stored as NaN.
"""

import argparse
import os
import numpy as np
import warnings
from anesthetic import read_chains

MODES = ['norm', 'bad', 'alcs_quad', 'alcs_bad_quad']

parser = argparse.ArgumentParser()
parser.add_argument('--catalog', default='o3_catalog.npy')
parser.add_argument('--results', default='results')
parser.add_argument('--out',     default='summary.npy')
parser.add_argument('--nboot',   type=int, default=100,
                    help='Bootstrap samples for logZ uncertainty (default 100)')
args = parser.parse_args()

catalog = np.load(args.catalog, allow_pickle=True).item()

N_BINS = 8000  # approximate number of frequency bins per detector

summary = {}

for event_name, cfg in catalog.items():
    ev_key = event_name.lower().replace('-', '_')
    event_summary = {}

    for mode in MODES:
        nan = float('nan')
        row = dict(
            logZ=nan, logZ_err=nan,
            tau_mean=nan, tau_std=nan,
            panom_mean=nan, panom_std=nan,
            Mc_mean=nan, Mc_std=nan,
            dL_mean=nan, dL_std=nan,
        )

        csv_path = os.path.join(args.results, ev_key, f'{mode}.csv')
        if not os.path.exists(csv_path):
            print(f'  MISSING  {event_name:20s}  {mode}')
            event_summary[mode] = row
            continue

        try:
            root = csv_path[:-4] if csv_path.endswith('.csv') else csv_path
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                chains = read_chains(root)

            logZ_samples = chains.logZ(args.nboot)
            row['logZ']     = float(logZ_samples.mean())
            row['logZ_err'] = float(logZ_samples.std())

            if 'M_c' in chains.columns:
                row['Mc_mean'] = float(chains['M_c'].mean())
                row['Mc_std']  = float(chains['M_c'].std())

            if 'd_L' in chains.columns:
                row['dL_mean'] = float(chains['d_L'].mean())
                row['dL_std']  = float(chains['d_L'].std())

            if 'tau' in chains.columns:
                row['tau_mean'] = float(chains['tau'].mean())
                row['tau_std']  = float(chains['tau'].std())

            if 'p_anom' in chains.columns:
                row['panom_mean'] = float(chains['p_anom'].mean())
                row['panom_std']  = float(chains['p_anom'].std())

            print(f'  OK       {event_name:20s}  {mode:14s}  '
                  f'logZ={row["logZ"]:10.2f} ± {row["logZ_err"]:.2f}')

        except Exception as exc:
            print(f'  ERROR    {event_name:20s}  {mode}  —  {exc}')

        event_summary[mode] = row

    summary[event_name] = event_summary

np.save(args.out, summary, allow_pickle=True)
print(f'\nSaved {args.out}')
print('Load with: summary = np.load("summary.npy", allow_pickle=True).item()')
