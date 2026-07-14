#!/usr/bin/env python3
"""
Summarize all completed O4 NS runs into a compact JSON for population analysis.

Run on Isambard login node (read-only, ~10-20 min):
  cd ~/GW/src
  python summarize_o4_results.py

Output: o4_summary.json  (~200-400 kB, safe to scp locally)
"""

import os, json, sys, time
import numpy as np

RESULTS_DIR = 'o4_results'
EVENT_LIST  = 'o4_event_list.json'
MODES       = ['norm_welch', 'norm_bw', 'alcs_welch', 'alcs_bw']
N_BINS      = 1969   # frequency bins at F_HIGH=512 Hz
OUT_FILE    = 'o4_summary.json'


def summarize_mode(csv_path, mode_name):
    from anesthetic import read_chains
    try:
        ch = read_chains(csv_path)
    except Exception as e:
        print(f"      ERROR: {e}")
        return None

    result = {}

    # Log evidence — matches working injection code pattern
    try:
        result['logZ']     = float(ch.logZ())
        result['logZ_err'] = float(ch.logZ(nsamples=200).std())
    except Exception as e:
        print(f"      logZ failed: {e}")
        result['logZ'] = None

    # GW parameter posteriors — .mean()/.std() are weighted in anesthetic
    for col, key in [('M_c', 'Mc'), ('d_L', 'dL')]:
        if col in ch.columns:
            result[f'{key}_mean']   = float(ch[col].mean())
            result[f'{key}_std']    = float(ch[col].std())
            result[f'{key}_median'] = float(ch[col].median())

    # ALCS hyperparameters
    if 'alcs' in mode_name:
        for col, key in [('tau', 'tau'), ('p_anom', 'p_anom')]:
            if col not in ch.columns:
                continue

            s = ch[col]
            mean   = float(s.mean())
            std    = float(s.std())
            median = float(s.median())

            # Percentiles via anesthetic's weighted quantile
            try:
                pcts = s.quantile([0.05, 0.16, 0.84, 0.95])
                p5, p16, p84, p95 = [float(pcts[q]) for q in [0.05, 0.16, 0.84, 0.95]]
            except Exception:
                # fallback: unweighted (good approximation at n_live=1400)
                vals = np.array(s)
                p5, p16, p84, p95 = np.percentile(vals, [5, 16, 84, 95])
                p5, p16, p84, p95 = float(p5), float(p16), float(p84), float(p95)

            # Skewness and excess kurtosis from posterior draws
            try:
                samps = np.array(ch.sample(2000)[col])
                z = (samps - mean) / std if std > 0 else np.zeros_like(samps)
                skew   = float(np.mean(z**3))
                exkurt = float(np.mean(z**4)) - 3.0
            except Exception:
                skew, exkurt = float('nan'), float('nan')

            result[f'{key}_mean']     = mean
            result[f'{key}_std']      = std
            result[f'{key}_median']   = median
            result[f'{key}_p5']       = p5
            result[f'{key}_p16']      = p16
            result[f'{key}_p84']      = p84
            result[f'{key}_p95']      = p95
            result[f'{key}_skewness'] = skew
            result[f'{key}_exkurt']   = exkurt

        # Expected anomalous bins
        if 'p_anom_mean' in result:
            result['N_anom_mean'] = N_BINS * result['p_anom_mean']

        # Flag posteriors near prior boundaries
        result['tau_near_upper'] = bool(result.get('tau_p95', 0.0) > 1.8)  # prior ceil = 2.0
        result['tau_near_lower'] = bool(result.get('tau_p5',  2.0) < 0.003) # prior floor = 0.001

    return result


def main():
    t_start = time.time()

    with open(EVENT_LIST) as f:
        events = json.load(f)
    print(f"Loaded {len(events)} events from {EVENT_LIST}")

    summary = []
    n_complete = 0

    for i, ev in enumerate(events):
        name    = ev['name']
        ev_dir  = os.path.join(RESULTS_DIR, name)

        # Check which modes have completed
        mode_csvs = {m: os.path.join(ev_dir, f'{m}.csv') for m in MODES}
        available = {m: p for m, p in mode_csvs.items() if os.path.exists(p)}

        if not available:
            continue

        print(f"[{i:3d}] {name}  ({len(available)}/4 modes)", flush=True)

        ev_result = {
            'idx':     i,
            'name':    name,
            'catalog': ev['catalog'],
            'gps':     ev['gps'],
            'snr':     ev.get('snr', None),
            'Mc_catalog': ev.get('Mc_det_median') or ev.get('Mc_src_median'),
            'dL_catalog': ev.get('dL_median'),
            'modes':   {},
        }

        for mode, csv_path in available.items():
            t0 = time.time()
            ev_result['modes'][mode] = summarize_mode(csv_path, mode)
            dt = time.time() - t0
            logZ_str = (f"logZ={ev_result['modes'][mode]['logZ']:.1f}"
                        if ev_result['modes'][mode] and ev_result['modes'][mode]['logZ'] is not None
                        else "logZ=?")
            print(f"      {mode:12s}  {logZ_str}  ({dt:.1f}s)", flush=True)

        # Derived Bayes factors
        m = ev_result['modes']
        def lnB(a, b):
            ma, mb = m.get(a), m.get(b)
            if ma and mb and ma['logZ'] is not None and mb['logZ'] is not None:
                return round(ma['logZ'] - mb['logZ'], 2)
            return None

        ev_result['lnB'] = {
            'alcs_bw_vs_norm_bw':       lnB('alcs_bw',    'norm_bw'),    # fiducial ALCS BF
            'alcs_welch_vs_norm_welch':  lnB('alcs_welch', 'norm_welch'), # ALCS BF on Welch
            'alcs_welch_vs_norm_bw':     lnB('alcs_welch', 'norm_bw'),    # ALCS welch vs BW norm
            'alcs_bw_vs_alcs_welch':     lnB('alcs_bw',    'alcs_welch'), # BW improvement inside ALCS
            'norm_bw_vs_norm_welch':     lnB('norm_bw',    'norm_welch'), # BW vs Welch (sanity)
        }

        # Parameter shifts: (ALCS_bw - norm_bw) / sigma_norm_bw
        def sigma_shift(col, mode_a='alcs_bw', mode_b='norm_bw'):
            ma, mb = m.get(mode_a), m.get(mode_b)
            if not ma or not mb: return None
            mu_a  = ma.get(f'{col}_mean')
            mu_b  = mb.get(f'{col}_mean')
            sig_b = mb.get(f'{col}_std')
            if None in (mu_a, mu_b, sig_b) or sig_b == 0: return None
            return round((mu_a - mu_b) / sig_b, 4)

        ev_result['sigma_shifts'] = {
            'Mc_alcs_bw_vs_norm_bw':    sigma_shift('Mc'),
            'dL_alcs_bw_vs_norm_bw':    sigma_shift('dL'),
            'Mc_alcs_welch_vs_norm_bw': sigma_shift('Mc', 'alcs_welch', 'norm_bw'),
            'dL_alcs_welch_vs_norm_bw': sigma_shift('dL', 'alcs_welch', 'norm_bw'),
        }

        summary.append(ev_result)
        n_complete += 1

    with open(OUT_FILE, 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone: {n_complete} events in {elapsed/60:.1f} min → {OUT_FILE}")
    print(f"File size: {os.path.getsize(OUT_FILE)/1e3:.1f} kB")

    # Quick sanity summary
    lnBs = [e['lnB']['alcs_bw_vs_norm_bw'] for e in summary
            if e['lnB']['alcs_bw_vs_norm_bw'] is not None]
    if lnBs:
        lnBs = np.array(lnBs)
        print(f"\nln B(alcs_bw vs norm_bw):  "
              f"median={np.median(lnBs):.1f}  "
              f"mean={np.mean(lnBs):.1f}  "
              f"frac>0={np.mean(lnBs>0):.0%}  "
              f"frac>3={np.mean(lnBs>3):.0%}")


if __name__ == '__main__':
    main()
