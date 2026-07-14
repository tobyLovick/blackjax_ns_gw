"""
Quick diagnostic: compare Welch vs BayesWave PSDs for a given event.
Prints ratio statistics and flags obvious problems (wrong units, ASD vs PSD, etc.)

Usage:
  python diagnose_psd.py --event-idx 0 --event-list o4_event_list.json
"""

import os, json, argparse
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--event-idx',  type=int, default=0)
    p.add_argument('--event-list', type=str, default='o4_event_list.json')
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.event_list) as f:
        events = json.load(f)
    ev = events[args.event_idx]
    name     = ev['name']
    data_dir = os.path.join('o4_data', name)

    print(f"\nEvent: {name}  (catalog: {ev['catalog']})")
    print(f"GPS:   {ev['gps']}")

    freqs = np.load(os.path.join(data_dir, 'frequencies.npy'))
    print(f"\nFrequency grid: {freqs[0]:.2f}--{freqs[-1]:.2f} Hz  "
          f"({len(freqs)} bins, df={freqs[1]-freqs[0]:.4f} Hz)")

    with open(os.path.join(data_dir, 'detectors.json')) as f:
        dets = json.load(f)

    for det in dets:
        welch = np.load(os.path.join(data_dir, f'{det}_psd_welch.npy'))
        bw    = np.load(os.path.join(data_dir, f'{det}_psd_bw.npy'))

        ratio = bw / welch

        print(f"\n--- {det} ---")
        print(f"  Welch PSD  median={np.median(welch):.3e}  "
              f"min={welch.min():.3e}  max={welch.max():.3e}")
        print(f"  BW PSD     median={np.median(bw):.3e}  "
              f"min={bw.min():.3e}  max={bw.max():.3e}")
        print(f"  BW/Welch   median={np.median(ratio):.3f}  "
              f"min={ratio.min():.3f}  max={ratio.max():.3f}")

        # Check if BW looks like ASD (sqrt of PSD)
        asd_ratio = np.sqrt(bw) / welch
        print(f"  sqrt(BW)/Welch  median={np.median(asd_ratio):.3f}  "
              f"(~1 → BW is stored as ASD not PSD!)")

        # Per-bin logL contribution difference (norm mode)
        df = freqs[1] - freqs[0]
        log_two_df_over_pi = np.log(2.0 * df / np.pi)
        strain = np.load(os.path.join(data_dir, f'{det}_strain.npy'))
        h = np.zeros_like(strain)  # null template — just checks PSD contribution
        B = 2.0 * df * np.abs(strain - h)**2
        logL_welch = log_two_df_over_pi - np.log(welch) - B / welch
        logL_bw    = log_two_df_over_pi - np.log(bw)    - B / bw
        print(f"  sum(logL) Welch={logL_welch.sum():.1f}  BW={logL_bw.sum():.1f}  "
              f"delta={logL_bw.sum()-logL_welch.sum():.1f} nats  (null template)")

if __name__ == '__main__':
    main()
