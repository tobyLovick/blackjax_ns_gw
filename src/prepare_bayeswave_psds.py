"""
Convert GWTC-1 BayesWave PSD .dat files → .npy on the analysis frequency grid.

The .dat files (from https://dcc.ligo.org/LIGO-P1900011/public) contain the
BayesWave median PSD used in GWTC-1 parameter estimation, at 0.125 Hz resolution.
We interpolate to the 0.25 Hz analysis grid (4s FFT) and save as
  {event_lower}_H1_psd_bayeswave.npy
  {event_lower}_L1_psd_bayeswave.npy

Usage:
    python prepare_bayeswave_psds.py
    python prepare_bayeswave_psds.py --psd-dir /path/to/GWTC1_PSDs
"""

import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--psd-dir', default='../../GWTC1_PSDs',
                    help='Directory containing GWTC1_GW*_PSDs.dat files')
args = parser.parse_args()

PSD_DIR = Path(args.psd_dir)

# Analysis frequency grid — same for all O1/O2 events (4s, 20–1024 Hz, Δf=0.25 Hz).
# GW150914 is used as reference; all events share this grid.
freqs_target = np.load('gw150914_frequencies.npy')
print(f"Target grid: {len(freqs_target)} bins, "
      f"{freqs_target[0]:.2f}–{freqs_target[-1]:.2f} Hz, "
      f"Δf={freqs_target[1]-freqs_target[0]:.4f} Hz")

EVENTS = {
    'GW150914': 'gw150914',
    'GW151226': 'gw151226',
    'GW170104': 'gw170104',
    'GW170608': 'gw170608',
    'GW170809': 'gw170809',
    'GW170814': 'gw170814',
    'GW170823': 'gw170823',
}

for event, name in EVENTS.items():
    dat_path = PSD_DIR / f'GWTC1_{event}_PSDs.dat'
    if not dat_path.exists():
        print(f"  MISSING: {dat_path}")
        continue

    # Column order is always: Freq, LIGO_Hanford_PSD, LIGO_Livingston_PSD [, Virgo_PSD]
    data = np.loadtxt(dat_path, comments='#')
    bw_freqs = data[:, 0]
    bw_H1    = data[:, 1]
    bw_L1    = data[:, 2]

    # BayesWave PSDs are smooth — linear interpolation in log-log is fine.
    # Extrapolate at edges using the boundary values (flat extrapolation).
    for ifo, bw_psd in [('H1', bw_H1), ('L1', bw_L1)]:
        log_psd_interp = np.interp(
            freqs_target,
            bw_freqs,
            np.log(bw_psd),
            left=np.log(bw_psd[0]),    # flat extrapolation below BW range
            right=np.log(bw_psd[-1]),  # flat extrapolation above BW range
        )
        psd_interp = np.exp(log_psd_interp)
        out = f'{name}_{ifo}_psd_bayeswave.npy'
        np.save(out, psd_interp)

    # Sanity check: compare to Welch at a few representative frequencies
    welch_H1 = np.load(f'{name}_H1_psd.npy') if Path(f'{name}_H1_psd.npy').exists() else None
    bw_H1_interp = np.exp(np.interp(freqs_target, bw_freqs, np.log(bw_H1),
                                     left=np.log(bw_H1[0]), right=np.log(bw_H1[-1])))
    bw_range = f"{bw_freqs[0]:.1f}–{bw_freqs[-1]:.1f} Hz"
    ratio_str = ""
    if welch_H1 is not None:
        ratio = bw_H1_interp / welch_H1
        ratio_str = f"  BW/Welch H1: median={np.median(ratio):.3f}, range=[{ratio.min():.3f}, {ratio.max():.3f}]"
    print(f"{event}: BW grid {bw_range} ({len(bw_freqs)} pts) → {len(freqs_target)} analysis bins{ratio_str}")

print("\nDone. Files saved in current directory.")
