"""
Compute a per-bin notch mask for GW150914 based on the data power |d|^2/S0.

Bins where |d|^2/S0 exceeds a threshold (default 20) are flagged as spectral
lines and set to 0.0 in the mask; all other bins are 1.0.

Using the raw data power (no waveform subtraction) makes the mask independent
of the signal model.  The GW150914 signal peaks around 100-200 Hz and
contributes only a handful of nats above the noise floor there; the spectral
lines at ~518 Hz (L1) and ~991/1000/1013 Hz (H1/L1) are orders of magnitude
larger and show up unambiguously.

Usage:
    cd src/
    python compute_notch_mask.py [--threshold 20] [--suffix ""]

    e.g. python compute_notch_mask.py --threshold 10 --suffix _t10

Outputs:
    gw150914_notch_mask_H1{suffix}.npy  (float64, 0.0/1.0, length = N_freq_bins)
    gw150914_notch_mask_L1{suffix}.npy
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=20.0,
                    help="|d|^2/S0 threshold above which bin is notched (default 20)")
parser.add_argument("--suffix", type=str, default="",
                    help="suffix appended to output filenames, e.g. _t10 (default: empty, overwrites standard mask)")
args = parser.parse_args()

frequencies = np.load("gw150914_frequencies.npy").astype(np.float64)
df = float(frequencies[1] - frequencies[0])

for det_name in ["H1", "L1"]:
    d  = np.load(f"gw150914_{det_name}_strain.npy")   # complex128
    S0 = np.load(f"gw150914_{det_name}_psd.npy")      # float64

    data_power = np.abs(d)**2 / S0                    # expected ~1 per bin

    mask = (data_power <= args.threshold).astype(np.float64)

    n_notched = int((mask == 0).sum())
    print(f"{det_name}: {len(mask)} bins total, {n_notched} notched "
          f"(|d|²/S₀ > {args.threshold}), {int(mask.sum())} retained")

    # Report the worst offenders
    top = np.argsort(data_power)[-8:][::-1]
    for i in top:
        flag = "NOTCHED" if mask[i] == 0.0 else "kept"
        print(f"  f={frequencies[i]:.2f} Hz  |d|²/S₀={data_power[i]:.1f}  [{flag}]")

    np.save(f"gw150914_notch_mask_{det_name}{args.suffix}.npy", mask)
    print(f"  Saved gw150914_notch_mask_{det_name}{args.suffix}.npy\n")
