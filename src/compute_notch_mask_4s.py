"""
Compute a per-bin notch mask for the 4s injection data based on |d|^2/S0.

Usage:
    python compute_notch_mask_4s.py [--threshold 20] [--suffix ""]
    python compute_notch_mask_4s.py --threshold 10 --suffix _t10

Outputs:
    4s_notch_mask_H1{suffix}.npy
    4s_notch_mask_L1{suffix}.npy
    4s_notch_mask_V1{suffix}.npy
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=20.0,
                    help="|d|^2/S0 threshold above which bin is notched (default 20)")
parser.add_argument("--suffix", type=str, default="",
                    help="suffix appended to output filenames, e.g. _t10")
args = parser.parse_args()

asd_paths = {
    "H1": "aLIGO_O4_high_asd.txt",
    "L1": "aLIGO_O4_high_asd.txt",
    "V1": "AdV_asd.txt",
}

frequencies_full = np.load("4s_frequency_array.npy").astype(np.float64)
freq_mask = (frequencies_full >= 20.0) & (frequencies_full <= 1024.0)
frequencies = frequencies_full[freq_mask]
df = float(frequencies[1] - frequencies[0])

for det_name in ["H1", "L1", "V1"]:
    d = np.load(f"4s_{det_name}_strain.npy")[freq_mask].astype(complex)

    f_asd, asd_vals = np.loadtxt(asd_paths[det_name], unpack=True)
    S0 = np.interp(frequencies, f_asd, asd_vals**2)

    data_power = np.abs(d)**2 / S0

    mask = (data_power <= args.threshold).astype(np.float64)
    n_notched = int((mask == 0).sum())
    print(f"{det_name}: {len(mask)} bins, {n_notched} notched (|d|²/S₀ > {args.threshold})")

    top = np.argsort(data_power)[-5:][::-1]
    for i in top:
        flag = "NOTCHED" if mask[i] == 0.0 else "kept"
        print(f"  f={frequencies[i]:.2f} Hz  |d|²/S₀={data_power[i]:.2f}  [{flag}]")

    np.save(f"4s_notch_mask_{det_name}{args.suffix}.npy", mask)
    print(f"  Saved 4s_notch_mask_{det_name}{args.suffix}.npy\n")
