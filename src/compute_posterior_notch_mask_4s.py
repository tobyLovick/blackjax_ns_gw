"""
Compute a posterior-informed notch mask for the 4s injection data.

Averages the normalised residual power B_k = 2*df*|d_k - h_k|^2/S_k over
N posterior draws.  Bins where the mean B_k exceeds the threshold are flagged.

Usage:
    python compute_posterior_notch_mask_4s.py [--n-samples 200] [--threshold 6] [--suffix _posterior]
    python compute_posterior_notch_mask_4s.py --samples-file blackjaxns_alcs_4s.csv

Outputs:
    4s_notch_mask_H1{suffix}.npy
    4s_notch_mask_L1{suffix}.npy
    4s_notch_mask_V1{suffix}.npy
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from astropy.time import Time
from anesthetic import read_chains
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.waveform import RippleIMRPhenomD

jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=200)
parser.add_argument("--threshold", type=float, default=6.0)
parser.add_argument("--suffix", type=str, default="_posterior")
parser.add_argument("--samples-file", type=str, default="../blackjaxns_nlive1400.csv")
args = parser.parse_args()

asd_paths = {"H1": "aLIGO_O4_high_asd.txt", "L1": "aLIGO_O4_high_asd.txt", "V1": "AdV_asd.txt"}
det_names = ["H1", "L1", "V1"]
det_objs   = [H1,   L1,   V1]

frequencies_full = np.load("4s_frequency_array.npy").astype(np.float64)
freq_mask_arr    = (frequencies_full >= 20.0) & (frequencies_full <= 1024.0)
frequencies      = frequencies_full[freq_mask_arr]
df               = float(frequencies[1] - frequencies[0])
freqs_jax        = jnp.array(frequencies)

det_data = {}
for name in det_names:
    f_asd, asd_vals = np.loadtxt(asd_paths[name], unpack=True)
    S0 = np.interp(frequencies, f_asd, asd_vals**2)
    det_data[name] = {
        "d": np.load(f"4s_{name}_strain.npy")[freq_mask_arr].astype(complex),
        "S": S0,
    }

waveform = RippleIMRPhenomD(f_ref=50)
epoch = 4 - 2
gmst  = Time(1126259642.413, format="gps").sidereal_time("apparent", "greenwich").rad

for det_obj, name in zip(det_objs, det_names):
    det_obj.frequencies = freqs_jax
    det_obj.data = jnp.array(det_data[name]["d"], dtype=jnp.complex128)
    det_obj.psd  = jnp.array(det_data[name]["S"])

print(f"Loading posterior samples from {args.samples_file} ...")
samples = read_chains(args.samples_file)
rows = samples.sample(args.n_samples)
print(f"Averaging over {args.n_samples} posterior draws ...")

mean_B = {name: np.zeros(len(frequencies)) for name in det_names}

for i, (_, row) in enumerate(rows.iterrows()):
    p = {k: float(row[k]) for k in
         ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "phase_c", "psi", "ra", "dec"]}
    p["gmst"] = gmst
    p["eta"]  = p["q"] / (1 + p["q"]) ** 2

    align_time   = jnp.exp(-1j * 2 * jnp.pi * freqs_jax * (epoch + p["t_c"]))
    waveform_sky = waveform(freqs_jax, p)

    for det_obj, name in zip(det_objs, det_names):
        h = np.array(det_obj.fd_response(freqs_jax, waveform_sky, p) * align_time)
        r = det_data[name]["d"] - h
        mean_B[name] += 2 * df * np.abs(r)**2 / det_data[name]["S"]

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{args.n_samples}")

for name in det_names:
    mean_B[name] /= args.n_samples

for name in det_names:
    B    = mean_B[name]
    mask = (B <= args.threshold).astype(np.float64)
    n_notched = int((mask == 0).sum())
    print(f"\n{name}: {n_notched} bins notched (mean B_res > {args.threshold})")
    top = np.argsort(B)[-8:][::-1]
    for i in top:
        flag = "NOTCHED" if mask[i] == 0 else "kept"
        print(f"  f={frequencies[i]:.2f} Hz  mean_B={B[i]:.2f}  [{flag}]")
    fname = f"4s_notch_mask_{name}{args.suffix}.npy"
    np.save(fname, mask)
    print(f"  Saved {fname}")
