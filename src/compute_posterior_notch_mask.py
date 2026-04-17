"""
Compute a posterior-informed notch mask for GW150914.

For each frequency bin, compute the mean residual power
    B_k = 2*df*|d_k - h_k(theta)|^2 / S_k
averaged over N posterior samples.  Under the signal model, clean bins
should average to ~1.  Bins that remain elevated after signal subtraction
are instrumental lines.

A 6-sigma threshold: P(B >= 6) = exp(-6) ~ 0.0025 per bin, so with
~4000 bins we expect ~10 false positives at threshold 6.  Adjust with
--threshold.

Usage:
    python compute_posterior_notch_mask.py [--n-samples 200] [--threshold 6] [--suffix _posterior]

Outputs:
    gw150914_notch_mask_H1{suffix}.npy
    gw150914_notch_mask_L1{suffix}.npy
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from astropy.time import Time
from anesthetic import read_chains
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.waveform import RippleIMRPhenomD

jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=200,
                    help="Number of posterior draws to average over (default 200)")
parser.add_argument("--threshold", type=float, default=6.0,
                    help="Mean B_k threshold above which bin is notched (default 6)")
parser.add_argument("--suffix", type=str, default="_posterior",
                    help="Suffix for output mask filenames (default _posterior)")
parser.add_argument("--samples-file", type=str,
                    default="blackjaxns_gw150914_notched.csv",
                    help="Fixed-PSD posterior samples CSV to draw from")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
freqs = np.load("gw150914_frequencies.npy").astype(np.float64)
df    = float(freqs[1] - freqs[0])
freqs_jax = jnp.array(freqs)

det_data = {}
for name in ["H1", "L1"]:
    det_data[name] = {
        "d": np.load(f"gw150914_{name}_strain.npy").astype(complex),
        "S": np.load(f"gw150914_{name}_psd.npy").astype(np.float64),
    }

waveform = RippleIMRPhenomD(f_ref=20)
post_trigger_duration = 2
epoch = 4 - post_trigger_duration
gps   = 1126259462.4
gmst  = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad

for jimdet, name in zip([H1, L1], ["H1", "L1"]):
    jimdet.frequencies = freqs_jax
    jimdet.data = jnp.array(det_data[name]["d"], dtype=jnp.complex128)
    jimdet.psd  = jnp.array(det_data[name]["S"])

# ---------------------------------------------------------------------------
# Draw posterior samples and accumulate mean residual power
# ---------------------------------------------------------------------------
print(f"Loading posterior samples from {args.samples_file} ...")
samples = read_chains(args.samples_file)
rows = samples.sample(args.n_samples)
print(f"Averaging residuals over {args.n_samples} posterior draws ...")

mean_B = {name: np.zeros(len(freqs)) for name in ["H1", "L1"]}

for i, (_, row) in enumerate(rows.iterrows()):
    p = {k: float(row[k]) for k in
         ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "phase_c", "psi", "ra", "dec"]}
    p["gmst"] = gmst
    p["eta"]  = p["q"] / (1 + p["q"]) ** 2

    align_time   = jnp.exp(-1j * 2 * jnp.pi * freqs_jax * (epoch + p["t_c"]))
    waveform_sky = waveform(freqs_jax, p)

    for jimdet, name in zip([H1, L1], ["H1", "L1"]):
        h = np.array(jimdet.fd_response(freqs_jax, waveform_sky, p) * align_time)
        r = det_data[name]["d"] - h
        mean_B[name] += 2 * df * np.abs(r) ** 2 / det_data[name]["S"]

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{args.n_samples}")

for name in ["H1", "L1"]:
    mean_B[name] /= args.n_samples

# ---------------------------------------------------------------------------
# Build and save masks
# ---------------------------------------------------------------------------
for name in ["H1", "L1"]:
    B    = mean_B[name]
    mask = (B <= args.threshold).astype(np.float64)

    n_notched = int((mask == 0).sum())
    print(f"\n{name}: {n_notched} bins notched (mean B_res > {args.threshold})")

    top = np.argsort(B)[-10:][::-1]
    for i in top:
        flag = "NOTCHED" if mask[i] == 0 else "kept"
        print(f"  f={freqs[i]:.2f} Hz  mean_B={B[i]:.2f}  [{flag}]")

    fname = f"gw150914_notch_mask_{name}{args.suffix}.npy"
    np.save(fname, mask)
    print(f"  Saved {fname}")
