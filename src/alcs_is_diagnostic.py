"""
Post-hoc IS diagnostic for the ALCS Laplace approximation on GW150914.

For each of M posterior samples of (theta, tau) drawn from the ALCS chain:
  - Compute per-bin MAP sigma_hat and Laplace Hessian H_j
  - Draw K samples from the Laplace proposal N(sigma_hat_j, 1/H_j)
  - Compute IS weights w^(k) = p(sigma^(k)) / q(sigma^(k))
  - Report per-bin ESS/K and the total IS correction to log L

Per-bin ESS/K ~ 1  => Laplace is a good proposal for each bin.
Total IS correction => systematic bias in the ALCS log-evidence estimate.

Usage:
    cd src/
    python alcs_is_diagnostic.py [--chain blackjaxns_alcs_gw150914.csv] [--M 20] [--K 2000]
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from anesthetic import read_chains
from astropy.time import Time
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.waveform import RippleIMRPhenomD

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--chain", default="blackjaxns_alcs_gw150914.csv")
parser.add_argument("--M", type=int, default=20, help="posterior samples to evaluate at")
parser.add_argument("--K", type=int, default=2000, help="IS draws per bin per sample")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
waveform = RippleIMRPhenomD(f_ref=20)
frequencies = jnp.array(np.load("gw150914_frequencies.npy"), dtype=jnp.float64)
detectors = [H1, L1]
for det in detectors:
    det.frequencies = frequencies
    det.data = jnp.array(np.load(f"gw150914_{det.name}_strain.npy"), dtype=jnp.complex128)
    det.psd  = jnp.array(np.load(f"gw150914_{det.name}_psd.npy"),    dtype=jnp.float64)

gps  = 1126259462.4
gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad
epoch = 4 - 2
df = float(frequencies[1] - frequencies[0])

# ---------------------------------------------------------------------------
# Draw posterior samples
# ---------------------------------------------------------------------------
samples = read_chains(args.chain)
draws   = samples.sample(args.M)

phy_keys = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "psi", "ra", "dec", "phase_c"]

taus        = []
ess_medians = []
ess_5th     = []
ess_25th    = []
is_corrs    = []

for i, (_, row) in enumerate(draws.iterrows()):
    p   = {k: float(row[k]) for k in phy_keys}
    tau = float(row["tau"])
    p["gmst"] = gmst
    p["eta"]  = p["q"] / (1 + p["q"])**2

    wf         = waveform(frequencies, p)
    align_time = jnp.exp(-1j * 2 * jnp.pi * frequencies * (epoch + p["t_c"]))

    B_list, S0_list = [], []
    for det in detectors:
        h_dec = det.fd_response(frequencies, wf, p) * align_time
        B     = np.array(2.0 * df * jnp.abs(det.data - h_dec)**2)
        B_list.append(B)
        S0_list.append(np.array(det.psd))

    B_all  = np.concatenate(B_list)
    S0_all = np.concatenate(S0_list)
    sig0   = 0.5 * np.log(S0_all)

    # Newton solve for sigma_hat
    s = sig0.copy()
    for _ in range(8):
        e2s = np.exp(-2 * s)
        f   = -2 + 2 * B_all * e2s - (s - sig0) / tau**2
        fp  = -4 * B_all * e2s - 1.0 / tau**2
        s  -= f / fp
    sig_hat = s

    H   = 4 * B_all * np.exp(-2 * sig_hat) + 1.0 / tau**2
    std = 1.0 / np.sqrt(H)

    # K Laplace samples: (K, N_bins)
    sig_samp = sig_hat[None, :] + std[None, :] * rng.standard_normal((args.K, len(B_all)))

    # True log-posterior (up to constants not depending on sigma)
    log_p = (-2 * sig_samp
             - B_all[None, :] * np.exp(-2 * sig_samp)
             - (sig_samp - sig0[None, :])**2 / (2 * tau**2))

    # Laplace log-density
    log_q = (-0.5 * np.log(2 * np.pi / H[None, :])
             - 0.5 * H[None, :] * (sig_samp - sig_hat[None, :])**2)

    log_w  = log_p - log_q                          # (K, N_bins)
    lw_max = log_w.max(axis=0, keepdims=True)
    w      = np.exp(log_w - lw_max)

    ess_per_bin = w.sum(axis=0)**2 / (w**2).sum(axis=0) / args.K

    # IS correction to log L (per-bin then summed)
    log_p_hat = (-2 * sig_hat
                 - B_all * np.exp(-2 * sig_hat)
                 - (sig_hat - sig0)**2 / (2 * tau**2))
    log_L_lap        = log_p_hat + 0.5 * np.log(2 * np.pi / H)
    log_L_IS_per_bin = lw_max[0, :] + np.log(w.mean(axis=0))
    is_corr          = float(np.sum(log_L_IS_per_bin - log_L_lap))

    taus.append(tau)
    ess_medians.append(float(np.median(ess_per_bin)))
    ess_5th.append(float(np.percentile(ess_per_bin, 5)))
    ess_25th.append(float(np.percentile(ess_per_bin, 25)))
    is_corrs.append(is_corr)

    print(f"Sample {i+1:3d}: tau={tau:.4f}  "
          f"ESS/K median={ess_medians[-1]:.3f}  5th={ess_5th[-1]:.3f}  "
          f"IS_corr={is_corr:+.1f} nats")

taus        = np.array(taus)
ess_medians = np.array(ess_medians)
ess_5th     = np.array(ess_5th)
ess_25th    = np.array(ess_25th)
is_corrs    = np.array(is_corrs)

print()
print("=== Summary ===")
print(f"Per-bin ESS/K: median={np.median(ess_medians):.3f}, "
      f"range=[{np.min(ess_5th):.3f}, {np.max(ess_medians):.3f}]")
print(f"IS correction to log L: mean={np.mean(is_corrs):.1f} ± {np.std(is_corrs):.1f} nats")
print(f"IS correction per bin:  {np.mean(is_corrs)/len(B_all):.4f} nats/bin")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.scatter(taus, ess_medians, c="steelblue", s=30, label="median per-bin")
ax.scatter(taus, ess_5th,     c="steelblue", s=10, marker="v", alpha=0.5, label="5th pct")
ax.axhline(0.5, ls="--", c="r", lw=1, label="ESS/K = 0.5")
ax.axhline(1.0, ls=":", c="grey", lw=1)
ax.set_xlabel(r"$\tau$")
ax.set_ylabel("Per-bin ESS/K")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)
ax.set_title("Laplace quality per frequency bin")

ax = axes[1]
ax.scatter(taus, is_corrs, c="firebrick", s=30)
ax.axhline(0, ls=":", c="grey", lw=1)
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"IS correction to $\log\mathcal{L}$ [nats]")
ax.set_title("Total IS correction (sum over bins)")

fig.tight_layout()
out = "alcs_is_diagnostic.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"\nSaved {out}")
