"""
Query the GWOSC GWTC-3-confident catalog, filter to BBH events with
p_astro >= PASTRO_THRESHOLD, and save a catalog .npy file containing
GPS times, Mc prior ranges, and detector lists.

Usage:
    python build_catalog.py [--pastro 0.99] [--out o3_catalog.npy]

The output .npy can be loaded with:
    catalog = np.load('o3_catalog.npy', allow_pickle=True).item()
    # -> dict of {event_name: {gps, mc, mc_low, mc_high, detectors, pastro}}
"""

import argparse
import numpy as np
import requests

CATALOG_URL = "https://gw-openscience.org/eventapi/json/GWTC-3-confident/"

parser = argparse.ArgumentParser()
parser.add_argument('--pastro', type=float, default=0.99,
                    help='Minimum p_astro threshold (default 0.99)')
parser.add_argument('--out', default='o3_catalog.npy',
                    help='Output filename (default o3_catalog.npy)')
args = parser.parse_args()

print(f"Fetching {CATALOG_URL} ...")
r = requests.get(CATALOG_URL, timeout=30)
r.raise_for_status()
data = r.json()

# GWOSC returns {"events": {"GW190521-074359": {...}, ...}}
# Each entry may have a "commonName" field with the canonical name.
events_raw = data.get('events', data)  # fallback if top-level is flat

catalog = {}
n_skipped_type   = 0
n_skipped_pastro = 0
n_skipped_nomc   = 0

for key, ev in events_raw.items():
    # Canonical name (e.g. "GW190521") preferred over catalog key
    name = ev.get('commonName', key)

    # --- type filter ---
    ev_type = ev.get('type', ev.get('classification', ''))
    if 'BBH' not in str(ev_type):
        n_skipped_type += 1
        continue

    # --- p_astro filter ---
    pastro = float(ev.get('p_astro', ev.get('pastro', 0.0)))
    if pastro < args.pastro:
        n_skipped_pastro += 1
        continue

    # --- GPS ---
    gps = float(ev.get('GPS', ev.get('gps', 0.0)))

    # --- chirp mass and 90% CI ---
    # Try source-frame first, then detector-frame
    mc       = ev.get('chirp_mass_source',       ev.get('chirp_mass',       None))
    mc_lower = ev.get('chirp_mass_source_lower',  ev.get('chirp_mass_lower',  None))
    mc_upper = ev.get('chirp_mass_source_upper',  ev.get('chirp_mass_upper',  None))

    if mc is None:
        print(f"  WARNING: no chirp mass for {name} — skipping")
        n_skipped_nomc += 1
        continue

    mc       = float(mc)
    mc_lower = float(mc_lower) if mc_lower is not None else 0.7 * mc
    mc_upper = float(mc_upper) if mc_upper is not None else 1.3 * mc

    # Prior range: 2× the 90% CI half-width on each side, floored at 1 Msun
    half_up   = mc_upper - mc
    half_down = mc - mc_lower
    half      = max(half_up, half_down, 0.1 * mc)   # at least 10% of mc
    mc_low  = max(mc - 2.0 * half, 1.0)
    mc_high = mc + 2.0 * half

    # --- detectors ---
    ifos_str   = ev.get('ifos', ev.get('instruments', 'H1,L1'))
    detectors  = [d.strip() for d in str(ifos_str).replace(' ', '').split(',')]
    # Normalise names just in case
    detectors  = [d for d in detectors if d in ('H1', 'L1', 'V1', 'K1')]

    catalog[name] = {
        'gps':       gps,
        'mc':        round(mc,      3),
        'mc_low':    round(mc_low,  2),
        'mc_high':   round(mc_high, 2),
        'detectors': detectors,
        'pastro':    round(pastro,  4),
    }
    print(f"  {name:16s}  GPS={gps:.1f}  "
          f"Mc={mc:.2f} [{mc_low:.1f},{mc_high:.1f}] Msun  "
          f"dets={detectors}  p_astro={pastro:.4f}")

print(f"\n{'─'*60}")
print(f"  Kept   : {len(catalog)} events")
print(f"  Skipped: {n_skipped_type} non-BBH, "
      f"{n_skipped_pastro} low p_astro (<{args.pastro}), "
      f"{n_skipped_nomc} no Mc")

# Save ordered by GPS time so --idx gives chronological order
catalog_sorted = dict(sorted(catalog.items(), key=lambda x: x[1]['gps']))

np.save(args.out, catalog_sorted, allow_pickle=True)
print(f"\nSaved {len(catalog_sorted)} events → {args.out}")
print("Load with: catalog = np.load('o3_catalog.npy', allow_pickle=True).item()")

# Print a quick index table
print(f"\n{'IDX':>4}  {'Event':16s}  {'GPS':>14}  Mc range")
for idx, (name, ev) in enumerate(catalog_sorted.items()):
    print(f"  {idx:2d}   {name:16s}  {ev['gps']:14.1f}  "
          f"[{ev['mc_low']:.1f}, {ev['mc_high']:.1f}]")
