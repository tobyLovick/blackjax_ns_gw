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

CATALOG_URL = "https://gwosc.org/eventapi/json/GWTC-3-confident/"

parser = argparse.ArgumentParser()
parser.add_argument('--pastro', type=float, default=0.99,
                    help='Minimum p_astro threshold (default 0.99)')
parser.add_argument('--out', default='o3_catalog.npy',
                    help='Output filename (default o3_catalog.npy)')
args = parser.parse_args()

print(f"Fetching {CATALOG_URL} ...")
r = requests.get(CATALOG_URL, timeout=30)
r.raise_for_status()
events_raw = r.json().get('events', r.json())

catalog = {}
n_skipped_type   = 0
n_skipped_pastro = 0
n_skipped_nomc   = 0

for key, ev in events_raw.items():
    name = ev.get('commonName', key)

    # --- BBH filter: lighter component above NS mass threshold ---
    # GWOSC has no explicit 'type' field; m2_source > 3 Msun means both are BHs
    m2 = ev.get('mass_2_source', None)
    if m2 is None or float(m2) < 3.0:
        n_skipped_type += 1
        continue

    # --- p_astro filter ---
    pastro = float(ev.get('p_astro', 0.0) or 0.0)
    if pastro < args.pastro:
        n_skipped_pastro += 1
        continue

    # --- GPS ---
    gps = float(ev.get('GPS', 0.0))

    # --- chirp mass ---
    # API returns lower/upper as signed offsets from the median (e.g. -4.0 means mc-4)
    mc       = ev.get('chirp_mass_source', ev.get('chirp_mass', None))
    mc_lower = ev.get('chirp_mass_source_lower', ev.get('chirp_mass_lower', None))
    mc_upper = ev.get('chirp_mass_source_upper', ev.get('chirp_mass_upper', None))

    if mc is None:
        print(f"  WARNING: no chirp mass for {name} — skipping")
        n_skipped_nomc += 1
        continue

    mc = float(mc)
    mc_lower = mc + float(mc_lower) if mc_lower is not None else 0.7 * mc
    mc_upper = mc + float(mc_upper) if mc_upper is not None else 1.3 * mc

    # Prior range: 2× the 90% CI half-width on each side, floored at 1 Msun
    half = max(mc_upper - mc, mc - mc_lower, 0.1 * mc)
    mc_low  = max(mc - 2.0 * half, 1.0)
    mc_high = mc + 2.0 * half

    # --- detectors: fetch per-event JSON and read from strain list ---
    jsonurl = ev.get('jsonurl', '')
    detectors = []
    if jsonurl:
        try:
            ev_detail = requests.get(jsonurl, timeout=30).json()
            ev_inner  = next(iter(ev_detail.get('events', {}).values()), {})
            strain    = ev_inner.get('strain', [])
            detectors = sorted({s['detector'] for s in strain
                                if s.get('detector') in ('H1', 'L1', 'V1', 'K1')})
        except Exception as exc:
            print(f"  WARNING: could not fetch {jsonurl}: {exc}")
    if not detectors:
        detectors = ['H1', 'L1']   # safe fallback

    catalog[name] = {
        'gps':       gps,
        'mc':        round(mc,      3),
        'mc_low':    round(mc_low,  2),
        'mc_high':   round(mc_high, 2),
        'detectors': detectors,
        'pastro':    round(pastro,  4),
    }
    print(f"  {name:20s}  GPS={gps:.1f}  "
          f"Mc={mc:.2f} [{mc_low:.1f},{mc_high:.1f}] Msun  "
          f"dets={detectors}  p_astro={pastro:.4f}")

print(f"\n{'─'*60}")
print(f"  Kept   : {len(catalog)} events")
print(f"  Skipped: {n_skipped_type} non-BBH, "
      f"{n_skipped_pastro} low p_astro (<{args.pastro}), "
      f"{n_skipped_nomc} no Mc")

catalog_sorted = dict(sorted(catalog.items(), key=lambda x: x[1]['gps']))

np.save(args.out, catalog_sorted, allow_pickle=True)
print(f"\nSaved {len(catalog_sorted)} events → {args.out}")
print("Load with: catalog = np.load('o3_catalog.npy', allow_pickle=True).item()")

print(f"\n{'IDX':>4}  {'Event':20s}  {'GPS':>14}  {'Dets':12s}  Mc range")
for idx, (n, e) in enumerate(catalog_sorted.items()):
    print(f"  {idx:2d}   {n:20s}  {e['gps']:14.1f}  "
          f"{','.join(e['detectors']):12s}  [{e['mc_low']:.1f}, {e['mc_high']:.1f}]")
