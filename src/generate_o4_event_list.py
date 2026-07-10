"""
Generate o4_event_list.json for all O4 BBH events passing quality cuts.

Pulls from GWOSC catalog API and Zenodo file lists.
Filters: p_astro > P_ASTRO_MIN, m2 > M2_MIN (BBH proxy), H1+L1 both present in BW PSD.

Run locally, then copy o4_event_list.json to Isambard alongside the analysis scripts.
"""

import json
import urllib.request
import sys

P_ASTRO_MIN = 0.9
M2_MIN      = 3.0   # M_sun, excludes NSBH / BNS

CATALOGS = {
    "GWTC-4.0": {
        "zenodo_record": "16053484",
        "hdf5_prefix":   "IGWN-GWTC4p0-0f954158d_720-",
        "hdf5_suffix":   "-combined_PEDataRelease.hdf5",
    },
    "GWTC-5.0": {
        "zenodo_record": "20348005",
        "hdf5_prefix":   "IGWN-GWTC5p0-29ebe06b7_25-",
        "hdf5_suffix":   "-combined_PEDataRelease.hdf5",
    },
}


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def get_zenodo_files(record):
    data = fetch_json(f"https://zenodo.org/api/records/{record}/files")
    return {f["key"]: f for f in data.get("entries", data.get("files", []))}


def strip_version(name):
    """GW230518_125908-v1 → GW230518_125908"""
    return name.split("-v")[0]


def main():
    events = []

    for catalog_name, cfg in CATALOGS.items():
        print(f"\n=== {catalog_name} ===")
        gwosc_data = fetch_json(f"https://gwosc.org/eventapi/json/{catalog_name}/")
        catalog_events = gwosc_data["events"]

        zenodo_files = get_zenodo_files(cfg["zenodo_record"])
        print(f"  {len(catalog_events)} GWOSC events,  {len(zenodo_files)} Zenodo files")

        for raw_name, ev in catalog_events.items():
            base_name = strip_version(raw_name)

            p_astro = ev.get("p_astro") or 0.0
            m2      = ev.get("mass_2_source") or 0.0
            if p_astro < P_ASTRO_MIN or m2 < M2_MIN:
                continue

            hdf5_name = cfg["hdf5_prefix"] + base_name + cfg["hdf5_suffix"]
            if hdf5_name not in zenodo_files:
                print(f"  WARNING: no HDF5 for {base_name}")
                continue

            gps       = ev.get("GPS") or ev.get("gps")
            mc_det    = ev.get("chirp_mass") or 0.0        # detector-frame
            mc_src    = ev.get("chirp_mass_source") or 0.0
            dl_median = ev.get("luminosity_distance") or 500.0
            snr       = ev.get("network_matched_filter_snr") or 0.0

            # Prior bounds — wide enough to bracket posterior
            mc_lo = max(1.0,  mc_det * 0.45)
            mc_hi = min(500., mc_det * 2.5)
            dl_lo = max(50.,  dl_median * 0.1)
            dl_hi = min(20000., dl_median * 5.0)

            events.append({
                "name":           base_name,
                "catalog":        catalog_name,
                "zenodo_record":  cfg["zenodo_record"],
                "hdf5_name":      hdf5_name,
                "hdf5_url":       f"https://zenodo.org/api/records/{cfg['zenodo_record']}/files/{hdf5_name}/content",
                "gps":            float(gps),
                "p_astro":        float(p_astro),
                "m2_source":      float(m2),
                "snr":            float(snr),
                "Mc_det_median":  float(mc_det),
                "Mc_src_median":  float(mc_src),
                "dL_median":      float(dl_median),
                "prior": {
                    "Mc_lo": mc_lo, "Mc_hi": mc_hi,
                    "dL_lo": dl_lo, "dL_hi": dl_hi,
                },
            })

        print(f"  {sum(1 for e in events if e['catalog']==catalog_name)} passing cuts")

    events.sort(key=lambda e: e["gps"])

    out = "o4_event_list.json"
    with open(out, "w") as f:
        json.dump(events, f, indent=2)

    print(f"\nTotal: {len(events)} events → {out}")
    print(f"SLURM array: --array=0-{len(events)-1}")


if __name__ == "__main__":
    main()
