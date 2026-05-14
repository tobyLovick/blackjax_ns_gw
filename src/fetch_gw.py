"""Fetch strain + Welch PSD for a named GW event from GWOSC."""
import argparse
import numpy as np
from jimgw.single_event.detector import H1, L1

EVENTS = {
    'GW150914': 1126259462.4,
    'GW151226': 1135136350.6,
    'GW170104': 1167559936.6,
    'GW170608': 1180922494.5,
    'GW170809': 1186302519.8,
    'GW170814': 1186741861.5,
    'GW170823': 1187529256.5,
}

parser = argparse.ArgumentParser()
parser.add_argument('--event', required=True, choices=list(EVENTS.keys()))
args = parser.parse_args()

gps  = EVENTS[args.event]
name = args.event.lower()

print(f"Fetching {args.event} (GPS={gps})...")
H1.load_data(gps, 2, 2, 20.0, 1024.0, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, 2, 2, 20.0, 1024.0, psd_pad=16, tukey_alpha=0.2)

np.save(f'{name}_frequencies.npy', np.array(H1.frequencies))
np.save(f'{name}_H1_strain.npy',   np.array(H1.data))
np.save(f'{name}_L1_strain.npy',   np.array(L1.data))
np.save(f'{name}_H1_psd.npy',      np.array(H1.psd))
np.save(f'{name}_L1_psd.npy',      np.array(L1.psd))

print(f"Done. {len(H1.frequencies)} bins, {H1.frequencies[0]:.1f}–{H1.frequencies[-1]:.1f} Hz")
