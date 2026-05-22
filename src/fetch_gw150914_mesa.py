"""Compute MESA (Burg/Maximum Entropy) PSD for GW150914 from off-source data.

Uses the identical 16s off-source window as jimgw's load_data (trigger-34s to
trigger-18s), then evaluates the MESA PSD at the same frequency bins as the
saved strain data.  Saves gw150914_H1_psd_mesa.npy and gw150914_L1_psd_mesa.npy.
"""
import numpy as np
from gwpy.timeseries import TimeSeries
from memspectrum import MESA
from scipy.interpolate import interp1d

GPS = 1126259462.4
GPS_START_PAD = 2
PSD_PAD = 16
FMIN = 20.0
FMAX = 1024.0

# Same off-source window jimgw uses: trigger - gps_start_pad - 2*psd_pad to
# trigger - gps_start_pad - psd_pad
start_psd = int(GPS) - GPS_START_PAD - 2 * PSD_PAD   # GPS - 34
end_psd   = int(GPS) - GPS_START_PAD - PSD_PAD         # GPS - 18

freqs = np.load('gw150914_frequencies.npy')

for ifo in ('H1', 'L1'):
    print(f"Fetching {ifo} off-source data ({start_psd}–{end_psd})...")
    td = TimeSeries.fetch_open_data(ifo, start_psd, end_psd, cache=True)
    dt = td.dt.value
    fs = 1.0 / dt

    print(f"  Running MESA on {len(td)} samples at {fs:.0f} Hz...")
    m = MESA()
    P, _, _ = m.solve(td.value)

    # Evaluate MESA PSD at target frequencies
    mesa_freqs, mesa_psd = m.spectrum(dt)
    # m.spectrum() returns two-sided PSD; gwpy Welch returns one-sided → ×2
    interp = interp1d(mesa_freqs, 2.0 * mesa_psd, kind='linear', fill_value='extrapolate')
    psd_at_freqs = interp(freqs)

    out = f'gw150914_{ifo}_psd_mesa.npy'
    np.save(out, psd_at_freqs)
    print(f"  Saved {out}  (shape {psd_at_freqs.shape})")

print("Done.")
