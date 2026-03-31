"""Download GW150914 strain + Welch PSD from GWOSC and save to .npy files."""
import numpy as np
from jimgw.single_event.detector import H1, L1

gps  = 1126259462.4
fmin = 20.0
fmax = 1024.0

print("Fetching H1 data...")
H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
print("Fetching L1 data...")
L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

np.save('gw150914_frequencies.npy', np.array(H1.frequencies))
np.save('gw150914_H1_strain.npy',   np.array(H1.data))
np.save('gw150914_L1_strain.npy',   np.array(L1.data))
np.save('gw150914_H1_psd.npy',      np.array(H1.psd))
np.save('gw150914_L1_psd.npy',      np.array(L1.psd))

print(f"Done. {len(H1.frequencies)} frequency bins, "
      f"{H1.frequencies[0]:.1f}–{H1.frequencies[-1]:.1f} Hz")
