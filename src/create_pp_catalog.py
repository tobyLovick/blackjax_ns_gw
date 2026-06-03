"""
Draw 500 prior samples for the PP test and save as pp_catalog.npy.

Prior (matches exactly what blackjax_pp_injection.py uses for recovery):
  M_c     ~ Uniform [8, 60] Msun
  q       ~ Uniform [0.2, 1.0]
  s1_z    = 0  (fixed)
  s2_z    = 0  (fixed)
  iota    ~ Sine    [0, pi]
  d_L     ~ p ~ d_L^2  [100, 2500] Mpc
  t_c     ~ Uniform [-0.05, 0.05] s
  phase_c ~ Uniform [0, 2pi]
  psi     ~ Uniform [0, pi]
  ra      ~ Uniform [0, 2pi]
  dec     ~ Cosine  [-pi/2, pi/2]

Spins fixed to zero (IMRPhenomD, non-spinning injections).
GPS fixed to GW150914 (sets gmst for antenna patterns).

Saves:
  pp_catalog.npy   — list of 500 dicts (one per injection)
  pp_catalog_arr.npy — (500, 11) float array (row order matches PP_PARAMS)
"""

import numpy as np
from astropy.time import Time

N_EVENTS     = 500
SEED         = 0
GPS_FIDUCIAL = 1126259462.4
MC_MIN, MC_MAX   = 8.0,   60.0
Q_MIN,  Q_MAX    = 0.2,   1.0
DL_MIN, DL_MAX   = 100.0, 2500.0
TC_MIN, TC_MAX   = -0.05, 0.05

rng  = np.random.default_rng(SEED)
gmst = Time(GPS_FIDUCIAL, format='gps').sidereal_time('apparent', 'greenwich').rad

catalog = []
for i in range(N_EVENTS):
    Mc      = rng.uniform(MC_MIN, MC_MAX)
    q       = rng.uniform(Q_MIN,  Q_MAX)
    iota    = np.arccos(rng.uniform(-1, 1))         # sine prior
    u_dl    = rng.uniform(0, 1)
    d_L     = (DL_MIN**3 + u_dl * (DL_MAX**3 - DL_MIN**3)) ** (1/3)
    t_c     = rng.uniform(TC_MIN, TC_MAX)
    phase_c = rng.uniform(0, 2*np.pi)
    psi     = rng.uniform(0, np.pi)
    ra      = rng.uniform(0, 2*np.pi)
    dec     = np.arcsin(rng.uniform(-1, 1))          # cosine prior

    catalog.append({
        'M_c':     Mc,
        'q':       q,
        's1_z':    0.0,
        's2_z':    0.0,
        'iota':    iota,
        'd_L':     d_L,
        't_c':     t_c,
        'phase_c': phase_c,
        'psi':     psi,
        'ra':      ra,
        'dec':     dec,
        'gmst':    gmst,
        'eta':     q / (1 + q)**2,
    })

np.save('pp_catalog.npy', catalog, allow_pickle=True)

# Also save as a plain (500, 11) array for quick indexing
PP_PARAMS = ['M_c','q','s1_z','s2_z','iota','d_L','t_c','phase_c','psi','ra','dec']
arr = np.array([[e[k] for k in PP_PARAMS] for e in catalog])
np.save('pp_catalog_arr.npy', arr)

print(f"Saved pp_catalog.npy and pp_catalog_arr.npy ({N_EVENTS} events)")
print(f"Parameter ranges in catalog:")
for j, k in enumerate(PP_PARAMS):
    print(f"  {k:<10} [{arr[:,j].min():.3f}, {arr[:,j].max():.3f}]")
