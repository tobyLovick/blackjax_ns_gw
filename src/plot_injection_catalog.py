"""Plot the injection catalog: Mc, q, d_L, SNR distributions + Mc-d_L scatter."""
import numpy as np
import matplotlib.pyplot as plt

catalog = np.load('injection_catalog.npy', allow_pickle=True)

Mcs  = np.array([e['M_c'] for e in catalog])
qs   = np.array([e['q']   for e in catalog])
dLs  = np.array([e['d_L'] for e in catalog])
snrs = np.array([e['snr'] for e in catalog])
m1s  = np.array([e['m1']  for e in catalog])
m2s  = np.array([e['m2']  for e in catalog])

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

kw = dict(bins=20, color='C0', edgecolor='white', linewidth=0.4)

axes[0].hist(Mcs,  **kw)
axes[0].set_xlabel(r'$\mathcal{M}_c\ [M_\odot]$', fontsize=11)

axes[1].hist(qs,   **kw)
axes[1].set_xlabel(r'$q = m_2/m_1$', fontsize=11)

axes[2].hist(dLs,  **kw)
axes[2].set_xlabel(r'$d_L\ [\mathrm{Mpc}]$', fontsize=11)

axes[3].hist(snrs, **kw)
axes[3].set_xlabel(r'Network SNR (H1+L1)', fontsize=11)

axes[4].hist(m1s,  **kw)
axes[4].set_xlabel(r'$m_1\ [M_\odot]$', fontsize=11)

sc = axes[5].scatter(dLs, Mcs, c=snrs, cmap='viridis', s=18, alpha=0.8)
axes[5].set_xlabel(r'$d_L\ [\mathrm{Mpc}]$', fontsize=11)
axes[5].set_ylabel(r'$\mathcal{M}_c\ [M_\odot]$', fontsize=11)
plt.colorbar(sc, ax=axes[5], label='SNR')

for ax in axes[:5]:
    ax.set_ylabel('Count', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
axes[5].spines[['top', 'right']].set_visible(False)

fig.suptitle(f'Injection catalog  (N={len(catalog)}, SNR≥8, Power Law+Peak mass model)',
             fontsize=11)
fig.tight_layout()
fig.savefig('injection_catalog.pdf', bbox_inches='tight')
print("Saved injection_catalog.pdf")
plt.show()
