# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 15:23:56 2025

Transient model: C dΔT/dt = F + λΔT + aΔT²
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


#-----------Fixed parameters------------------------
lam_values = np.linspace(-2,2,21)
alpha_values = np.linspace(-0.2,0.2,21)

F2 = 3.71
C = 8.36*10**8

a = np.arange(-0.1,0.1,0.001)

#constantss
alpha_top_1 = -0.5
alpha_top_2 = 0.0
alpha_top_3 = 0.5

lam_bottom_1 = -0.5
lam_bottom_2 = 0.0
lam_bottom_3 = 0.5

# time stepping
years = 200
dt_days = 30.0
dt = dt_days * 24 * 3600
nsteps = int((years * 365.0) / dt_days)
time = np.arange(nsteps+1) * dt / (3600*24*365)

#---------Function-----------------------
def del_T_trans(lam, alpha, F, C, dt, nsteps):
    T = np.zeros(nsteps+1)
    for i in range(nsteps):
        dTdt = (F + lam*T[i] + alpha*T[i]**2) / C
        T[i+1] = T[i] + dTdt*dt
    return T

#-----------------------------------------------------

fig, axs = plt.subplots(2, 3, figsize=(18, 8))

#lamb da values
cmap_name = 'copper'
cmap = plt.get_cmap(cmap_name)
norm = Normalize(vmin=lam_values.min(), vmax=lam_values.max())
sm = ScalarMappable(norm=norm, cmap=cmap)

for lam in lam_values:
    axs[0,0].plot(time,del_T_trans(lam, alpha_top_1, F2, C, dt, nsteps), color=cmap(norm(lam)))

axs[0,0].set_title(f"α = {alpha_top_1:.1f}")
axs[0,0].set_ylabel("ΔT (K)")
axs[0,0].set_ylim(0,30)
axs[0,0].grid(True)
plt.colorbar(sm, ax=axs[0,0]).set_label('λ')


for lam in lam_values:
    axs[0,1].plot(time,del_T_trans(lam, alpha_top_2, F2, C, dt, nsteps), color=cmap(norm(lam)))

axs[0,1].set_title(f"α = {alpha_top_2:.1f}")
axs[0,1].set_ylim(0,30)
axs[0,1].grid(True)
plt.colorbar(sm, ax=axs[0,1]).set_label('λ')


for lam in lam_values:
    axs[0,2].plot(time,del_T_trans(lam, alpha_top_3, F2, C, dt, nsteps), color=cmap(norm(lam)))

axs[0,2].set_title(f"α = {alpha_top_3:.1f}")
axs[0,2].set_ylim(0,30)
axs[0,2].set_xlim(0,200)
axs[0,2].grid(True)
plt.colorbar(sm, ax=axs[0,2]).set_label('λ')


#-Alpha values
cmap_name = 'plasma'
cmap = plt.get_cmap(cmap_name)
norm = Normalize(vmin=alpha_values.min(), vmax=alpha_values.max())
sm = ScalarMappable(norm=norm, cmap=cmap)

for alpha in alpha_values:
    axs[1,0].plot(time,del_T_trans(lam_bottom_1, alpha, F2, C, dt, nsteps), color=cmap(norm(alpha)))

axs[1,0].set_title(f"λ = {lam_bottom_1:.1f}")
axs[1,0].set_ylabel("ΔT (K)")
axs[1,0].set_xlabel("Time (years)")
axs[1,0].set_ylim(0,30)
axs[1,0].grid(True)
plt.colorbar(sm, ax=axs[1,0]).set_label('α')


for alpha in alpha_values:
    axs[1,1].plot(time,del_T_trans(lam_bottom_2, alpha, F2, C, dt, nsteps), color=cmap(norm(alpha)))

axs[1,1].set_title(f"λ = {lam_bottom_2:.1f}")
axs[1,1].set_xlabel("Time (years)")
axs[1,1].set_ylim(0,30)
axs[1,1].grid(True)
plt.colorbar(sm, ax=axs[1,1]).set_label('α')


for alpha in alpha_values:
    axs[1,2].plot(time,del_T_trans(lam_bottom_3, alpha, F2, C, dt, nsteps), color=cmap(norm(alpha)))

axs[1,2].set_title(f"λ = {lam_bottom_3:.1f}")
axs[1,2].set_xlabel("Time (years)")
axs[1,2].set_ylim(0,30)
axs[1,2].grid(True)
plt.colorbar(sm, ax=axs[1,2]).set_label('α')


plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.show()
