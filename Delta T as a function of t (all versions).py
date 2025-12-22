# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 15:23:56 2025

Transient model: C dΔT/dt = F + λΔT + aΔT²
First graph only, all lines clearly labelled.
"""

import numpy as np
import matplotlib.pyplot as plt

#-----------Fixed parameters------------------------
#lam_max = 0
lam_max = -0.79
lam_mean = -1.17
lam_min = -1.78
lam_values = [lam_max, lam_mean, lam_min]

F2 = 3.71
C = 8.36*10**8

# a range (same as original)
a = np.arange(-0.1,0.1,0.001)

# choose α values

ac = -0.035
am = 0.03
ah = 0.058
alpha_values = [ac, am, ah]

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



plt.plot([],[],' ',label= 'F = 3.71')
for lam in lam_values:
    for alpha in alpha_values:
        if lam == lam_max:
            Tseries = del_T_trans(lam, alpha, F2, C, dt, nsteps)
            lbl = "λ=" + str(lam) + ", α=" + str(alpha)
            plt.plot(time, Tseries,'-', label=lbl)
        if lam == lam_mean:
            Tseries = del_T_trans(lam, alpha, F2, C, dt, nsteps)
            lbl = "λ=" + str(lam) + ", α=" + str(alpha)
            plt.plot(time, Tseries,'--', label=lbl)
        if lam == lam_min:
            Tseries = del_T_trans(lam, alpha, F2, C, dt, nsteps)
            lbl = "λ=" + str(lam) + ", α=" + str(alpha)
            plt.plot(time, Tseries,':',label=lbl)


plt.title("Transient ΔT for 2xCO$_2$")
plt.ylabel("ΔT (K)")
plt.xlabel("Time (years)")
plt.ylim(0,14)
plt.grid(True)
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

F4 = 7.42

plt.plot([],[],' ',label= 'F = 7.42')
for lam in lam_values:
    for alpha in alpha_values:
        if lam == lam_max:
            Tseries = del_T_trans(lam, alpha, F4, C, dt, nsteps)
            lbl = "λ=" + str(lam) + ", α=" + str(alpha)
            plt.plot(time, Tseries,'-', label=lbl)
        if lam == lam_mean:
            Tseries = del_T_trans(lam, alpha, F4, C, dt, nsteps)
            lbl = "λ=" + str(lam) + ", α=" + str(alpha)
            plt.plot(time, Tseries,'--', label=lbl)
        if lam == lam_min:
            Tseries = del_T_trans(lam, alpha, F4, C, dt, nsteps)
            lbl = "λ=" + str(lam) + ", α=" + str(alpha)
            plt.plot(time, Tseries,':',label=lbl)


plt.title("Transient ΔT for 4xCO$_2$")
plt.ylabel("ΔT (K)")
plt.xlabel("Time (years)")
plt.ylim(0,30)
plt.grid(True)
plt.legend(bbox_to_anchor=(1, 1))
plt.show()