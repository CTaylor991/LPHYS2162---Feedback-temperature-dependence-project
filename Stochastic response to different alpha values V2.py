# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 11:48:53 2025
-------------------------------------------------------------------------
DIFFERENT Beta VALUE
-------------------------------------------------------------------------
@author: charl
"""

import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()



#-----Set variables and any parameters that might want to be changed---------------------

lam  = -1.28
#a = 0.058   #α
#a_values = [-1, -0.035,0.03,0.058]
a_values = [-1,-0.5,0.5,1]
a_values = [-0.1,-0.05,0.05,0.1]
beta = -4*10**-6
C = 8.36*10**8
sigma = 7700

# Range of F values
F_values = [3.7]

# Ensemble size
n_ensemble = 20
ensemble_prints = 20

alpha = 0.4

#-----Step parameters-------------------

years = 200
dt = 60*60*24
t_end = years*365*24*60*60
nsteps = int(t_end/dt)
steprange = range(nsteps)

time_years = np.arange(nsteps)*dt/(60*60*24*365)

#-----Define function-------------------

def integrate_stochastic(F, lam, a, beta, C, sigma, dt, nsteps):
    T = np.zeros(nsteps)
    for i in range(nsteps - 1):
        T_D = (F + lam*T[i] + a*T[i]**2 + beta*T[i]**5) / C
        noise = (sigma/C)*np.sqrt(dt)*np.random.normal(0,1)
        T[i+1] = T[i] + T_D*dt + noise
    return T

#-----Run ensembles---------------------

T_ensemble = {}
T_mean = {}

print("Running ensembles...")

for F in F_values:
    
    for a in a_values:

        print(f"\nRunning ensemble for F = {F} W/m^2 and α = {a}".format(F))

        T_members = np.zeros((n_ensemble, nsteps))

        for run in range(n_ensemble):
            T_members[run] = integrate_stochastic(F, lam, a, beta, C, sigma, dt, nsteps)
            milestone = int(((run + 1) / n_ensemble) * 100)
            if milestone % 25 == 0:
                elapsed = round(time.time() - start_time, 1)
                print(f"  {milestone}% completed in: {elapsed}s)")

        T_ensemble[(F, a)] = T_members
        T_mean[(F, a)] = np.mean(T_members, axis=0)

#-----Plotting--------------------------

plt.figure(figsize=(10,5))

colormaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges, plt.cm.Greys]
combo_index = 0

for F in F_values:
    # cmap = colormaps[combo_index]
    # combo_index += 1

    # shades = np.linspace(0.4, 0.8, ensemble_prints)
    for a in a_values:
        cmap = colormaps[combo_index]
        combo_index += 1

        shades = np.linspace(0.4, 0.8, ensemble_prints)
        for i in range(ensemble_prints):
            plt.plot(time_years,
                     T_ensemble[(F, a)][i],
                     color=cmap(shades[i]),
                     linewidth=0.6,
                     alpha=alpha)

        plt.plot(time_years,
                 T_mean[(F, a)],
                 color=cmap(0.95),
                 linewidth=3,
                 label=f"α = {a}")

plt.xlabel("Time (years)")
plt.ylabel("ΔT (K)")
plt.title(f"Stochastic Temperature Response - F:{F}, β:{beta}, λ: {lam}, σ:{sigma}, Ensemble per mean:{n_ensemble}")
plt.grid(True)
plt.xlim(0, years)
plt.legend()
#plt.yticks(np.arange(0, 5, 0.5))
#plt.ylim(0,8)
plt.tight_layout()
plt.show()
