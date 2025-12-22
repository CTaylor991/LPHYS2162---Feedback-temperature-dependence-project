# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 18:59:26 2025

@author: charl
"""

import numpy as np
import matplotlib.pyplot as plt


#----Set variables---------------------

lam = -1.28
a = 0.058
beta = -0.001
C = 8.36*10**8
sigma = 7700

#Range of F's for plotting
F_values = [0,1,3.71,7.42]

#Step parameters
years = 200
dt = 60*60*24
t_end = years*365*25*60*60
nsteps = int(t_end/dt)
steprange = range(nsteps)

time_years = np.arange(nsteps)*dt/(60*60*24*365)

T_eq = {}
T_all = {}

#----Run through simulation-------------

for F in F_values:
    T=0.0
    T_series=[]
    
    for i in steprange:
        a = (F + lam*T + a*T**2 + beta*T**5) / C
        b = (sigma / C) * np.sqrt(dt) * np.random.normal(0, 1)
        T = T +a*dt +b
        T_series.append(T)
        
    T_series = np.array(T_series)
    T_all[F] = T_series
    T_eq[F] = T_series[-1]
    
for F in F_values:
    plt.plot(time_years, T_all[F], label="F = {:.2f} W/m$^2$".format(F))

plt.xlabel("Time (years)")
plt.ylabel("ΔT (K)")
plt.title("Delta T with noise")
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.xlim(0,years)
plt.show()