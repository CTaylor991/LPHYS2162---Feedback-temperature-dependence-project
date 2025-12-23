# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 19:33:24 2025

@author: charl
"""

import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

#-----Set variables
lam = -1.28
a = 0.058
beta = -4*10**-6
C = 8.36*10**8
sigma = 7700
F=0

#Step parameters

years = 300
dt = 60*60*24
t_end = years*365*24*60*60
nsteps = int(t_end/dt)
steprange = range(nsteps)

time_years = np.arange(nsteps)*dt/(60*60*24*365)

#spin up paremeters
spinup_years = 100
spinup_steps = int(spinup_years*365*24*60*60/dt)


# noise parameters
sigma = 7700
n_ensemble = 5000


#--------Define function-----------------------

def integrate_stochastic(F, lam, a, beta, C, sigma, dt, nsteps):
    T = np.zeros(nsteps)
    for i in range(nsteps - 1):
        T_D = (F + lam * T[i] + a * T[i]**2 + beta * T[i]**5) / C
        noise = (sigma/C)*np.sqrt(dt)*np.random.normal(0,1)
        T[i+1] = T[i]+(T_D*dt)+noise
    return T


std_values = []


example_runs = []
example_runs2 = []
example_runs3 = []

n_examples_to_plot = 3
n_examples_to_plot2 = 10
n_examples_to_plot3 = 25

print("Running ensemble of {} simulations...".format(n_ensemble))

for run in range(n_ensemble):
    milestone = int(((run + 1) / n_ensemble) * 100)
    elapsed = round(time.time() - start_time,1)
    if milestone != int((run / n_ensemble) * 100):
        remainingtime = round((100-milestone)*(elapsed/milestone),2)/60
        print(f"Run has completed: {milestone}%, Total time taken: {elapsed}s, remaining time: {remainingtime} minutes")
        #print("Run", run + 1, "of", n_ensemble)
    
    T_series = integrate_stochastic(F, lam, a, beta, C, sigma, dt, nsteps)

    T_stationary = T_series[spinup_steps:]

    std_T = np.std(T_stationary)
    std_values.append(std_T)

    if run < n_examples_to_plot:
        example_runs.append(T_stationary)
    if run < n_examples_to_plot2:
        example_runs2.append(T_stationary)
    if run < n_examples_to_plot3:
        example_runs3.append(T_stationary)

# Convert to NumPy array for convenience
std_values = np.array(std_values)

mean_std = np.mean(std_values)
std_of_std = np.std(std_values)

print("\nNoise amplitude σ = {:.1f} W m^-2".format(sigma))
print("Number of ensemble members =", n_ensemble)
print("Mean std(ΔT) = {:.4f} K".format(mean_std))
print("Std of std(ΔT) = {:.4f} K".format(std_of_std))

#----------Plotting------------------------
time_stationary = np.arange(len(example_runs[0])) * dt / (3600 * 24 * 365)

plt.figure(figsize=(10,5))

for i, T_ex in enumerate(example_runs):
    T_shifted = T_ex - T_ex[0]
    plt.plot(time_stationary, T_shifted, linewidth=0.8, label="Run {}".format(i + 1))

plt.xlabel("Time since spin-up (years)")
plt.ylabel("ΔT (K)")
plt.title(f"Example Ensemble Members (shifted to start at ΔT = 0) (σ={sigma}, {len(example_runs)} of {n_ensemble} ensemble runs)")
plt.grid(True)
plt.xlim(0,years-spinup_years)
#plt.legend()
plt.tight_layout()
plt.show()

time_stationary = np.arange(len(example_runs2[0])) * dt / (3600 * 24 * 365)

plt.figure(figsize=(10,5))

for i, T_ex in enumerate(example_runs2):
    T_shifted = T_ex - T_ex[0]
    plt.plot(time_stationary, T_shifted, linewidth=0.8, label="Run {}".format(i + 1))

plt.xlabel("Time since spin-up (years)")
plt.ylabel("ΔT (K)")
plt.title(f"Example Ensemble Members (shifted to start at ΔT = 0) (σ={sigma}, {len(example_runs2)} of {n_ensemble} ensemble runs)")
plt.grid(True)
plt.xlim(0,years-spinup_years)
#plt.legend()
plt.tight_layout()
plt.show()

time_stationary = np.arange(len(example_runs3[0])) * dt / (3600 * 24 * 365)

plt.figure(figsize=(10,5))

for i, T_ex in enumerate(example_runs3):
    T_shifted = T_ex - T_ex[0]
    plt.plot(time_stationary, T_shifted, linewidth=0.8, label="Run {}".format(i + 1))

plt.xlabel("Time since spin-up (years)")
plt.ylabel("ΔT (K)")
plt.title(f"Ensemble Members (σ={sigma}, {len(example_runs3)} of {n_ensemble} ensemble runs)")
plt.grid(True)
plt.xlim(0,years-spinup_years)
#plt.legend()
plt.tight_layout()
plt.show()


#-Histo--------------------------------

plt.figure(figsize=(8,5))

plt.hist(std_values, bins=30, edgecolor='black')

plt.axvline(mean_std, color='red', linestyle='--',
            label="Mean = {:.3f} K".format(mean_std))

plt.xlabel("Standard deviation of ΔT (K)")
plt.ylabel("Number of ensemble members")
plt.title(f"Ensemble Distribution of Temperature Variability (σ={sigma}, {n_ensemble} ensemble runs)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
