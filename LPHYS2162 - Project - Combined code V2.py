# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 15:04:39 2025

LPHYS2162 - Feedback temperature dependence - all

@author: charl
"""
import numpy as np
import matplotlib.pyplot as plt

#----------------------------Part 1--------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#/-/-/-/-/-/-/-Recreating the graphs-/-/-/-/-/-/-/-/-/-/-/-/-/--/-/-/-/-/-/-/-/-/-/-/-/-/--/-/-/-/-/-/-/-/-/-/-/-
#/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
#-----------Fixed parameters------------------------
T0 = 287
CO2_0 = 270
N_T0_C0 = 0


#-----------Define functions-------------------------
def N(Temps, lambda1, a, F):
    deltaT = Temps - T0
    forcing = (lambda1*deltaT) + (a*(deltaT**2)) + F
    return forcing


def N_lin(Temps, lambda1, F):
    deltaT = Temps - T0
    forcing_lin = lambda1*deltaT + F
    return forcing_lin


#----------Plot 1a------------------------------------
#-----------------------------------------------------

#----------Variables--------------------------------
lam = -0.88
Temps = np.arange(285,297,0.1)
zeroline =[]

for i in Temps:
    zeroline = np.append(zeroline,0)


ac = -0.035
am = 0.03
ah = 0.058

F2x = 3.71


Nlin = N_lin(Temps, lam, F2x)
Nac = N(Temps, lam, ac, F2x)
Nam = N(Temps, lam, am, F2x)
Nah = N(Temps, lam, ah, F2x)

Temps_small = np.arange(286,288.01,0.1)
Nlinsmall = N_lin(Temps_small,lam,F2x)


#Plot parameter
plt.plot([], [], ' ', label="λ = -0.88 W/m$^2$/K")

plt.plot(Temps,zeroline, color = 'black')
plt.plot(Temps,Nlin, '--', color = 'black', label="Linear")
plt.plot(Temps,Nac, 'b-', label="a$_c$=-0.035, ΔT$_{2x}$=3.7K")
plt.plot(Temps,Nam, 'g-', label="a$_m$=0.03, ΔT$_{2x}$=5.1K")
plt.plot(Temps,Nah, 'r-', label="a$_h$=0.058, ΔT$_{2x}$=?")

plt.plot(Temps_small, Nlinsmall,  color = 'magenta', label="ΔT<2K (286K-288K)")

plt.xlabel("T (K)")
plt.ylabel("N (W/m$^2$)")
plt.ylim((-2,8))
plt.xlim((285,297))
plt.legend()
plt.title("")
plt.grid(True)
plt.show()

#----------Plot 1d------------------------------------
#-----------------------------------------------------
lam = -1.28
Temps = np.arange(286,300,0.1)

zeroline =[]

for i in Temps:
    zeroline = np.append(zeroline,0)

ah = 0.058

F2x = 3.71
F4x = 7.42



Nlin2x = N_lin(Temps, lam, F2x)
Nlin4x = N_lin(Temps, lam, F4x)
N2x = N(Temps, lam, ah, F2x)
N4x = N(Temps, lam, ah, F4x)



#-------------What should F be in these????--------------
Nlin1 = N_lin(Temps, -1.28, 1)
Nquad1 = N(Temps, -1.28, 0.058, 1)
Nlin2 = N_lin(Temps, -1.28, 2)
Nquad2 = N(Temps, -1.28, 0.058, 2)
#--------------------------------------------------------

plt.plot([], [], ' ', label="λ = -1.28 W/m$^2$/K")
plt.plot([], [], ' ', label="a$_h$ = 0.058 W/m$^2$/K")

plt.plot(Temps,zeroline, color = 'black')
plt.plot(Temps,Nlin1, '--', color = 'black')
plt.plot(Temps,Nquad1, color = 'black')
plt.plot(Temps,Nlin2, '--', color = 'black')
plt.plot(Temps,Nquad2, color = 'black')
plt.plot(Temps,Nlin2x, 'b--', label="2x CO$_2$ (Linear)")
plt.plot(Temps,N2x, 'b', label="2x CO$_2$ (Quad)")
plt.plot(Temps,Nlin4x, 'r--', label="4x CO$_2$ (Linear)")
plt.plot(Temps,N4x, 'r', label="4x CO$_2$ (Quad)")


plt.xlabel("T (K)")
plt.ylabel("N (W/m$^2$)")
plt.ylim((-6,20))
plt.xlim((286,300))
plt.legend()
plt.grid(True)
#plt.legend(["2x CO$_2$ (Linear)","2x CO$_2$ (Quad)", "4x CO$_2$ (Linear)", "4x CO$_2$ (Quad)"],)
plt.show()

#----------Plot 2a & 2b + combined one----------------------------------
#-----------------------------------------------------------------------

#-----------Fixed parameters------------------------
lam_max = -0.79
lam_mean = -1.17
lam_min = -1.78
lam_0 = 0.0000001

F2 = 3.71
F4 = F2*2

cap = 8.36*(10**8)

a = np.arange(-0.1,0.1,0.001)

#---------Function-----------------------

def del_T(lam, a, F):
     A = a
     B = lam
     C = F
     D = (B**2)-(4*A*C)
     D = np.where(D>=0, D, np.nan)
     result_plus = np.where((A < -0.00000001) | (A > 0.00000001), -(B+np.sqrt(D))/(2*A), -F/lam)
     return result_plus


#--------------------------------

del_T_0_2x_plus= del_T(lam_0,a,F2)
del_T_max_2x_plus= del_T(lam_max,a,F2)
del_T_mean_2x_plus= del_T(lam_mean,a,F2)
del_T_min_2x_plus= del_T(lam_min,a,F2)


plt.plot(a,del_T_0_2x_plus, color = 'k', label="λ=0")
plt.plot(a,del_T_max_2x_plus, color = 'magenta', label="λ$_{Max}$ = -0.79")
plt.plot(a,del_T_mean_2x_plus, color = 'green', label="λ$_{Mean}$ = -1.17")
plt.plot(a,del_T_min_2x_plus, color = 'yellow', label="λ$_{Min}$ = -1.78")
plt.legend()
plt.title("2xCO$_2$")
plt.ylabel("ΔT$_{2x}$ (K)")
plt.xlabel("a (W/m$^2$/k$^2$)")
plt.yticks(np.arange(0,11,2))
plt.ylim(0, 10)
plt.grid(True)
plt.show()

del_T_max_4x_plus= del_T(lam_max,a,F4)
del_T_mean_4x_plus= del_T(lam_mean,a,F4)
del_T_min_4x_plus= del_T(lam_min,a,F4)
del_T_0_4x_plus= del_T(lam_0,a,F4)



plt.plot(a,del_T_0_4x_plus, color = 'k', label="λ=0")
plt.plot(a,del_T_max_4x_plus, color = 'magenta', label="λ$_{Max}$ = -0.79")
plt.plot(a,del_T_mean_4x_plus, color = 'green', label="λ$_{Mean}$ = -1.17")
plt.plot(a,del_T_min_4x_plus, color = 'yellow', label="λ$_{Min}$ = -1.78")
plt.legend()
plt.yticks(np.arange(0,21,5))
plt.title("4xCO$_2$")
plt.ylabel("ΔT$_{4x}$ (K)")
plt.xlabel("a (W/m$^2$/k$^2$)")
plt.ylim(0, 20.1)
plt.grid(True)
plt.show()


#plt.plot(a,del_T_0_2x_plus, color = 'k', label="2x - λ=0")
#plt.plot(a,del_T_0_4x_plus, '--',color = 'k', label="4x - λ=0")
plt.plot(a,del_T_max_2x_plus, color = 'magenta', label="2x - λ$_{Max}$ = -0.79")
plt.plot(a,del_T_mean_2x_plus, color = 'green', label="2x - λ$_{Mean}$ = -1.17")
plt.plot(a,del_T_min_2x_plus, color = 'yellow', label="2x - λ$_{Min}$ = -1.78")
plt.plot(a,del_T_max_4x_plus, '--', color = 'magenta', label="4x - λ$_{Max}$ = -0.79")
plt.plot(a,del_T_mean_4x_plus, '--', color = 'green', label="4x - λ$_{Mean}$ = -1.17")
plt.plot(a,del_T_min_4x_plus, '--', color = 'yellow', label="4x - λ$_{Min}$ = -1.78")
plt.legend()
plt.ylabel("ΔT$_{4x}$ (K)")
plt.xlabel("a (W/m$^2$/k$^2$)")
plt.grid(True)
plt.show()

#/-/-/-/-/-/-/-Additional plot to observe variations of alpha and lambda/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
#/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
#-----------Fixed parameters------------------------
lam_max = -0.79
lam_mean = -1.17
lam_min = -1.78
lam_0 = 0

a_max = 0.06
a_mean = 0.01
a_min = -0.04
a_0 = 0

F = np.arange(-5,5,0.1)

#---------Function-----------------------

def del_T(lam, a, F):
     A = a
     B = lam
     C = F
     D = (B**2)-(4*A*C)
     D = np.where(D>=0, D, np.nan)
     result_plus = np.where((A < -0.00001) | (A > 0.00001), -(B+np.sqrt(D))/(2*A), -F/lam)
     return result_plus
 
#-----------Variations----------------------

lam_max_a_max = del_T(lam_max,a_max,F)
lam_mean_a_max = del_T(lam_mean,a_max,F)
lam_min_a_max = del_T(lam_min,a_max,F)

lam_max_a_mean = del_T(lam_max,a_mean,F)
lam_mean_a_mean = del_T(lam_mean,a_mean,F)
lam_min_a_mean = del_T(lam_min,a_mean,F)

lam_max_a_min = del_T(lam_max,a_min,F)
lam_mean_a_min = del_T(lam_mean,a_min,F)
lam_min_a_min = del_T(lam_min,a_min,F)

plt.plot(F,lam_max_a_max,label="λ$_{Max}$, a$_{Max}$")
plt.plot(F,lam_mean_a_max,label="λ$_{Mean}$, a$_{Max}$")
plt.plot(F,lam_min_a_max,label="λ$_{Min}$, a$_{Max}$")

plt.legend()
plt.title("ΔT as a function of F")
plt.ylabel("ΔT (K)")
plt.xlabel("F (W/m$^2$)")
#plt.yticks(np.arange(0,11,2))
#plt.ylim(0, 10)
plt.grid(True)
plt.show()

plt.plot(F,lam_max_a_mean,label="λ$_{Max}$, a$_{Mean}$")
plt.plot(F,lam_mean_a_mean,label="λ$_{Mean}$, a$_{Mean}$")
plt.plot(F,lam_min_a_mean,label="λ$_{Min}$, a$_{Mean}$")

plt.legend()
plt.title("ΔT as a function of F")
plt.ylabel("ΔT (K)")
plt.xlabel("F (W/m$^2$)")
#plt.yticks(np.arange(0,11,2))
#plt.ylim(0, 10)
plt.grid(True)
plt.show()

plt.plot(F,lam_max_a_min,label="λ$_{Max}$, a$_{Min}$")
plt.plot(F,lam_mean_a_min,label="λ$_{Mean}$, a$_{Min}$")
plt.plot(F,lam_min_a_min,label="λ$_{Min}$, a$_{Min}$")

plt.legend()
plt.title("ΔT as a function of F")
plt.ylabel("ΔT (K)")
plt.xlabel("F (W/m$^2$)")
#plt.yticks(np.arange(0,11,2))
#plt.ylim(0, 10)
plt.grid(True)
plt.show()

plt.plot(F,lam_max_a_max,label="λ$_{Max}$, a$_{Max}$")
plt.plot(F,lam_max_a_mean,label="λ$_{Max}$, a$_{Mean}$")
plt.plot(F,lam_max_a_min,label="λ$_{Max}$, a$_{Min}$")
plt.legend()
plt.title("ΔT as a function of F")
plt.ylabel("ΔT (K)")
plt.xlabel("F (W/m$^2$)")
#plt.yticks(np.arange(0,11,2))
#plt.ylim(0, 10)
plt.grid(True)
plt.show()

plt.plot(F,lam_mean_a_max,label="λ$_{Mean}$, a$_{Max}$")
plt.plot(F,lam_mean_a_mean,label="λ$_{Mean}$, a$_{Mean}$")
plt.plot(F,lam_mean_a_min,label="λ$_{Mean}$, a$_{Min}$")
plt.legend()
plt.title("ΔT as a function of F")
plt.ylabel("ΔT (K)")
plt.xlabel("F (W/m$^2$)")
#plt.yticks(np.arange(0,11,2))
#plt.ylim(0, 10)
plt.grid(True)
plt.show()

plt.plot(F,lam_min_a_max,label="λ$_{Min}$, a$_{Max}$")
plt.plot(F,lam_min_a_mean,label="λ$_{Min}$, a$_{Mean}$")
plt.plot(F,lam_min_a_min,label="λ$_{Min}$, a$_{Min}$")
plt.legend()
plt.title("ΔT as a function of F")
plt.ylabel("ΔT (K)")
plt.xlabel("F (W/m$^2$)")
#plt.yticks(np.arange(0,11,2))
#plt.ylim(0, 10)
plt.grid(True)
plt.show()

#-----Subplots-------------

fig, axs = plt.subplots(2, 3, figsize=(15, 8))


axs[0,0].plot(F,lam_max_a_max,label="λ$_{Max}$, a$_{Max}$")
axs[0,0].plot(F,lam_mean_a_max,label="λ$_{Mean}$, a$_{Max}$")
axs[0,0].plot(F,lam_min_a_max,label="λ$_{Min}$, a$_{Max}$")
axs[0,0].legend()
axs[0,0].set_title("ΔT as a function of F")
axs[0,0].set_ylabel("ΔT (K)")
axs[0,0].set_xlabel("F (W/m$^2$)")
axs[0,0].grid(True)


axs[0,1].plot(F,lam_max_a_mean,label="λ$_{Max}$, a$_{Mean}$")
axs[0,1].plot(F,lam_mean_a_mean,label="λ$_{Mean}$, a$_{Mean}$")
axs[0,1].plot(F,lam_min_a_mean,label="λ$_{Min}$, a$_{Mean}$")
axs[0,1].legend()
axs[0,1].set_title("ΔT as a function of F")
axs[0,1].set_ylabel("ΔT (K)")
axs[0,1].set_xlabel("F (W/m$^2$)")
axs[0,1].grid(True)


axs[0,2].plot(F,lam_max_a_min,label="λ$_{Max}$, a$_{Min}$")
axs[0,2].plot(F,lam_mean_a_min,label="λ$_{Mean}$, a$_{Min}$")
axs[0,2].plot(F,lam_min_a_min,label="λ$_{Min}$, a$_{Min}$")
axs[0,2].legend()
axs[0,2].set_title("ΔT as a function of F")
axs[0,2].set_ylabel("ΔT (K)")
axs[0,2].set_xlabel("F (W/m$^2$)")
axs[0,2].grid(True)


axs[1,0].plot(F,lam_max_a_max,label="λ$_{Max}$, a$_{Max}$")
axs[1,0].plot(F,lam_max_a_mean,label="λ$_{Max}$, a$_{Mean}$")
axs[1,0].plot(F,lam_max_a_min,label="λ$_{Max}$, a$_{Min}$")
axs[1,0].legend()
axs[1,0].set_title("ΔT as a function of F")
axs[1,0].set_ylabel("ΔT (K)")
axs[1,0].set_xlabel("F (W/m$^2$)")
axs[1,0].grid(True)


axs[1,1].plot(F,lam_mean_a_max,label="λ$_{Mean}$, a$_{Max}$")
axs[1,1].plot(F,lam_mean_a_mean,label="λ$_{Mean}$, a$_{Mean}$")
axs[1,1].plot(F,lam_mean_a_min,label="λ$_{Mean}$, a$_{Min}$")
axs[1,1].legend()
axs[1,1].set_title("ΔT as a function of F")
axs[1,1].set_ylabel("ΔT (K)")
axs[1,1].set_xlabel("F (W/m$^2$)")
axs[1,1].grid(True)


axs[1,2].plot(F,lam_min_a_max,label="λ$_{Min}$, a$_{Max}$")
axs[1,2].plot(F,lam_min_a_mean,label="λ$_{Min}$, a$_{Mean}$")
axs[1,2].plot(F,lam_min_a_min,label="λ$_{Min}$, a$_{Min}$")
axs[1,2].legend()
axs[1,2].set_title("ΔT as a function of F")
axs[1,2].set_ylabel("ΔT (K)")
axs[1,2].set_xlabel("F (W/m$^2$)")
axs[1,2].grid(True)

plt.tight_layout()
plt.show()


#/-/-/-/-/-/-/-Heat Capacity-/-/-/-/-/-/-/-/-/-/-/-/-/-

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


#----------------------------Part 2--------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#/-/-/-/-/-/-/-Delta T as a function of F with the 5th term/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
#/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-

#-----------Fixed parameters------------------------

lam = -1.28                 
a = 0.058                   
beta = -0.001               
C = 8.36 * 10**8            

# Radiative forcing values
#F_values = np.linspace(0, 20, 10)
F_values = [0,0.15,2,3.45,3.5,3.7]
F_values = [0,2,3.7]

years = 200
dt = 60*60*24                                    # seconds (~1.16 days)
t_end = years * 365 * 24 * 3600
nsteps = int(t_end / dt)

time = np.arange(0, t_end, dt) / (3600 * 24 * 365)

T_eq = {}
T_all = {}

#-----------------Define function----------------------

for F in F_values:

    T = 0.0
    T_series = []

    for i in range(nsteps):        
        dTdt = (F + lam*T + a*(T**2) + beta*(T**5)) / C
        T = T + dTdt * dt
        T_series.append(float(T))

    T_series = np.array(T_series)
    T_all[F] = T_series
    T_eq[F] = T_series[-1]



print("Equilibrium temperature:")
for F in F_values:
    print("F = {:.2f} W/m^2  ->  ΔT_eq ≈ {:.4f} K".format(F, T_eq[F]))


plt.figure(figsize=(10,6))

for F in F_values:
    plt.plot(time, T_all[F], label="F = {:.2f} W/m$^2$".format(F))

plt.xlabel("Time (years)")
plt.ylabel("ΔT (K)")
plt.title("Transient Temperature Response to Radiative Forcing")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#/-/-/-/-/-/-/-Delta T as a function of F with the noise/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
#/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-


#----Set variables---------------------

lam = -1.28
a = 0.058
a = -0.035
beta = -0.001
C = 8.36*10**8
sigma = 7700

#Range of F's for plotting
F_values = [0,2,3.7]

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
plt.legend()
plt.xlim(0,years)
plt.tight_layout()
plt.show()