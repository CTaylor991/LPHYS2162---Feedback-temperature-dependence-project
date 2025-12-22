# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 15:23:56 2025

@author: charl
"""

import numpy as np
import matplotlib.pyplot as plt

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