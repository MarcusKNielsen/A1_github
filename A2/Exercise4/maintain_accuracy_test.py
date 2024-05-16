import numpy as np
import matplotlib.pyplot as plt
from Burgersadvection import *

eps = 0.01/np.pi
T = 1.6037/np.pi

m_test = np.array([100])
m_test = m_test + 1

m = m_test[0]

a = 0.005
t,U,x,h,k,n = solve_Burgers_low_memory_a_input(T,m,eps,a,U_initial,non_uni=True)

x_idx = np.argsort(abs(x))[:3]
x_idx = np.sort(x_idx)

hl = (x[x_idx[1]] - x[x_idx[0]])
hr = (x[x_idx[2]] - x[x_idx[1]])


Du_left    = (U[x_idx[1]] - U[x_idx[0]])/hl
Du_central = (U[x_idx[2]] - U[x_idx[0]])/(hr+hl)
Du_right   = (U[x_idx[2]] - U[x_idx[1]])/hr

plt.figure()
plt.plot(x,U,"-o")
plt.show()

#%%

m_vals = [200,400,600,800,1000]
a_vals = [0.14, 0.37, 0.57, 0.78, 1.0]

plt.figure()
plt.plot(m_vals,a_vals,"o-")
plt.xlabel(r"$m$: number of grid points")
plt.ylabel(r"$a$:parameter")
plt.show()