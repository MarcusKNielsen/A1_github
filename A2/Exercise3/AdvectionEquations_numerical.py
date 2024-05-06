import numpy as np
from scipy.sparse import diags
from AdvectionEquations_FTBS import*
import matplotlib.pyplot as plt

#%% 3.3

a_input = 0.5
Cr = 0.8
m = 100
N = m+2
h = 2/(N-1)
k = (Cr*h)*a_input
T = 3 # wave period is 2 and we need 40 periods, so compute until time T=80
Tarr, Uarr, x = solve_advection(m,a_input,k,T)
x = np.linspace(-1,1,N)
u_ex = u_exact(x,T,a_input)

def g(theta,c):
    return np.sqrt(1+2*c*(c-1)*(1-np.cos(theta)))

T_lin = np.linspace(0,40,100)
G_sol = []

for t in T_lin:
    G_sol.append(g(h*80*np.pi,Cr))

plt.plot(G_sol,label = )

plt.figure()
plt.plot(x,u_ex,label="Exact")
plt.plot(x,Uarr[-1:].ravel(),'-o',label="Sol")
plt.legend()

plt.figure()
theta = np.linspace(0,7,100)
Crlin = np.arange(0,2,0.4)

for c in Crlin:
    plt.plot(theta, g(theta,c), label = f"C = {c}")

plt.xlabel(r"$\theta$")
plt.ylabel(r"$|g(\xi)|$")
plt.legend()
plt.show() 






