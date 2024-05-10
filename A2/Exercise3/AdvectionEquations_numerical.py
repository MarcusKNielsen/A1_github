import numpy as np
from scipy.sparse import diags
from AdvectionEquations_FTBS import*
import matplotlib.pyplot as plt

#%% 3.3

a_input = 0.5
Cr = 0.8
L = 1 # wave length
Nx = 100 # Points pr wave length
delta_x = L/Nx
m = 199 # isolating in h = 2/m = 1/100 = delta_x
N = m+2
h = 2/(N-1)
k = (Cr*h)/a_input
T_wave_period = 2
num_wave_period = 13.7
T = num_wave_period*T_wave_period # wave period is 2 and we need 40 periods, so compute until time T=80

Tarr, Uarr, x = solve_advection(m,a_input,k=k,T=T)

def angle_dispersion(theta,c):
    return np.arctan(c*np.sin(theta)/(1-c*(1-np.cos(theta))))

print(angle_dispersion(2*np.pi*h,Cr))

u_ex = u_exact(x,T,a_input)

def g_mag(theta,c):
    return np.sqrt(1+2*c*(c-1)*(1-np.cos(theta)))

y = angle_dispersion(2*np.pi*h,Cr)
n = (Tarr[-1]/k)
U0 = u_exact(x-n*y/(2*np.pi),0,a_input)
G_sol = U0*g_mag(2*np.pi*h,Cr)**(Tarr[-1]/k)
print(f"g(2 pi)  = {g_mag(2*np.pi*h,Cr)}")


plt.figure()
plt.plot(x,u_ex,linewidth=2.0,label="Exact")
plt.plot(x,Uarr[-1:].ravel(),'-',linewidth=2.0,markerfacecolor='none',label="Numerical")
plt.plot(x,G_sol,"--",linewidth=2.0,label = "Prediction")
plt.xlabel("x: space")
plt.ylabel("u: solution value")
plt.title(f"Dispersion and Diffusion at t = {T}")
plt.legend()

#%%

plt.figure()
plt.plot(a_input * Tarr,label=r"$at$")
plt.plot(y * Tarr /(2 * np.pi*k),"--",label=r"$\dfrac{y(2 \pi)}{2 \pi k} t $")
plt.legend()
plt.show()


#%%

plt.figure()
theta = np.linspace(0,7,100)
Crlin = np.arange(0,2,0.4)

for c in Crlin:
    plt.plot(theta, g_mag(theta,c), label = f"C = {c}")

plt.xlabel(r"$\theta$")
plt.ylabel(r"$|g(\xi)|$")
plt.legend()
plt.show() 








