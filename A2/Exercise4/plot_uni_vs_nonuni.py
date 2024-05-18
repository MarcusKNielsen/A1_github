import numpy as np
import matplotlib.pyplot as plt
from Burgersadvection import *


m = 200
eps = 0.01/np.pi
T = 1.6037/np.pi

Tarr, Uarr, x, h = solve_Burgers(T,m,eps,U_initial, non_uni=True)

X,T_mesh = np.meshgrid(x,Tarr)

#%%

fs = 12

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

plt.suptitle(r"Uniform Grid", fontsize=fs+2)

ax = axes[0]
ax0 = ax.pcolormesh(X, T_mesh, Uarr)
ax.set_title("Numerical Solution",fontsize=fs)
ax.set_xlabel("x: space",fontsize=fs)
ax.set_ylabel("t: time",fontsize=fs)
fig.colorbar(ax0, ax=ax)

ax = axes[1]
ax.plot(x,Uarr[-1],"o-",markersize=4)
ax.set_title("Numerical Solution at Terminal Time",fontsize=fs)
ax.set_xlabel("x: space",fontsize=fs)
ax.set_ylabel("u: solution",fontsize=fs)

fig.subplots_adjust(wspace=0.4,bottom=0.2,top=0.85)
plt.show()

#%%

x = np.linspace(-1,1,30)
vals = np.zeros_like(x)

a_list = [1, 0.5, 0.2]

plt.figure(figsize=(8, 4))

plt.title("Non-Uniform Grid")

i = 1
for a in a_list:
    plt.plot(g(x,a),vals + i,".",label = f"a={a}")
    i -= 1
    
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.xlabel("x: space")
plt.axis([-1.1,1.1,-2.0,2.5])
plt.legend()
plt.show()
    