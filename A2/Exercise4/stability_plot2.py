import numpy as np
import matplotlib.pyplot as plt
from stability_plot import get_stability_mesh

M = 50
eps = 0.1

# 1

tau = [1, 0.003, 0.0025, 0.0015]

stability_region = np.zeros([M,M,len(tau)])

for i in range(len(tau)):

    stability_region[:,:,i], k, h = get_stability_mesh(eps,M,tau[i])


#%%

from scipy.optimize import fsolve

def eq_con(x1):
    def g(x2):
        return 2*eps*x2/x1**2 + 2*x2/x1 - 1
    
    # Check if x1 is a scalar (single value)
    if np.isscalar(x1):
        # Handle scalar input
        root = fsolve(g, 0)
        return root[0]
    else:
        # Handle array input
        return np.array([eq_con(element) for element in x1])

H = np.linspace(10**(-2.2), 0.086, M)

#%%

fs = 12 # fontsize

n,m = 2,2

fig, axes = plt.subplots(n, m, figsize=(12, 12))

plt.suptitle(r"Empirical: $S_e$", fontsize=fs+10)


stability_cut = (np.abs(2*eps*k/h**2 + 2*k/h - 1) < 10**(-3))

it = 0
for i in range(n):
    for j in range(m):
        ax = axes[i,j]
        contour = ax.contourf(h, k, stability_region[:,:,it], levels=[-0.5, 0.5, 1.5], cmap='viridis')
        ax.grid(c='k', ls='-', alpha=0.3)
        ax.set_xlabel("h: spatial step",fontsize=fs+1)
        ax.set_ylabel("k: time step",fontsize=fs+1)
        ax.text(0.08, 0.0065, 'Stable', fontsize=fs+3, va='top', ha='right', color='black')
        ax.text(0.02, 0.014, 'Unstable', fontsize=fs+3, va='top', ha='left', color='white')
        ax.set_title(r"$\tau = $" + f"{tau[it]}",fontsize=fs+3)
        cbar = fig.colorbar(contour, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['0.0', '1.0'],fontsize=fs)
        
        ax.plot(H,eq_con(H),"r--",label=r"$\dfrac{2 \epsilon k}{h^2} + \dfrac{2k}{h} = 1$")
        
        ax.legend(fontsize=fs+1)
        
        it += 1

plt.subplots_adjust(wspace=0.1, hspace=0.3, bottom=0.1,left=0.1, right=0.99, top = 0.9)
plt.show()



