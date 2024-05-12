import numpy as np
import matplotlib.pyplot as plt
from stability_plot import get_stability_mesh

M = 20
eps = 0.1

# 1

tau = [1, 0.003, 0.0025, 0.0015]

stability_region = np.zeros([M,M,len(tau)])

for i in range(len(tau)):

    stability_region[:,:,i], k, h = get_stability_mesh(eps,M,tau[i])


#%%


H = np.linspace(10**(-2.2), 0.086, M)

def stability_cut(h,eps):
    return 1/(2*eps/h**2 + 2/h)

fs = 12 # fontsize

n,m = 2,2

fig, axes = plt.subplots(n, m, figsize=(12, 12))

plt.suptitle(r"Empirical: $S_e$", fontsize=fs+10)

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
        
        #ax.plot(H,eq_con(H),"r--",label=r"$\dfrac{2 \epsilon k}{h^2} + \dfrac{2k}{h} = 1$")
        ax.plot(H,stability_cut(H,eps),"r--",label=r"$\dfrac{2 \epsilon k}{h^2} + \dfrac{2k}{h} = 1$")
        
        ax.legend(fontsize=fs+1)
        
        it += 1

plt.subplots_adjust(wspace=0.1, hspace=0.3, bottom=0.1,left=0.1, right=0.99, top = 0.9)
plt.show()



