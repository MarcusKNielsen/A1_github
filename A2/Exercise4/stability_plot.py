import numpy as np
import matplotlib.pyplot as plt
from Burgersadvection import *

def get_stability_mesh(eps,M,tol):

    H = np.linspace(10**(-4), 10**(-1.0), M)
    K = np.linspace(10**(-4), 10**(-1.7), M)
    m = np.ceil(2/H - 2)
    H = 2/(m+2)
    
    h, k = np.meshgrid(H, K)
    
    
    
    stability_region = np.zeros([M,M])
    
    T = 1
    
    n = 0
    N = M*M
    
    
    for i in range(M):
        for j in range(M):
            m = 2/h[i,j] - 2 
            stability_region[i,j] = solve_Burgers_stability_test(T,int(m),h[i,j],k[i,j],eps,U_exact,U_dx_exact,tol)
            
            if n % 50 == 0:
                print(f"Progress: {np.round((n/N) * 100,1)} %")
            
            n += 1
    return stability_region, k, h

#%%

if __name__ == "__main__":

    M = 20
    tol = 0.001
    eps = 0.1
    
    stability_region, k, h = get_stability_mesh(eps,M,tol)
    
    #%%
    
    fs = 12 # fontsize
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Prediction plot
    stability_region_pred = (2*eps*k/h**2 + 2*k/h <= 1)
    ax = axes[1]
    contour = ax.contourf(h, k, stability_region_pred, levels=[-0.5, 0.5, 1.5], cmap='viridis')
    ax.grid(c='k', ls='-', alpha=0.3)
    ax.set_xlabel("h: spatial step",fontsize=fs+1)
    ax.set_ylabel("k: time step",fontsize=fs+1)
    ax.text(0.08, 0.0065, 'Stable', fontsize=fs+3, va='top', ha='right', color='black')
    ax.text(0.01, 0.015, 'Unstable', fontsize=fs+3, va='top', ha='left', color='white')
    ax.set_title(r"Prediction: $S_p$",fontsize=fs+3)
    cbar = fig.colorbar(contour, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0.0', '1.0'],fontsize=fs)
    
    # Empirical plot
    ax = axes[0]
    contour = ax.contourf(h, k, stability_region, levels=[-0.5, 0.5, 1.5], cmap='viridis')
    ax.grid(c='k', ls='-', alpha=0.3)
    ax.set_xlabel("h: spatial step",fontsize=fs+1)
    ax.set_ylabel("k: time step",fontsize=fs+1)
    ax.text(0.08, 0.0065, 'Stable', fontsize=fs+3, va='top', ha='right', color='black')
    ax.text(0.01, 0.015, 'Unstable', fontsize=fs+3, va='top', ha='left', color='white')
    ax.set_title(r"Empirical: $S_e$",fontsize=fs+3)
    cbar = fig.colorbar(contour, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0.0', '1.0'],fontsize=fs)  # Set text labels
    
    # Discrepancy plot
    ax = axes[2]
    discrepancy = stability_region - stability_region_pred 
    contour = ax.contourf(h, k, discrepancy, levels=[-1.5, -0.5, 0.5, 1.5], cmap='viridis')
    ax.grid(c='k', ls='-', alpha=0.3)
    ax.set_xlabel("h: spatial step",fontsize=fs+1)
    ax.set_ylabel("k: time step",fontsize=fs+1)
    ax.set_title(r"discrepancy: $S_e-S_p$",fontsize=fs+3)
    cbar = fig.colorbar(contour, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels([r'-1.0','0.0', '1.0'],fontsize=fs)
    
    plt.subplots_adjust(wspace=0.3, bottom=0.15,left=0.07)
    plt.show()

