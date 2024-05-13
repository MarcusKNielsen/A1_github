import numpy as np
import matplotlib.pyplot as plt
from stability_plot import get_stability_mesh
from Burgersadvection import *

M = 50
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

def stability_cut2(h,eps):
    return 1/(2*eps/h**2 + 1/h)

H2 = np.linspace(10**(-2.2), 0.073, M)
    

h_test = np.array([0.0200, 0.0525, 0.0600, 0.0730, 0.0700, 0.0790, 0.0800])
k_test = np.array([0.0025, 0.0100, 0.0125, 0.0175, 0.0175, 0.0175, 0.0175])
m_test = np.ceil(2/h_test - 2)
h_test = 2/(m_test+2)


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
        
        
        ax.plot( H, stability_cut(H,eps) ,"g--",label=r"$\dfrac{2 \epsilon k}{h^2} + \dfrac{2k}{h} = 1$")
        ax.plot(H2, stability_cut2(H2,eps),"r--",label=r"$\dfrac{2 \epsilon k}{h^2} + \dfrac{ k}{h} = 1$")
        
        ax.plot(h_test,k_test,".",color="red",markersize=10)
        
        ax.legend(fontsize=fs+1)
        
        it += 1


plt.subplots_adjust(wspace=0.1, hspace=0.3, bottom=0.1,left=0.1, right=0.99, top = 0.9)
plt.show()


#%%


eps = 0.1
T = 1.0

for i in range(len(m_test)):

    mi = int(m_test[i])
    ki = k_test[i]
    
    t,U,x,_ = solve_Burgers(T,mi,ki,eps,U_exact,non_uni = False)
    solution_check(t, U, x, eps, U_exact, exact = True)
    
    U_test1 = U[-1,:]
    
    plt.figure()
    plt.plot(x,U_test1,label=f"m={mi+2},h={np.round(2/(mi+2),3)},k={ki}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("t=1")
    plt.legend()
    plt.show()














