import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import cg, inv, spsolve
from functions import poisson_A5, poisson_b5
import inspect


xk_cg = []
res_cg = []
iter_cg = []
def report_cg(xk):
    frame = inspect.currentframe().f_back
    xk_cg.append(xk.copy()) 
    res_cg.append(frame.f_locals['resid'])
    iter_cg.append(frame.f_locals['iter_'])

xk_pcg = []
res_pcg = []
iter_pcg = []
def report_pcg(xk):
    frame = inspect.currentframe().f_back
    xk_pcg.append(xk.copy()) 
    res_pcg.append(frame.f_locals['resid'])
    iter_pcg.append(frame.f_locals['iter_'])
    

def error_bound(cond,k):
    return 2 * ( (np.sqrt(cond) - 1)/(np.sqrt(cond) + 1) )**k

#%% setup problem
m=25
A_sp = poisson_A5(m)
e = np.ones((m*m, 1))
M = inv(spdiags([e.flatten(),e.flatten(),e.flatten(),-10*e.flatten(),e.flatten(),e.flatten(),e.flatten()], [-m-1,-m,-1,0,1,m,m+1], m*m,m*m)) 
F = poisson_b5(m)

u0 = np.zeros(m*m)

# solve problem using conjugate gradient
u_cg,info_cg = cg(-A_sp,-F,u0,M=None,callback=report_cg)
u_pcg,info_pcg = cg(-A_sp,-F,u0,M=M,callback=report_pcg)

u_exact = spsolve(A_sp,F)

#%%

xk_cg  = np.array(xk_cg).T
xk_pcg = np.array(xk_pcg).T
err_cg  = xk_cg  - u_exact[:, np.newaxis]
err_pcg = xk_pcg - u_exact[:, np.newaxis]

C = (u0 - u_exact).T @ (-A_sp) @ (u0 - u_exact)

Niter_cg = iter_cg[-1]
Niter_pcg = iter_pcg[-1]
rate_cg = np.zeros(Niter_cg)
rate_pcg = np.zeros(Niter_pcg)
for i in range(Niter_cg):
    rate_cg[i] = np.sqrt(err_cg[:,i].T @ (-A_sp) @ err_cg[:,i])/C
    
for i in range(Niter_pcg):
    rate_pcg[i] = np.sqrt(err_pcg[:,i].T @ (-A_sp) @ err_pcg[:,i])/C

#%%

condA = np.linalg.cond(A_sp.toarray())
condMA = np.linalg.cond((M @A_sp).toarray())

plt.figure()
plt.plot(iter_cg, np.log(rate_cg), "-o")
plt.plot(iter_cg, np.log(error_bound(condA, iter_cg)))
plt.xlabel(r"$k$", fontsize=12)
plt.ylabel(r"$\log\left( \dfrac{\Vert e_k \Vert_A}{\Vert e_0 \Vert_A} \right)$")

plt.plot(iter_pcg, np.log(rate_pcg), "-o")
plt.plot(iter_cg, np.log(error_bound(condMA, iter_cg)))

# Adjust the subplot parameters to give some padding
plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)

plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

# Plot for CG
axs[0].plot(iter_cg, np.log(rate_cg), "-o", label="CG Rate")
axs[0].plot(iter_cg, np.log(error_bound(condA, iter_cg)), label="CG Error Bound")
axs[0].set_xlabel(r"$k$", fontsize=12)
axs[0].set_ylabel(r"$\log\left( \dfrac{\Vert e_k \Vert_A}{\Vert e_0 \Vert_A} \right)$", fontsize=12)
axs[0].legend()

# Plot for PCG
axs[1].plot(iter_pcg, np.log(rate_pcg), "-o", label="PCG Rate")
axs[1].plot(iter_pcg, np.log(error_bound(condMA, iter_pcg)), label="PCG Error Bound")
axs[1].set_xlabel(r"$k$", fontsize=12)
# axs[1].set_ylabel(r"$\log\left( \dfrac{\Vert e_k \Vert_A}{\Vert e_0 \Vert_A} \right)$", fontsize=12) # Optional, as it's the same as the left
axs[1].legend()

# Adjust the subplot parameters to give some padding
plt.subplots_adjust(left=0.12, right=0.95, top=0.89, bottom=0.15)  # Adjust top to make space for the main title

# Add a main title for the entire figure
fig.suptitle("Convergence Test", fontsize=16)

plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns, adjust the figure size as needed

# Plot for A^-1
img1 = axs[0].imshow(inv(A_sp).toarray())  # Assuming A_sp is your sparse matrix
fig.colorbar(img1, ax=axs[0])  # Add a colorbar to the first subplot
axs[0].set_title(r"Matrix structure $A^{-1}$")
axs[0].set_xlabel("k: node index")
axs[0].set_ylabel("k: node index")

# Plot for M^-1
img2 = axs[1].imshow(M.toarray())  # Assuming M is your matrix
fig.colorbar(img2, ax=axs[1])  # Add a colorbar to the second subplot
axs[1].set_title(r"Matrix structure $M^{-1}$")
axs[1].set_xlabel("k: node index")
axs[1].set_ylabel("k: node index")

plt.tight_layout()  # Adjust the layout to make sure everything fits without overlapping
plt.show()




