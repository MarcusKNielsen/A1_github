import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, kron, eye, csr_matrix
from scipy.sparse.linalg import cg, inv
from functions import*
import inspect

#%% Problem 3.a

m=5
U = np.ones(m*m)
AU = Amult(U,m)
AU2 = poisson_A5(m)@U

xk_cg = []
res_cg = []
iter_cg = []
def report_cg(xk):
    frame = inspect.currentframe().f_back
    xk_cg.append(xk)
    res_cg.append(frame.f_locals['resid'])
    iter_cg.append(frame.f_locals['iter_'])

xk_pcg = []
res_pcg = []
iter_pcg = []
def report_pcg(xk):
    frame = inspect.currentframe().f_back
    xk_pcg.append(xk)
    res_pcg.append(frame.f_locals['resid'])
    iter_pcg.append(frame.f_locals['iter_'])
    
#%% Conjugate gradient method 
<<<<<<< HEAD
""" A_sp = poisson_A5(m)
=======

def error_bound(cond,k):
    return 2 * ( (np.sqrt(cond) - 1)/(np.sqrt(cond) + 1) )**k

A_sp = poisson_A5(m)
>>>>>>> 5b4243c653f6c2eea2c14837888c73daea6061ba
e = np.ones((m*m, 1))
M = inv(spdiags([e.flatten(),e.flatten(),e.flatten(),-10*e.flatten(),e.flatten(),e.flatten(),e.flatten()], [-m-1,-m,-1,0,1,m,m+1], m*m,m*m))
F = poisson_b5(m)
u_cg,info_cg = cg(-A_sp,-F,M=None,callback=report_cg)
u_pcg,info_pcg = cg(-A_sp,-F,M=M,callback=report_pcg)

fig = plt.figure()
plt.title("Sparse Laplacian Matrix Structure")
img = plt.imshow(inv(A_sp).toarray())
fig.colorbar(img)
plt.xlabel("k: node index")
plt.ylabel("k: node index")


condA = np.linalg.cond(-A_sp.toarray())
print("cond of A", condA)
print("cond of precond:",np.linalg.cond(M.toarray()@A_sp.toarray()))

plt.figure()
plt.plot(iter_cg,res_cg,"-o",color="blue",label="cg method")
plt.plot(iter_pcg,res_pcg,"-o",color="green",label="pcg method")
plt.plot(iter_cg,error_bound(condA,iter_cg))
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.legend()
plt.title("Convergence plot of pcg method") """

#%% Jacobi and under relaxed Jacobi


omega = np.linspace(0,2,50)
m = np.arange(10,100,10)
max_eig = np.zeros(len(omega))

fs = 12

fig = plt.figure(figsize=(8, 4))  # Adjust figure size as needed

for mi in m:
    h = 1/(1+mi)
    p = np.arange(mi/2,mi+1)
    q = np.arange(mi/2,mi+1)

    for idx, o in enumerate(omega):
        eig = np.array([[eigenvalues_5point_relax(h,pi,qj,o) for pi in p] for qj in q])
        max_eig[idx] = np.max(np.abs(eig))
    
    plt.plot(omega, max_eig, label=f"m={mi}")

<<<<<<< HEAD
plt.hlines(1,xmin=np.min(omega),xmax=np.max(omega),label="threshold of 1",color="black",linestyle="dashed")
plt.vlines(2/3,ymin=0,ymax=3,color="red",linestyle="dashed",label=r"Optimal $\omega=\frac{2}{3}$ ")
plt.legend()
=======
plt.title("Minimizing eigenvalues")
plt.xlabel(r"$\omega$",fontsize=fs)
plt.ylabel(r"$\max_{\dfrac{m}{2}\leq p,q \leq m}|\gamma_{p,q}|$",fontsize=fs)
>>>>>>> 5b4243c653f6c2eea2c14837888c73daea6061ba

plt.hlines(1, xmin=np.min(omega), xmax=np.max(omega), label="threshold of 1", color="black", linestyle="dashed")
plt.vlines(2/3, ymin=0, ymax=3, color="red", linestyle="dashed", label=r"$\omega = \frac{2}{3}$")

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside of plot

plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()

#%% Smooth function test
omega_opt= 2/3
m = 5 
U = np.ones(m*m)
# Right hand side of system of equations
F = poisson_b5(m)

Uk = smooth(U,omega_opt,m,F)
print("Test of smooth solver: Uk=",Uk)

#%%














