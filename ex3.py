import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, kron, eye
from scipy.sparse.linalg import cg
from functions import*
import inspect

m=5
U = np.ones(m*m)
AU = Amult(U,m)
AU2 = poisson_A5(m)@U

#print(np.allclose(AU,AU2))

res_cg = []
iter_cg = []
def report_cg(xk):
    frame = inspect.currentframe().f_back
    res_cg.append(frame.f_locals['resid'])
    iter_cg.append(frame.f_locals['iter_'])

res_pcg = []
iter_pcg = []
def report_pcg(xk):
    frame = inspect.currentframe().f_back
    res_pcg.append(frame.f_locals['resid'])
    iter_pcg.append(frame.f_locals['iter_'])
    
#%% Solving with matlab 
A_sp = poisson_A5(m)
e = np.ones((m*m, 1))
M = spdiags([e.flatten(),-4*e.flatten(),e.flatten()], [-1,0,1], m*m,m*m)
F = poisson_b5(m)
u_cg,info_cg = cg(-A_sp,-F,M=None,callback=report_cg)
u_pcg,info_pcg = cg(-A_sp,-F,M=M,callback=report_pcg)

print("cond of precond:",np.linalg.cond(np.linalg.inv(M.toarray())@A_sp.toarray()))
print("cond of A", np.linalg.cond(A_sp.toarray()))
plt.figure()
plt.plot(iter_cg,res_cg,color="blue",label="cg method")
plt.plot(iter_pcg,res_pcg,color="green",label="pcg method")
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.legend()
plt.title("Convergence plot of pcg method")
plt.show()








