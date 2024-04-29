import numpy as np
from scipy.sparse import diags
from AdvectionEquations_FTBS import*
import matplotlib.pyplot as plt

a_input = 0.5
Cr = 0.8
m = 98
N = m+2
h = 2/(N-1)
k = (Cr*h)*a_input
T = 15
Tarr, Uarr, x = solve_advection(m,a_input,k,T)
x = np.linspace(-1,1,N)
u_ex = u_exact(x,T,a_input)

plt.figure()
plt.plot(x,u_ex,label="Exact")
plt.plot(x,Uarr[-1:].ravel(),'-o',label="Sol")
plt.legend()
plt.show()





