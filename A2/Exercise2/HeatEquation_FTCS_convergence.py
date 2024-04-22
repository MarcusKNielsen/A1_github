import numpy as np
from scipy.sparse import diags
from HeatEquation_FTCS import*
import matplotlib.pyplot as plt

eps = 0.1
E = []
H = []

s = np.arange(2,7)
for s_i in s:

    m = 2**s_i - 1
    h = 1/(m+1)

    Tarr, Uarr, x = solve_diffusion(m,eps)
    err = solution_check(Tarr, Uarr, x, eps, False)

    E.append(np.max(np.abs(err)))
    H.append(h)

a,b = np.polyfit(np.log(H),np.log(E),1)
plt.figure()
plt.plot(np.log(H),np.log(E),"o-","b",label="convergence test")
plt.plot(np.log(H),a*np.log(H)+b,"-","r",label="Helper line of order 2")
print(a)
plt.legend()
plt.show()





    