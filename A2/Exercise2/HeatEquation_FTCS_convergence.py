import numpy as np
from scipy.sparse import diags
from HeatEquation_FTCS import*
import matplotlib.pyplot as plt

eps = 0.1
E = []
H = []

s = np.arange(2,10)
for s_i in s:

    m = 2**s_i - 1
    h = 2/(m+1)

    Tarr, Uarr, x = solve_diffusion(m,eps)
    err = solution_check(Tarr, Uarr, x, eps, False)

    E.append(np.max(np.abs(err)))
    H.append(h)

ms = 14

a,b = np.polyfit(np.log(H),np.log(E),1)
plt.figure()
plt.plot(np.log(H),np.log(E),"bo-",label="Empirical")
plt.plot(np.log(H),2*np.log(H)+3,"r-",label=r"$O(h^2)$")
plt.xlabel(r"$\log(h)$",fontsize=ms)
plt.ylabel(r"$\log(\Vert E^N \Vert_\infty )$",fontsize=ms)
plt.title(r"Convergence Test with $k=h^2$",fontsize=ms+1)
plt.legend(fontsize=ms-1)
plt.subplots_adjust(left=0.15,bottom=0.15)
plt.show()





    