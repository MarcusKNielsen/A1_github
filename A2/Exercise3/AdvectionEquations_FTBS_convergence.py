import numpy as np
from scipy.sparse import diags
from AdvectionEquations_FTBS import*
import matplotlib.pyplot as plt

a_input = 0.5 
E = []
H = []

s = np.arange(5,15)
for s_i in s:

    m = 2**s_i - 1
    h = 2/(m+1)

    Tarr, Uarr, x = solve_advection(m,a_input)
    err = solution_check(Tarr, Uarr, x, a_input, False)

    E.append(np.max(np.abs(err)))
    H.append(h)

a,b = np.polyfit(np.log(H),np.log(E),1)
plt.figure()
print(a)
plt.plot(np.log(H),np.log(E),"bo-",label="convergence test")
plt.plot(np.log(H),a*np.log(H)+b,"r-",label="Helper line of order 1")
plt.legend()
plt.show()





    