import numpy as np
from scipy.sparse import diags
from AdvectionEquations_FTBS import*
import matplotlib.pyplot as plt

a_input = 0.5
E = []
H = []

s = np.arange(3,10)
for s_i in s:

    # Compute solution and error
    m = 2**s_i - 1
    Tarr, Uarr, x = solve_advection(m,a_input)
    err = solution_check(Tarr, Uarr, x, a_input, True)

    # append global error
    E.append(np.max(np.abs(err[-1])))
    
    # append mesh size´
    h = 2/(m+1)
    H.append(h)


#%%

# plot for k = h

ms = 14

a,b = np.polyfit(np.log(H),np.log(E),1)
plt.figure()
plt.plot(np.log(H),np.log(E),"bo-",label="Empirical")
plt.plot(np.log(H),1*np.log(H)+3,"r-",label=r"$O(h)$")
plt.xlabel(r"$\log(h)$",fontsize=ms)
plt.ylabel(r"$\log(\Vert E^N \Vert_\infty )$",fontsize=ms)
plt.title(r"Convergence Test with $k=h$",fontsize=ms+1)
plt.legend(fontsize=ms-1)
plt.subplots_adjust(left=0.15,bottom=0.15)
plt.show()

#%%

# plot for k = h/a

ms = 14

plt.figure()
plt.plot(np.log(H),E,"bo-",label="Empirical")
#plt.hlines(0, min(np.log(H)), max(np.log(H)),colors="red",linestyles="--",label="0")
plt.xlabel(r"$\log(h)$",fontsize=ms)
plt.ylabel(r"$\Vert E^N \Vert_\infty$",fontsize=ms)
plt.title(r"Convergence Test with $ak=h$",fontsize=ms+1)
plt.ylim([-np.min(E)*10, np.max(E)*10]) 
plt.legend(fontsize=ms-1)
plt.subplots_adjust(left=0.15,bottom=0.15)
plt.show()



    