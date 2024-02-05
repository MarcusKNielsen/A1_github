
#%% Importing libraries
import math
import matplotlib.pyplot as plt
import numpy as np


#%% Exercise c
def u(x):
    return math.exp(math.cos(x))
    #return math.exp(x)

#exact_value = math.exp(0)
exact_value = -math.exp(1)

def d2u_b(h):
    # (4,0)
    x=0
    #a1 = [11/(12*h**2), -14/(3*h**2), 19/(2*h**2), -26/(3*h**2), 35/(12*h**2)]
    a1 = [11/(12*h**2), -56/(12*h**2), 114/(12*h**2), -104/(12*h**2), 35/(12*h**2)]
    d2u_1 = 0
    for i,a in zip(range(-4, 1),a1):
        d2u_1 += u(x+i*h)*a
    
    return d2u_1

def d2u_c(h):
    # (2,2)
    x=0
    #a2 = [-1/(12*h**2), 4/(3*h**2), -5/(2*h**2), 4/(3*h**2), -1/(12*h**2)]
    a2 = np.array([-1/12,16/12,-30/12,16/12,-1/12])/(h**2)
    d2u_2 = 0
    for i,a in zip(range(-2, 3),a2):
        d2u_2 += u(x+i*h)*a
    
    return d2u_2

print("Exact value:",exact_value)
print("Backward:", d2u_b(0.001))
print("Centered:", d2u_c(0.001))

#%% Checking order

h = 0.1
a_forward = np.array([35/12, -104/12, 114/12,-56/12, 11/12])/(h**2)
a_backward = np.array([11/12,-56/12,114/12,-104/12,35/12])/(h**2)
a_centered = np.array([-1/12,16/12,-30/12,16/12,-1/12])/(h**2)

def C(n,a,alpha,beta):
    out = 0
    
    if n == 2:
        out -= h**(-n)
    
    for a,m in zip(a,range(-alpha,beta+1)):
        out += a*m**n/(math.factorial(n))

    return out 

print("Forword:",[C(i,a_forward,0,4) for i in range(7)])
print("Backword:",[C(i,a_backward,4,0) for i in range(7)])
print("Centered:",[C(i,a_centered,2,2) for i in range(7)])

#%% Exercise d

# Computing the values of h for different values of s
h_list = [1/(2**s) for s in range(2,20)]

# Initializing error lists
error_backword = [abs(exact_value - d2u_b(h_list[i])) for i in range(len(h_list))]
error_centered = [abs(exact_value - d2u_c(h_list[i])) for i in range(len(h_list))]

# Convergence rates 
CR_backword = 3
CR_centered = 4

# Making subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))
#ax1.loglog(h_list, error1, "-o")
ax1.plot(np.log10(h_list),np.log10(error_backword),"-o")
ax1.plot(np.log10(h_list),CR_backword*np.log10(h_list))
ax1.plot(np.log10(h_list),CR_centered*np.log10(h_list))
ax1.set_title('Backward operator')  
ax1.set_xlabel(r'$\log(h)$')
ax1.set_ylabel(r'$\log(\Vert \hat{u} - u \Vert )$') 
#ax2.loglog(h_list, error2, "-o")
ax2.plot(np.log10(h_list),np.log10(error_centered),"-o")
ax2.plot(np.log10(h_list),CR_centered*np.log10(h_list))
ax2.set_title('Centered operator')
ax2.set_xlabel(r'$\log(h)$')
ax2.set_ylabel(r'$\log(\Vert \hat{u} - u \Vert) $')  

plt.tight_layout()

plt.show()


# %%
