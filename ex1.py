
#%% Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math

#%% Exercise c
def u(x):
    return np.exp(np.cos(x))

# The second derivative evaluated in 0:
exact_value = -np.exp(1)

def du(h,x,alpha,beta,a):
    # alpha: start stencil
    # beta: end stencil 
    # h: the stencil size
    # a: coefficients depending on FDM
    
    # The approximation to an arbitrary derivative
    result = 0
    for i,a in zip(range(-alpha, beta+1), a):
        result += u(x+i*h)*a
    
    return result

#%% Order of accuracy 

# The mesh size
h = 0.1 
# The coefficient from FDM of second order derivative backword and centered using 5 points
a_backward = np.array([11/12,-56/12,114/12,-104/12,35/12])/(h**2)
a_centered = np.array([-1/12,16/12,-30/12,16/12,-1/12])/(h**2)
print("Approximating second derivative using different stencils:")
print("Exact value:",exact_value)
print("Backward:", du(h=h,x=0,alpha=4,beta=0,a = a_backward))
print("Centered:", du(h=h,x=0,alpha=2,beta=2,a = a_centered))

def C(n,a,alpha,beta,dev):
    # Checking order of accuracy based on the coefficients
    # Following equation 3 from problem description
    result = 0
    if n == dev:
        result -= h**(-n)
    
    for a,m in zip(a,range(-alpha,beta+1)):
        result += a*m**n/(math.factorial(n))

    return result 

print("Backward:",[C(i,a_backward,alpha=4,beta=0,dev=2) for i in range(7)])
print("Centered:",[C(i,a_centered,alpha=2,beta=2,dev=2) for i in range(7)])

#%% Exercise d

# Computing the values of h for different values of s
h_list = np.array([1/(2**s) for s in range(2,20)])

# Initializing error lists
error_backword = abs(exact_value - du(h=h_list,x=0,alpha=4,beta=0,a = a_backward))
error_centered = abs(exact_value - du(h=h_list,x=0,alpha=2,beta=2,a = a_centered))

# Convergence rates 
CR_backword = 3
CR_centered = 4

# Making subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))
ax1.plot(np.log10(h_list),np.log10(error_backword),"-o",label="backward FDM")
#ax1.plot(np.log10(h_list),CR_backword*np.log10(h_list))
ax1.plot(np.log10(h_list),CR_centered*np.log10(h_list),label="Helper line of order 4")
ax1.set_title('Backward operator')  
ax1.set_xlabel(r'$\log(h)$')
ax1.set_ylabel(r'$\log(\Vert \hat{u} - u \Vert )$') 
ax1.legend()

ax2.plot(np.log10(h_list),np.log10(error_centered),"-o",label="Centered FDM")
ax2.plot(np.log10(h_list),CR_centered*np.log10(h_list),label="Helper line of order 4")
ax2.set_title('Centered operator')
ax2.set_xlabel(r'$\log(h)$')
ax2.set_ylabel(r'$\log(\Vert \hat{u} - u \Vert) $')  
ax2.legend()
plt.tight_layout()

plt.show()
