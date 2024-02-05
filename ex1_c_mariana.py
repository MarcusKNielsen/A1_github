
#%% Importing libraries
import math
import matplotlib.pyplot as plt
import numpy as np


#%% Exercise c
def u(x):
    return math.exp(math.cos(x))

exact_value = -math.exp(1)

def d2u_1(h):
    # (4,0)
    a1 = [11/(12*h**2), -14/(3*h**2), 19/(2*h**2), -26/(3*h**2), 35/(12*h**2)]
    d2u_1 = 0
    for i in range(-4, 1):
        d2u_1 += u(i*h)*a1[i+4]
    
    return d2u_1

def d2u_2(h):
    # (2,2)
    a2 = [-1/(12*h**2), 4/(3*h**2), -5/(2*h**2), 4/(3*h**2), -1/(12*h**2)]
    d2u_2 = 0
    for i in range(-2, 3):
        d2u_2 += u(i*h)*a2[i+2]
    
    return d2u_2

print("Backward:", d2u_1(0.001))
print("Centered", d2u_2(0.001))

#%% Exercise d

# Range of h list 
range_s = 20-2
# Computing the values of h for different values of s
h_list = [1/(2**s) for s in range(2,20)]

# Initializing error lists
error1 = []
error2 = []

# Computing the errors
for i in range(range_s):
    error1.append(abs(exact_value - d2u_1(h_list[i])))
    error2.append(abs(exact_value - d2u_2(h_list[i])))

# Convergence rates 
CR_1 = 4
CR_2 = 4

# Making subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))
#ax1.loglog(h_list, error1, "-o")
ax1.plot(np.log10(h_list),np.log10(error1),"-o")
ax1.plot(np.log10(h_list),CR_1*np.log10(h_list))
ax1.set_title('Backward operator')  
ax1.set_xlabel('h')
ax1.set_ylabel('d2u(h)') 
#ax2.loglog(h_list, error2, "-o")
ax2.plot(np.log10(h_list),np.log10(error2),"-o")
ax2.plot(np.log10(h_list),CR_2*np.log10(h_list))
ax2.set_title('Centered operator')
ax2.set_xlabel('h')
ax2.set_ylabel('d2u(h)')  

plt.tight_layout()

#plt.show()

#%% Problem e

a_forward = [35/12, -104/12, 114/12,-56/12, 11/12]
a_backward = [11/12,-56/12,114/12,-104/12,35/12]

h = 1

def C(n,a,alpha,beta):
    out = 0
    
    if n == 2:
        out -= h**(-n)
    
    for i,m in enumerate(range(-alpha,beta+1)):
        
        out += a[i]*m**n/(math.factorial(n))

    return out 

print([C(i,a_forward,0,4) for i in range(7)])
print([C(i,a_backward,4,0) for i in range(7)])


