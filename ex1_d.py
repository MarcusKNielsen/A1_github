import math
import matplotlib.pyplot as plt
import numpy as np


def u(x):
    return math.exp(math.cos(x))


def d2u(x):
    return -math.cos(x)*math.exp(math.cos(x))+math.sin(x)*math.exp(math.cos(x))

n = 10

err1 = [0 for i in range(n)]
err2 = [0 for i in range(n)]

H = np.zeros(n)
refcon4 = np.zeros(n)
refcon3 = np.zeros(n)

for j in range(n):
    
    d2u_1 = 0
    d2u_2 = 0
    
    h = 1/(2**j)
    H[j] = h
    
    # Order 3 and 4 convergence
    refcon3[j] = h**3
    refcon4[j] = h**4

    # (4,0)
    a1 = [11/(12*h**2), -14/(3*h**2), 19/(2*h**2), -26/(3*h**2), 35/(12*h**2)]
    # (2,2)
    a2 = [-1/(12*h**2), 4/(3*h**2), -5/(2*h**2), 4/(3*h**2), -1/(12*h**2)]

    for i in range(-4, 1):
        d2u_1 += u(i*h)*a1[i+4]

    for i in range(-2, 3):
        d2u_2 += u(i*h)*a2[i+2]

    err1[j] = abs(d2u(0) - d2u_1)

    err2[j] = abs(d2u(0) - d2u_2)

plt.loglog(H, err1,'-o')
plt.loglog(H, err2,'-o')
plt.loglog(H,refcon4,'--')
plt.legend(["(4,0)","(2,2)","$h^2$"])
plt.xlabel("h")
plt.ylabel("$||e||$")
plt.show()
