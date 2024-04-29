import numpy as np
import matplotlib.pyplot as plt

m = 100

def r(p,c):
    return np.sqrt(( 1-c )**2 + c**2 + 2*( 1-c )*c*np.cos(2*np.pi*p/(m+1)))
 
p = np.linspace(1,m+1,m+1)
 
plt.figure()
plt.plot(p,r(p,0.2),label="c=0.2")
plt.plot(p,r(p,0.5),label="c=0.5")
plt.plot(p,r(p,0.7),label="c=0.7")
plt.plot(p,r(p,1.0),label="c=1.0")
plt.plot(p,r(p,1.2),label="c=1.2")
plt.plot(p,r(p,1.5),label="c=1.5")
plt.xlabel("p")
plt.ylabel("r(p,c)")
plt.legend()
plt.show()

