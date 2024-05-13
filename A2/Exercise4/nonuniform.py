import numpy as np
import matplotlib.pyplot as plt


def g(eps,a):
    return (1-a)*eps**3+a*eps

eps = np.linspace(-1,1,20)
zerovec = np.zeros(len(eps))
onesvec = np.ones(len(eps))
x = g(eps,0.1) 

plt.figure()
plt.plot(x,zerovec,'-o',label="nonuni")
plt.plot(eps,onesvec,'-o',label="uni")
plt.legend()
plt.figure()
plt.plot(x,"-o",label="non uni")
plt.plot(eps,"-o",label="uni")
plt.legend()
plt.show()




