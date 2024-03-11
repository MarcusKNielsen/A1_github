import numpy as np
import matplotlib.pyplot as plt

def u(x):
    return np.exp(np.cos(x))

def u_approx(h):
    return (-1/8) * u(-3*h/2) + (6/8) * u(-h/2) + (3/8) * u(h/2)

h_list = np.logspace(-4,-1,15)

e_list = u_approx(h_list) - u(0)

plt.figure()
plt.plot(np.log(h_list),np.log(np.abs(e_list)),"-o")
plt.plot(np.log(h_list), 4 * np.log(h_list) + np.log(3*4*np.exp(1)/128), label = r"$\log(|\tau|)$")
plt.xlabel(r"$\log(h)$")
plt.ylabel(r"$\log(|D^0u(0) - u(0)|)$")
plt.title("Convergence Test of Interpolating Stencil")
plt.legend()
plt.show()