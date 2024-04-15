import numpy as np 
from scipy.linalg import toeplitz

n = 20
vec_row = np.zeros(n)
vec_col = np.zeros(n)
vec_row[0] = 1
vec_row[1] = -1
vec_col[0] = 1
vec_col[-1]=-1

A = 2*2/0.2*toeplitz(vec_row, vec_col)
print(A)

eigenvecs,eigenvalues = np.linalg.eig(A)
print(eigenvalues)
print(np.real(eigenvalues))
print(np.imag(eigenvalues))
import matplotlib.pyplot as plt

plt.plot(np.real(eigenvalues),np.imag(eigenvalues),".")
plt.show()





