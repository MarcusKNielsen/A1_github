
"""
This is an implementation of Newtons method to solve exercise 2 b.
The implementation is a ajusted version of the one describe in section
"2.16.1 Discretization of the nonlinear boundary value proble"
page 38 in the book (Finite Difference Methods-Leveque)
"""


import numpy as np
from numpy.linalg import solve, norm
import matplotlib.pyplot as plt

# Define uniform grid/mesh
m = 200
x = np.linspace(0,1,m+2)
h = 1/(m+1)

# Boundary conditions
alpha = -1
beta = 1.5

# Parameters
eps = 0.01


#%%

# This is the function 
def res(U):
    G = np.zeros(m)
    for j in range(1,m+1):
        
        term1 = eps * ( U[j+1] -2*U[j] + U[j-1] )/h**2
        term2 = U[j] * ( (U[j+1] - U[j-1])/(2*h) - 1 )
        
        G[j-1] = term1 + term2
    return G

def Jac(U):
    # Construct Jacobian
    J = np.zeros([m,m])
    for i in range(m):
        
        if i != 0:
            J[i,i-1] = eps/h**2 - U[i]/(2*h)
            
        J[i,i] = -2*eps/h**2 + (U[i+1] - U[i-1])/(2*h) - 1
    
        if i != m-1:
            J[i,i+1] = eps/h**2 + U[i]/(2*h)
    return J

#%%

# Initial Guess
U = np.ones(m+2)
U[0]  = alpha
U[-1] = beta

#%%

# Newtons Method
tol = 10**(-10)         # Tolerance
r = 10                  # residual
k = 0                   # Iteration counter
MaxIter = 100           # Max number of iterations

while r > tol and k < MaxIter:
    
    # Solve linear system
    G = res(U)
    J = Jac(U)
    delta = solve(J,-G)
    
    # Save old U to calculate residual
    Uold = np.copy(U)
    
    # Update U
    U[1:-1] += delta
    
    # Calculate residual
    r = norm(U-Uold,np.inf)
    
    #
    k += 1

#%%
plt.figure()
plt.plot(x,U)
plt.xlabel("x")
plt.ylabel("U")
plt.show()

"""
The plottet solution matches the one in Figure 2.7 page 47 in the book 
(Finite Difference Methods-Leveque)
"""


