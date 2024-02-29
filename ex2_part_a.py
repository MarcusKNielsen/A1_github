
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



def res(U,h,m):
    G = np.zeros(m)
    for j in range(1,m+1):
        
        term1 = eps * ( U[j+1] -2*U[j] + U[j-1] )/h**2
        term2 = U[j] * ( (U[j+1] - U[j-1])/(2*h) - 1 )
        
        G[j-1] = term1 + term2

    return G

def Jac(U,h,m):
    # Construct Jacobian
    J = np.zeros([m,m])
    for i in range(1,m+1):
        
        if i != 1: # Lower diagonal
            J[i-1,i-2] = eps/h**2 - U[i]/(2*h)
        
        # Diagonal
        J[i-1,i-1] = -2*eps/h**2 + (U[i+1] - U[i-1])/(2*h) - 1
    
        if i != m: # Upper diagonal
            J[i-1,i] = eps/h**2 + U[i]/(2*h)

    return J



# Initial Guess
U = np.ones(m+2)
U[0]  = alpha
U[-1] = beta

Uinit = np.copy(U)



def NewtonsMethod(res,J,U,h,m,tol=10**(-12),MaxIter=100):

    r = 10                  # residual
    k = 0                   # Iteration counter
    
    while r > tol and k < MaxIter:
        
        # Solve linear system
        G = res(U,h,m)
        J = Jac(U,h,m)
        delta = solve(J,-G)
        
        # Update U
        U[1:-1] += delta
        
        # Calculate residual
        r = norm(delta,np.inf)
        
        # Update iteration counter
        k += 1
    print(f"Newtons method: m={m}, k={k}")
    return U

U = NewtonsMethod(res,Jac,U,h,m)



fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(x,Uinit)
ax[0].set_title("Initial guess")
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$U_0$")

ax[1].plot(x, U)
ax[1].set_title("Numerical Solution")
ax[1].set_xlabel("x")
ax[1].set_ylabel("U")

plt.show()

"""
The plottet solution matches the one in Figure 2.7 page 47 in the book 
(Finite Difference Methods-Leveque)
"""

#%%

"""
The new part is finding the global error this is exercise 2 part a (b).
"""

"""
First we compute a reference solution
"""

def setup_U(m):
    x = np.linspace(0,1,m+2)
    h = 1/(m+1)
    
    # Boundary conditions
    alpha = -1
    beta = 1.5
    
    # Parameters
    eps = 0.01
    
    # Initial Guess
    U = np.ones(m+2)
    U[0]  = alpha
    U[-1] = beta
    return x,h,U

# First we calculate a very fine solution
# convergens test
N = 6
m = 8193 
x,h,U = setup_U(m-2)
Uref = NewtonsMethod(res,Jac,U,h,m-2)
err = np.zeros(N)
H = np.zeros(N)
k = 1
for i in range(N,0,-1):
    m = int((m+1)/2)
    x1,h,U = setup_U(m-2)
    print(f"{np.allclose(x1, x[::2**k])}")
    U = NewtonsMethod(res,Jac,U,h,m-2)
    print("len of U", len(U), "len of U_hat", len(Uref[::2**k]))
    err[i-1] = norm(U-Uref[::2**k],np.inf)
    print(f"{np.allclose(U, Uref[::2**k],rtol=1e-02,atol=1e-02)}")
    H[i-1] = h
    k += 1 
    print(f"i={i}, m={m}, h={h}, E = {err[i-1]}")
    
#%%
a,b = np.polyfit(np.log(H), np.log(err), 1)
plt.figure()
plt.plot(np.log(H),np.log(err),"o-",label="Estimated Global Error")
plt.plot(np.log(H),b+2+a*np.log(H),label=r"$O(h^2)$")
plt.xlabel(r"$\log(h)$")
plt.ylabel(r"$\log(\Vert U_g - U_f \Vert)$")
plt.title("loglog plot of the Estimated Global Error vs. Step size")
plt.legend()
plt.show()








