import numpy as np
from scipy.sparse import diags

eps = 0.1

def u_exact(x,t):
    term1 = np.exp(-eps *  1**2 * t) * np.cos( 1*x)
    term2 = np.exp(-eps *  4**2 * t) * np.cos( 4*x)
    term3 = np.exp(-eps * 16**2 * t) * np.cos(16*x)
    return term1 + term2 + term3


def solve(m):
    # Define the main diagonal
    main_diag = -2 * np.ones(m)
    
    # Define the off diagonals
    off_diag = np.ones(m - 1)
    
    # Create the diagonals
    diagonals = [off_diag,main_diag, off_diag]
    
    # Create A matrix
    h = 1/(m+1)
    A = diags(diagonals, [-1, 0, 1], format='csr') / (h**2)
    
    
    k = h**2
    
    valid_scheme = (k*eps/h**2 <= 0.5)
    print(valid_scheme)
    
    def forward(U,g):
        I = np.eye(m)
        return (I+eps*k*A) @ U + eps*k*g
    
    x = np.linspace(-1,1,m+2)
    U = u_exact(x[1:-1],0).reshape((m, 1))
    g = np.zeros([m,1])
    
    T = 0.1
    t = 0
    
    Tarr = np.zeros(int(T/k)+2)
    Uarr = np.zeros([int(T/k)+2,m])
    Uarr[0,:] = U.ravel()
    i = 1
    
    
    while t < T:
        
        # Update boundary
        g[0] = u_exact(-1,t)/ (h**2)
        g[-1] = u_exact(1,t)/ (h**2)
        
        # Update solution
        U = forward(U,g)
        Uarr[i,:] = U.ravel()
        
        # update time
        t += k
        Tarr[i] = t
    
        # update counter
        i += 1
        
        
    X,T_mesh = np.meshgrid(x[1:-1],Tarr)
    
    uexact = u_exact(X,T_mesh)
    
    err = np.max(np.abs(uexact-Uarr))

    return err


E = []
s = np.arange(2,10)
H = []
for i in range(len(s)):
    m = 2**s[i] - 1
    h = 1/(m+1)
    E.append(solve(m))
    H.append(h)


import matplotlib.pyplot as plt
plt.plot(np.log(H),np.log(E),"o-")






    