import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

def u_exact(x,t,a):
    
    y = x-a*t
    func = np.sin(2*np.pi*y)

    return func
 
def solve_advection(m,a,k=None,T=None):

    # Number of spacial grid points
    N = m+2

    # Define the main diagonal
    main_diag = np.ones(N)

    # Define the off diagonals
    off_diag = -1*np.ones(N - 1)
    
    one_diag = -1*np.ones(1)

    # Create the diagonals
    diagonals = [off_diag,main_diag,one_diag]

    # Create A matrix
    h = 2/(N-1)
    A = -a*diags(diagonals, [-1, 0, N-1], format='csr') / h

    if not k:
        k = h/a

    if not (k*a/h <= 1):
        raise Exception("Invalid scheme")
     
    def forward(U,k):
        return k*A@U + U

    x = np.linspace(-1,1,N)
    U = u_exact(x,0,a).reshape((N, 1))

    if not T:
        T = 2
    t = 0

    M = int(np.ceil(T/k))+2
    Tarr = np.zeros(M)
    Uarr = np.zeros([M,N])
    Uarr[0,:] = U.ravel()

    i = 1
    while t < T:
        
        # Update solution
        U = forward(U,k)
        U[0] = U[-1]
        Uarr[i,:] = U.ravel()
        
        # update time
        t += k
        Tarr[i] = t
        
        # update counter
        i += 1

    Uarr = Uarr[:i,:]
    Tarr = Tarr[:i]


    return Tarr, Uarr, x
    

def solution_check(Tarr, u, x, a, plot):

    # Computing the error 
    X,T_mesh = np.meshgrid(x,Tarr)

    uexact = u_exact(X,T_mesh,a)
 
    # Computing error first
    err = u - uexact
 
    if plot == True:

        fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        
        cbar_fraction = 0.09
        ax0 = ax[0].pcolormesh(X, T_mesh, uexact)
        ax[0].set_title(r"Exact Solution: $\hat{U}$")
        ax[0].set_xlabel("x: space")
        ax[0].set_ylabel("t: time")
        fig.colorbar(ax0, ax=ax[0], fraction=cbar_fraction)

        ax1 = ax[1].pcolormesh(X, T_mesh, u)
        ax[1].set_title("Numerical Solution: $U$")
        ax[1].set_xlabel("x: space")
        ax[1].set_ylabel("t: time")
        fig.colorbar(ax1, ax=ax[1], fraction=cbar_fraction)

        ax2 = ax[2].pcolormesh(X, T_mesh, err)
        ax[2].set_title("Error: $U-\hat{U}$")
        ax[2].set_xlabel("x: space")
        ax[2].set_ylabel("t: time")
        fig.colorbar(ax2, ax=ax[2], fraction=cbar_fraction)

        fig.subplots_adjust(wspace=0.6,bottom=0.15)
        plt.show()

    return err

#%%

if __name__ == "__main__":

    # Define the size of your matrix
    m = 2**6  # Replace with the actual size of your matrix
    a = 0.5

    Tarr, Uarr, x = solve_advection(m,a)
    solution_check(Tarr, Uarr, x, a, True)
