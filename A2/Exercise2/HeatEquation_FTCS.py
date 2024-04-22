import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

def u_exact(x,t,eps):
    term1 = np.exp(-eps *  1**2 * t) * np.cos( 1*x)
    term2 = np.exp(-eps *  4**2 * t) * np.cos( 4*x)
    term3 = np.exp(-eps * 16**2 * t) * np.cos(16*x)
    return term1 + term2 + term3
 
def solve_diffusion(m,eps):
    # Define the main diagonal
    main_diag = -2 * np.ones(m)

    # Define the off diagonals
    off_diag = np.ones(m - 1)

    # Create the diagonals
    diagonals = [off_diag,main_diag,off_diag]

    # Create A matrix
    h = 2/(m+1)
    A = diags(diagonals, [-1, 0, 1], format='csr') / (h**2)

    k = h**2

    if not (k*eps/h**2 <= 0.5):
        raise Exception("Invalid scheme")
     
    def forward(U,g):
        return U+eps*k*A@U + eps*k*g

    x = np.linspace(-1,1,m+2)
    U = u_exact(x[1:-1],0,eps).reshape((m, 1))
    g = np.zeros([m,1])

    T = 0.1
    t = 0

    Tarr = np.zeros(int(np.ceil(T/k))+1)
    Uarr = np.zeros([int(np.ceil(T/k))+1,m])
    Uarr[0,:] = U.ravel()
    i = 1

    #%%

    while t < T:
        
        # Update boundary
        g[0]  = u_exact(-1,t,eps) / (h**2)
        g[-1] = u_exact( 1,t,eps) / (h**2)
        
        # Update solution
        U = forward(U,g)
        Uarr[i,:] = U.ravel()
        
        # update time
        t += k
        Tarr[i] = t
        
        # update counter
        i += 1

    return Tarr, Uarr, x
    

def solution_check(Tarr, u, x, eps, plot):

    # Computing the error 
    X,T_mesh = np.meshgrid(x[1:-1],Tarr)

    uexact = u_exact(X,T_mesh,eps)

    # Computing error first
    err = uexact-u

    if plot == True:


        fig, ax = plt.subplots(1, 3, figsize=(10, 4))

        cbar_fraction = 0.05
        ax0 = ax[0].pcolormesh(X, T_mesh, uexact)
        ax[0].set_title("Exact Solution")
        #ax[0].set_aspect('equal', 'box')
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("t")
        fig.colorbar(ax0, ax=ax[0], fraction=cbar_fraction)

        ax1 = ax[1].pcolormesh(X, T_mesh, u)
        ax[1].set_title("Numerical Solution")
        #ax[1].set_aspect('equal', 'box')
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("t")
        fig.colorbar(ax1, ax=ax[1], fraction=cbar_fraction)

        ax2 = ax[2].pcolormesh(X, T_mesh, err)
        ax[2].set_title("Error")
        #ax[2].set_aspect('equal', 'box')
        ax[2].set_xlabel("x")
        ax[2].set_ylabel("t")
        fig.colorbar(ax2, ax=ax[2], fraction=cbar_fraction)

        fig.subplots_adjust(wspace=0.4)
        plt.show()

    return err

#%%

if __name__ == "__main__":

    # Define the size of your matrix
    m = 2**8  # Replace with the actual size of your matrix
    eps = 0.1

    Tarr, Uarr, x = solve_diffusion(m,eps)
    solution_check(Tarr, Uarr, x,eps, True)