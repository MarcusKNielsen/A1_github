import numpy as np
import matplotlib.pyplot as plt

def solution_check(Tarr, u, x, eps, plot=True):

    # Computing the error 
    X,T_mesh = np.meshgrid(x,Tarr)

    uexact = U_exact(X,T_mesh,eps)

    # Computing error first
    err = uexact-u

    if plot == True:

        fig, ax = plt.subplots(1, 3, figsize=(10, 4))

        cbar_fraction = 0.05
        ax0 = ax[0].pcolormesh(X, T_mesh, uexact)
        ax[0].set_title("Exact Solution")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("t")
        fig.colorbar(ax0, ax=ax[0], fraction=cbar_fraction)

        ax1 = ax[1].pcolormesh(X, T_mesh, u)
        ax[1].set_title("Numerical Solution")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("t")
        fig.colorbar(ax1, ax=ax[1], fraction=cbar_fraction)

        ax2 = ax[2].pcolormesh(X, T_mesh, np.abs(err))
        ax[2].set_title("Error")
        ax[2].set_xlabel("x")
        ax[2].set_ylabel("t")
        fig.colorbar(ax2, ax=ax[2], fraction=cbar_fraction)

        fig.subplots_adjust(wspace=0.4)
        plt.show()

    return err

def U_exact(x,t,eps):
    
    return -np.tanh((x+0.5-t)/(2*eps))+1

def forward(t,U,eps,k,h):

    N = len(U)
    Unxt = np.zeros(N)

    Unxt[1:-1] = U[1:-1] + (eps*k/h**2) *(U[0:-2]-2*U[1:-1] + U[2:]) - (k/h)*(U[1:-1]*(U[1:-1]-U[0:-2]))

    Unxt[0] = U_exact(-1,t,eps)
    Unxt[-1] = U_exact(1,t,eps)

    return Unxt

eps = 0.1
m = 4
h = 2/(m+1)
k = h**2
T = 1.6037/np.pi
U = np.zeros([int(np.ceil(T/k))+1,m+2])

t = np.zeros(int(np.ceil(T/k))+1)
j = 0

x = np.linspace(-1,1,m+2)
U[0,:] = U_exact(x,0,eps)

while t[j] < T:

    U[j+1,:] = forward(t[j],U[j,:],eps,k,h)

    t[j+1] = t[j] + k
    j += 1


solution_check(t, U, x, eps)

debug = True







