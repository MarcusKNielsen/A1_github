import numpy as np
import matplotlib.pyplot as plt

def solution_check(Tarr, u, x, eps, U_exact, plot=True,exact = True):

    # Computing the error 
    X,T_mesh = np.meshgrid(x,Tarr)

    uexact = U_exact(X,T_mesh,eps)

    # Computing error first
    err = uexact-u

    if plot == True:

        if exact:
            fig, ax = plt.subplots(1, 3, figsize=(10, 4))

            cbar_fraction = 0.05
            ax1 = ax[1].pcolormesh(X, T_mesh, uexact)
            ax[1].set_title("Exact Solution")
            ax[1].set_xlabel("x")
            ax[1].set_ylabel("t")
            fig.colorbar(ax1, ax=ax[1], fraction=cbar_fraction)

            ax2 = ax[2].pcolormesh(X, T_mesh, np.abs(err))
            ax[2].set_title("Error")
            ax[2].set_xlabel("x")
            ax[2].set_ylabel("t")
            fig.colorbar(ax2, ax=ax[2], fraction=cbar_fraction)

            ax0 = ax[0].pcolormesh(X, T_mesh, u)
            ax[0].set_title("Numerical Solution")
            ax[0].set_xlabel("x")
            ax[0].set_ylabel("t")
            fig.colorbar(ax0, ax=ax[0], fraction=cbar_fraction)

        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            cbar_fraction = 0.05

            ax0 = ax.pcolormesh(X, T_mesh, u)
            ax.set_title("Numerical Solution")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.colorbar(ax0, ax=ax, fraction=cbar_fraction)

        fig.subplots_adjust(wspace=0.4)
        plt.show()

    return err

def U_exact(x,t,eps):
    
    return -np.tanh((x+0.5-t)/(2*eps))+1

def U_initial(x,t,eps):
    
    return -np.sin(np.pi*x)

def forward(t,U,eps,k,h,U_func):

    N = len(U)
    Unxt = np.zeros(N)

    Unxt[1:-1] = U[1:-1] + (eps*k/h**2) *(U[0:-2]-2*U[1:-1] + U[2:]) - (k/h)*(U[1:-1]*(U[1:-1]-U[0:-2]))

    Unxt[0] = U_func(-1,t,eps)
    Unxt[-1] = U_func(1,t,eps)

    return Unxt

def forward_higher_order(t,U,eps,k,h,U_func):

    N = len(U)
    Unxt = np.zeros(N)

    Unxt[1:-1] = U[1:-1] + (eps*k/h**2) *(U[0:-2]-2*U[1:-1] + U[2:]) - (k/h)*(U[1:-1]*(U[1:-1]-U[0:-2]))

    Unxt[0] = U_func(-1,t,eps)
    Unxt[-1] = U_func(1,t,eps)

    return Unxt

def solve_Burgers(T,m,eps,U_func):

    h = 2/(m+1)
    k = h**2
    U = np.zeros([int(np.ceil(T/k))+1,m+2])

    t = np.zeros(int(np.ceil(T/k))+1)
    j = 0

    x = np.linspace(-1,1,m+2)
    U[0,:] = U_func(x,0,eps)

    while t[j] < T:

        U[j+1,:] = forward(t[j],U[j,:],eps,k,h,U_func)

        t[j+1] = t[j] + k
        j += 1

    return t,U,x 


eps = 0.01/np.pi
T = 1.6037/np.pi
m = 100
t,U,x = solve_Burgers(T,m,eps,U_initial)
solution_check(t, U, x, eps, U_initial, exact = False)
closest_zero_index = np.argsort(abs(x))[:2]


#%% Convergence 

T = 1.6037/np.pi
eps = 0.1
E = []
H = []

s = np.arange(5,11)
for s_i in s:

    # Compute solution and error
    m = 2**s_i - 1
    Tarr, Uarr, x = solve_Burgers(T,m,eps,U_exact)
    err = solution_check(Tarr, Uarr, x, eps,U_exact, plot=False)

    # append global error
    E.append(np.max(np.abs(err[-1])))
    
    # append mesh size´
    h = 2/(m+1)
    H.append(h)

ms = 14

a,b = np.polyfit(np.log(H),np.log(E),1)
print(a)
plt.figure()
plt.plot(np.log(H),np.log(E),"bo-",label="Empirical")
plt.plot(np.log(H),1*np.log(H)+3,"r-",label=r"$O(h)$")
plt.xlabel(r"$\log(h)$",fontsize=ms)
plt.ylabel(r"$\log\left(\Vert E^N \Vert_\infty \right)$",fontsize=ms)
plt.title(r"Convergence Test with $k=h^2$",fontsize=ms+1)
plt.legend(fontsize=ms-1)
plt.subplots_adjust(left=0.15,bottom=0.15)
plt.show()




    





