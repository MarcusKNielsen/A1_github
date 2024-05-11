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
            fig, axes = plt.subplots(1, 3, figsize=(10, 4))

            #cbar_fraction = 0.05
            ax = axes[0]
            cc = ax.pcolormesh(X, T_mesh, uexact)
            ax.set_title(r"Exact Solution: $\hat{U}$")
            ax.set_xlabel('x: space')
            ax.set_ylabel('t: time')
            fig.colorbar(cc, ax=ax)
            
            ax = axes[1]
            cc = ax.pcolormesh(X, T_mesh, u)
            ax.set_title(r"Numerical Solution: $U$")
            ax.set_xlabel('x: space')
            ax.set_ylabel('t: time')
            fig.colorbar(cc, ax=ax)

            ax = axes[2]
            cc = ax.pcolormesh(X, T_mesh, err)
            ax.set_title(r"Error: $U - \hat{U}$")
            ax.set_xlabel('x: space')
            ax.set_ylabel('t: time')
            fig.colorbar(cc, ax=ax)

        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            cbar_fraction = 0.05

            ax0 = ax.pcolormesh(X, T_mesh, u)
            ax.set_title("Numerical Solution")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.colorbar(ax0, ax=ax, fraction=cbar_fraction)

        fig.subplots_adjust(wspace=0.5,bottom=0.2)
        plt.show()

    return err

def U_exact(x,t,eps):
    return 1-np.tanh((x+0.5-t)/(2*eps))

def U_dx_exact(x,t,eps):
    return (-1/(2*eps)) * (1-np.tanh((x+0.5-t)/(2*eps)))**2

def U_initial(x,t,eps):
    return -np.sin(np.pi*x)

def forward_time_mix_space(t,U,eps,k,h,U_func):

    N = len(U)
    Unxt = np.zeros(N)

    DU = np.zeros(N-2)

    for i in range(1,len(U)-1):

        if U[i] > 0:
            DU[i-1] = (1/h)*(U[i]-U[i-1])
        elif U[i] < 0:
            DU[i-1] = (1/h)*(U[i+1]-U[i])


    Unxt[1:-1] = U[1:-1] + (eps*k/h**2) *(U[0:-2]-2*U[1:-1] + U[2:]) - k*(U[1:-1]*DU)

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

def check_stability(k,h,eps):
    return eps*k/h**2 + k/h <= 0.5

def solve_Burgers(T,m,eps,U_func):
    
    h = 2/(m+1)
    #k = 0.0125
    k = h**2
    
    print(f"Is the scheme stable? {check_stability(k,h,eps)}")
    
    U = np.zeros([int(np.ceil(T/k))+2,m+2])

    t = np.zeros(int(np.ceil(T/k))+2)
    j = 0
    
    x = np.linspace(-1,1,m+2)
    U[0,:] = U_func(x,0,eps)

    while t[j] < T:

        U[j+1,:] = forward_time_mix_space(t[j],U[j,:],eps,k,h,U_func)

        t[j+1] = t[j] + k
        j += 1

    return t[:j],U[:j],x,h

def solve_Burgers_stability_test(T,m,h,k,eps,U_func,U_dx_func,tol):
    
    U = np.zeros([int(np.ceil(T/k))+2,m+2])

    t = np.zeros(int(np.ceil(T/k))+2)
    j = 0
    
    x = np.linspace(-1,1,m+2)
    U[0,:] = U_func(x,0,eps)
    
    
    while t[j] < T:

        U[j+1,:] = forward_time_mix_space(t[j],U[j,:],eps,k,h,U_func)

        t[j+1] = t[j] + k
        

        if np.max(U[j+1,:]) > np.max(U_func(x,t[j+1],eps)) + tol:
            return False
        elif np.min(U[j+1,:]) < np.min(U_func(x,t[j+1],eps)) - tol:
            return False
        
        #err = np.abs(U[j+1,:] - U_func(x,t[j+1],eps))
        #i = np.argmax(err)
        #max_err = err[i]
        
        #if max_err/np.abs(U_dx_func(x[i],t[j+1],eps)) > tol:
        #    return False
        
        j += 1
    
    return True

if __name__ == "__main__":
    
    m = int(np.ceil(2/0.06 - 2))
    eps = 0.1
    Tarr, Uarr, x, h = solve_Burgers(1,m,eps,U_exact)
    err = solution_check(Tarr, Uarr, x, eps,U_exact, plot=True)
    
    #%%
    
    eps = 0.01/np.pi
    T = 1.6037/np.pi
    m = 1500+1
    t,U,x,h = solve_Burgers(T,m,eps,U_initial)
    
    #solution_check(t, U, x, eps, U_initial, exact = False)
    x_zero_index = np.argsort(abs(x))[:3]
    x_zero_index = np.sort(x_zero_index)
    Dx_0_c = (U[-1,x_zero_index[2]]-U[-1,x_zero_index[0]])/(2*h)
    Dx_0_f = (-U[-1,x_zero_index[1]]+U[-1,x_zero_index[2]])/h
    Dx_0_b = (U[-1,x_zero_index[1]]-U[-1,x_zero_index[0]])/h
    
    idx = np.sort(np.argsort(abs(x))[:5])
    Usol = U[-1,:]
    Dx_higher_stencil = (Usol[idx[0]]-8*Usol[idx[1]]+8*Usol[idx[3]]-Usol[idx[4]])/(12*h)
    
    idx = np.sort(np.argsort(abs(x))[:7])
    Usol = U[-1,:]
    Dx_higher_stencil2 = (-Usol[idx[0]]+9*Usol[idx[1]]-45*Usol[idx[2]]+45*Usol[idx[4]]-9*Usol[idx[5]]+Usol[idx[6]])/(60*h)
    
    print(Dx_0_b,Dx_0_c,Dx_0_f,Dx_higher_stencil,Dx_higher_stencil2)
    
    plt.figure()
    plt.plot(x,U[-1],"-o")
    plt.show()
    
    Debug = True
    
    
    
    
    #%% Convergence 
    
    T = 1.6037/np.pi
    eps = 0.1
    E = []
    H = []
    
    s = np.arange(6,10)
    for s_i in s:
    
        # Compute solution and error
        m = 2**s_i - 1
        Tarr, Uarr, x, h = solve_Burgers(T,m,eps,U_exact)
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
    
    
    
    
        
    




