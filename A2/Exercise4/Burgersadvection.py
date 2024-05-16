import numpy as np
import matplotlib.pyplot as plt

# def fdcoeffV(k,xbar,x):

#     # Function found in:
#     # https://rjleveque.github.io/amath585w2020/notebooks/html/fdstencil.html

#     x = np.array(x)  # in case a list or tuple passed in, convert to numpy array
#     n = len(x)
#     if k >=n:
#         raise ValueError('*** len(x) must be larger than k')
        
#     A = np.ones((n,n))
#     xrow = x - xbar  # displacement vector
    
#     for i in range(1,n):
#         A[i,:] = (xrow**i) / np.math.factorial(i)
      
#     condA = np.linalg.cond(A)  # condition number
#     if condA > 1e8:
#         print("Warning: condition number of Vandermonde matrix is approximately %.1e" % condA)
        
#     b = np.zeros(x.shape)
#     b[k] = 1.
    
#     c = np.linalg.solve(A,b)
    
#     return c

def fdcoeffV(k,xbar,x):
    
    if k == 1:
        
        h = x[1] - x[0]
        
        a = -1/h
        b =  1/h
        
        return np.array([a,b])
    
    if k == 2:
        
        hl = x[1] - x[0]
        hr = x[2] - x[1]
    
        a = (2/(hl*hr)) * hr/(hl+hr)
        b = (2/(hl*hr)) * hl/(hl+hr)
        c = (2/(hl*hr)) * (-1)
        
        return np.array([a,c,b])
    

def solution_check(Tarr, u, x, eps, U_exact, plot=True, exact = True):

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
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            cbar_fraction = 0.05

            ax0 = ax.pcolormesh(X, T_mesh, u)
            ax.set_title("Numerical Solution")
            ax.set_xlabel("x: space")
            ax.set_ylabel("t: time")
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

def forward_time_mix_space(t,U,eps,k,h,U_func,x=False,non_uni = False):

    N = len(U)
    Unxt = np.zeros(N)

    DU = np.zeros(N-2)
    DDU = np.zeros(N-2)

    for i in range(1,len(U)-1):

        if U[i] > 0:

            # Change stencils if non uniform grid
            if non_uni:
                
                c = fdcoeffV(1, x[i], x[i-1:i+1])

                DU[i-1] = c[0]*U[i-1]+c[1]*U[i]
            else:
                DU[i-1] = (1/h)*(U[i]-U[i-1])

        elif U[i] < 0:

            if non_uni: # Change stencils if non uniform grid
                c = fdcoeffV(1, x[i], x[i:i+2])
                DU[i-1] = c[0]*U[i]+c[1]*U[i+1] 
            else:
                DU[i-1] = (1/h)*(U[i+1]-U[i])

    if non_uni: # Change stencils if non uniform grid

        for i in range(1,len(U)-1):
            
            c = fdcoeffV(2, x[i], x[i-1:i+2])

            DDU[i-1] = c[0]*U[i-1] + c[1]*U[i] + c[2]*U[i+1]

        Unxt[1:-1] = U[1:-1] + eps*k * DDU - k*(U[1:-1]*DU)

    else:

        Unxt[1:-1] = U[1:-1] + (eps*k/h**2) *(U[0:-2]-2*U[1:-1] + U[2:]) - k*(U[1:-1]*DU)

    Unxt[0] = U_func(-1,t,eps)
    Unxt[-1] = U_func(1,t,eps)

    return Unxt

def check_stability(k,h,eps):
    return eps*k/h**2 + k/h <= 0.5

def solve_Burgers(T,m,eps,U_func,non_uni=False):
    
    x = np.linspace(-1,1,m+2)
    
    if non_uni == False:
        h = 2/(m+1)
        k = h/(2*eps/h + 2)
    else:
        x = g(x)
        h = np.min(x[1:] - x[:-1])
        k = h/(2*eps/h + 2)
    
    #print(f"Is the scheme stable? {check_stability(k,h,eps)}")
    
    U = np.zeros([int(np.ceil(T/k))+2,m+2])

    t = np.zeros(int(np.ceil(T/k))+2)
    j = 0

    U[0,:] = U_func(x,0,eps)

    while t[j] < T:

        if non_uni:
            U[j+1,:] = forward_time_mix_space(t[j],U[j,:],eps,k,h,U_func,x,non_uni=True)
        else:
            U[j+1,:] = forward_time_mix_space(t[j],U[j,:],eps,k,h,U_func)

        t[j+1] = t[j] + k

        if j % 500 == 0:
            print(f"progress:{np.round(j/np.ceil(T/k)*100,2)}%")

        j += 1

    return t[:j],U[:j],x,h

def solve_Burgers_low_memory(T,m,eps,U_func,non_uni=False):
    
    if non_uni == False:
        h = 2/(m+1)
        k = h/(2*eps/h + 2)
    
    t = 0
    
    j = 0
    
    x = np.linspace(-1,1,m+2)

    # Change grid if non uniform grid
    if non_uni:
        x = g(x)
        h = np.min(x[1:] - x[:-1])
        k = h/(2*eps/h + 2)
    

    U = U_func(x,0,eps)

    while t < T:

        if non_uni:
            U = forward_time_mix_space(t,U,eps,k,h,U_func,x,non_uni=True)
        else:
            U = forward_time_mix_space(t,U,eps,k,h,U_func)

        t += k

        if j % 1000 == 0:
            print(f"progress:{np.round(j/np.ceil(T/k)*100,2)}%")

        j += 1

    return t,U,x,h,k,j

def solve_Burgers_low_memory_a_input(T,m,eps,a,U_func,non_uni=False):
    
    if non_uni == False:
        h = 2/(m+1)
        k = h/(2*eps/h + 2)
    
    t = 0
    
    j = 0
    
    x = np.linspace(-1,1,m+2)

    # Change grid if non uniform grid
    if non_uni:
        x = g(x,a)
        h = np.min(x[1:] - x[:-1])
        k = h/(2*eps/h + 2)
    

    U = U_func(x,0,eps)

    while t < T:

        if non_uni:
            U = forward_time_mix_space(t,U,eps,k,h,U_func,x,non_uni=True)
        else:
            U = forward_time_mix_space(t,U,eps,k,h,U_func)

        t += k

        if j % 1000 == 0:
            print(f"progress:{np.round(j/np.ceil(T/k)*100,2)}%")

        j += 1

    return t,U,x,h,k,j

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
        
        j += 1
    
    return True

def g(eps,a=0.08):
    return (1-a)*eps**3+a*eps

#m = 31
#xi = np.linspace(-1,1,m)
#plt.plot(g(xi,0.01),np.zeros_like(xi),".")

if __name__ == "__main__":
    
    m = 500+1
    eps = 0.01/np.pi
    T = 1.6037/np.pi
    Tarr, Uarr, x, h = solve_Burgers(T,m,eps,U_initial)
    err = solution_check(Tarr, Uarr, x, eps,U_exact, plot=True, exact=False)
    
    
    
    #%%
    
    eps = 0.01/np.pi
    T = 1.6037/np.pi
    m = 200+1

    #t,U,x,h = solve_Burgers(T,m,eps,U_initial,non_uni = True)
    #solution_check(t, U, x, eps, U_initial, exact = False)
    #Usol = U[-1,:]
    
    t,Usol,x,h,k,j = solve_Burgers_low_memory(T,m,eps,U_initial,non_uni=True)
    
    x_idx = np.argsort(abs(x))[:3]
    x_idx = np.sort(x_idx)
    
    hl = (x[x_idx[1]] - x[x_idx[0]])
    hr = (x[x_idx[2]] - x[x_idx[1]])
    
    
    Du_left    = (Usol[x_idx[1]] - Usol[x_idx[0]])/hl
    Du_central = (Usol[x_idx[2]] - Usol[x_idx[0]])/(hr+hl)
    Du_right   = (Usol[x_idx[2]] - Usol[x_idx[1]])/hr
    
    print(Du_left, Du_central, Du_right)
    
    # x_idx = np.argsort(abs(x))[:7]
    # x_idx = np.sort(x_idx)

    # x_stencil = x[x_idx]
    
    # c = fdcoeffV2(1,x_stencil[3],x_stencil[2:5])
    # Dx_0_c = c[0]*Usol[x_idx[2]]+c[1]*Usol[x_idx[3]]+c[2]*Usol[x_idx[4]]

    # c = fdcoeffV2(1,x_stencil[3],x_stencil[3:5])
    # Dx_0_f = c[0]*Usol[x_idx[3]]+c[1]*Usol[x_idx[4]]

    # c = fdcoeffV2(1,x_stencil[3],x_stencil[2:4])
    # Dx_0_b = c[0]*Usol[x_idx[2]]+c[1]*Usol[x_idx[3]]

    # c = fdcoeffV2(1,x_stencil[3],x_stencil[1:6])
    # Dx_higher1 = c[0]*Usol[x_idx[1]]+c[1]*Usol[x_idx[2]]+c[2]*Usol[x_idx[3]]+c[3]*Usol[x_idx[4]]+c[4]*Usol[x_idx[5]]
    
    # print(Dx_0_b,Dx_0_c,Dx_0_f,Dx_higher1)

    plt.figure()
    plt.plot(x,Usol,"-o")
    plt.show()
    
    #%% Convergence 
    
    T = 0.1
    eps = 0.1
    E = []
    H = []
    
    s = np.arange(6,12)
    for s_i in s:
    
        # Compute solution and error
        m = 2**s_i - 1
        print(m)
        Tarr, Uarr, x, h = solve_Burgers(T,m,eps,U_exact,non_uni=False)
        err = solution_check(Tarr, Uarr, x, eps,U_exact, plot=False)
    
        # append global error
        E.append(np.max(np.abs(err[-1])))
        
        # append mesh size 
        h = 2/(m+1)
        H.append(h)
    
    #%%
    
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
    
    
    
    
        
    




