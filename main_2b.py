import numpy as np 
from scipy.sparse import spdiags, kron, eye
import matplotlib.pyplot as plt
from functions import*
from scipy.sparse.linalg import spsolve # scipy's sparse solver

# Check that the sparse A matrix looks correct using imshow
A_sparse = poisson_A5(4).todense()
fig = plt.figure()
plt.title("Sparse Laplacian Matrix Structure")
img = plt.imshow(A_sparse)
fig.colorbar(img)
plt.xlabel("k: node index")
plt.ylabel("k: node index")
#plt.show()

# Check that the sparse A matrix looks correct using imshow
A_sparse = poisson_A9(4).todense()
fig = plt.figure()
plt.title("Sparse Laplacian Matrix Structure")
img = plt.imshow(A_sparse)
fig.colorbar(img)
plt.xlabel("k: node index")
plt.ylabel("k: node index")
#plt.show()

#%% Solve for 5 and 9 point stencil 
m = 100
A5 = poisson_A5(m)
b5 = poisson_b5(m)
A9 = poisson_A9(m)
b9 = poisson_b9(m,correction=True)

# The grid
x = np.linspace(0,1,m+2)
y = np.linspace(0,1,m+2)
X,Y = np.meshgrid(x[1:-1],y[1:-1])
u_exact = exactfunc(X,Y)

u_solution5 = spsolve(A5, b5)
u_solution5 = u_solution5.reshape(m, m)
u_solution9 = spsolve(A9, b9)
u_solution9 = u_solution9.reshape(m, m)

err5 = u_exact-u_solution5
err9 = u_exact-u_solution9

#%% Plotting solution 

fig, ax = plt.subplots(1, 3, figsize=(14, 4))

cbar_fraction = 0.045

ax0 = ax[0].pcolormesh(X, Y, u_exact)
ax[0].set_title("Exact Solution")
ax[0].set_aspect('equal', 'box')
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
fig.colorbar(ax0, ax=ax[0], fraction=cbar_fraction)

ax1 = ax[1].pcolormesh(X, Y, u_solution5)
ax[1].set_title("Numerical Solution 5 point")
ax[1].set_aspect('equal', 'box')
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
fig.colorbar(ax1, ax=ax[1], fraction=cbar_fraction)

ax2 = ax[2].pcolormesh(X, Y, err5)
ax[2].set_title("Error")
ax[2].set_aspect('equal', 'box')
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
fig.colorbar(ax2, ax=ax[2], fraction=cbar_fraction)

fig.subplots_adjust(wspace=0.3)

#plt.show()


fig, ax = plt.subplots(1, 3, figsize=(14, 4))

cbar_fraction = 0.045

ax0 = ax[0].pcolormesh(X, Y, u_exact)
ax[0].set_title("Exact Solution")
ax[0].set_aspect('equal', 'box')
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
fig.colorbar(ax0, ax=ax[0], fraction=cbar_fraction)

ax1 = ax[1].pcolormesh(X, Y, u_solution9)
ax[1].set_title("Numerical Solution 9 point")
ax[1].set_aspect('equal', 'box')
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
fig.colorbar(ax1, ax=ax[1], fraction=cbar_fraction)

ax2 = ax[2].pcolormesh(X, Y, err9)
ax[2].set_title("Error")
ax[2].set_aspect('equal', 'box')
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
fig.colorbar(ax2, ax=ax[2], fraction=cbar_fraction)

fig.subplots_adjust(wspace=0.3)

#plt.show()

#%% Error analysis

N = 6
H = np.zeros(N)
E5_inf = np.zeros(N)
E9_inf = np.zeros(N)

for i in range(N):
    
    m = 10*2**i
    
    H[i] = 1/(m+1)
    
    x = np.linspace(0,1,m+2)
    y = np.linspace(0,1,m+2)
    X,Y = np.meshgrid(x[1:-1],y[1:-1])
    u_exact = exactfunc(X,Y)

    A5 = poisson_A5(m)
    b5 = poisson_b5(m)
    A9 = poisson_A9(m)
    b9 = poisson_b9(m,correction=True)

    u_solution5 = spsolve(A5, b5)
    u_solution9 = spsolve(A9, b9)
    u_solution5 = u_solution5.reshape(m, m)
    u_solution9 = u_solution9.reshape(m, m)

    # the error and norm of the error 
    err5 = np.abs(u_solution5-u_exact)
    err9 = np.abs(u_solution9-u_exact)
    E5_inf[i] = np.max(err5)
    E9_inf[i] = np.max(err9)
    
a5,b5 = np.polyfit(np.log(H), np.log(E5_inf), 1)
a9,b9 = np.polyfit(np.log(H), np.log(E9_inf), 1)
print("a5=",a5,"a9=",a9)
plt.figure()
#plt.plot(np.log(H),np.log(E5_inf),"o-",color="green")
plt.plot(np.log(H),np.log(E9_inf),"o-",color="green",label="Infinity norm of the global error")
#plt.plot(np.log(H),b5+a5*np.log(H),color="red")
plt.plot(np.log(H),b9+a9*np.log(H),color="red",label="Helper line of order 4")
plt.xlabel(r"$\log(h)$")
plt.ylabel(r"$\log(E)$")
plt.legend()
plt.show()

