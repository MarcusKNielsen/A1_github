import numpy as np 
from scipy.sparse import spdiags, kron, eye
import matplotlib.pyplot as plt

"""
Ændringer

- Har tilføjet et 1/h**2 i A matricen. Har også tilføjet et format csc til matricen, dette skal bruges
  for at løse problemt med scipy's sparse solver.


- Jeg har tilføjet en funktion f(x,y), dette er højre siden af poissons ligning, nabla^2 u = f .
  Man kan finde f ved at regne nabla^2 u_exact analytisk, f vil så være resultatet af det.
  Dette skal incorporeres i b vektoren.
  
  
- I konstruktionen af b vektoren var der byttet om på i og j. Så vidt jeg har forstået så er j række
  indexet, så når j ændres så ændres y. og så tilsvarende for i og x.


- For a løse det linære system sparsed har jeg skiftet from numpy.linalg.solve til
  scipy's sparse solver.
  
"""


def exactfunc(x,y):
    return np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)

def f(x,y): 
    term1 = 2*np.sin(4*np.pi*(x+y))
    term2 = (x**2+y**2)*np.cos(4*np.pi*x*y)
    return -(4*np.pi)**2*(term1+term2)

def poissonA(m):
    h = 1/(m+1)
    e = np.ones(m)
    S = spdiags([e,-2*e,e], [-1, 0, 1], m, m, format="csc")
    I = eye(m, format="csc")
    A = kron(I, S) + kron(S, I)
    return A/h**2

# Check that the sparse A matrix looks correct using imshow
A_sparse = poissonA(4).todense()
fig = plt.figure()
plt.title("Sparse Laplacian Matrix Structure")
img = plt.imshow(A_sparse)
fig.colorbar(img)
plt.xlabel("k: node index")
plt.ylabel("k: node index")
plt.show()

def mat_A(m):

    h = 1/(m+1)

    A   = np.zeros([m*m,m*m])
    k   = 0

    for j in range(m*m):
        for i in range(m*m):
            
            if j == i:

                k = i
                # Computing the current row
                row = np.zeros(m*m)

                row[k] = -4

                if k > 0:
                    row[k-1] = 1

                if k >= m:
                    row[k-m] = 1

            # Inserting the current row
            A[j,:] = row
         
        # Making symmetric
        A = np.tril(A)+np.triu(A.T,1)

    return A/h**2

def is_less_than(t1, t2):
    return t1[0] <= t2[0] and t1[1] <= t2[1]

def vec_b(m):

    h = 1/(m+1)
    
    x = np.linspace(0,1,m+2)
    y = np.linspace(0,1,m+2)
    b = np.zeros(m*m)

    for j in range(1,m+1): # y
        for i in range(1,m+1): # x

            k = i+m*(j-1)

            point = (i,j)    

            # The corners
            if (1,1) == point: # left bottom corner
                b[k-1] += -exactfunc(x[i-1],y[j])/h**2 - exactfunc(x[i],y[j-1])/h**2  
            elif point == (m,1): # right bottem corner
                b[k-1] += -exactfunc(x[i+1],y[j])/h**2 - exactfunc(x[i],y[j-1])/h**2  
            elif (1,m) == point: # left upper corner
                b[k-1] += -exactfunc(x[i-1],y[j])/h**2 - exactfunc(x[i],y[j+1])/h**2 
            elif point == (m,m): # right upper corner
                b[k-1] += -exactfunc(x[i+1],y[j])/h**2 - exactfunc(x[i],y[j+1])/h**2  
            elif is_less_than((1,1),point) and is_less_than(point, (m,1)): # bottem row
                b[k-1] += -exactfunc(x[i],y[j-1])/h**2
            elif is_less_than((1,m),point) and is_less_than(point, (m,m)): # upper row
                b[k-1] += -exactfunc(x[i],y[j+1])/h**2  
            elif is_less_than((1,1),point) and is_less_than(point,(1,m)):
                b[k-1] += -exactfunc(x[i-1],y[j])/h**2 # left side 
            elif is_less_than((m,1),point) and is_less_than(point,(m,m)):
                b[k-1] += -exactfunc(x[i+1],y[j])/h**2 # right side 

            b[k-1] += f(x[i],y[j]) # tilføjelse af f i b vektoren

    return b


m = 100
A = poissonA(m)
b = vec_b(m)


#%%

from scipy.sparse.linalg import spsolve # scipy's sparse solver

x = np.linspace(0,1,m+2)
y = np.linspace(0,1,m+2)


u_solution = spsolve(A, b)
u_solution = u_solution.reshape(m, m)

X,Y = np.meshgrid(x[1:-1],y[1:-1])

u_exact = exactfunc(X,Y)

err = u_exact-u_solution

#%%

fig, ax = plt.subplots(1, 3, figsize=(14, 4))

cbar_fraction = 0.045

ax0 = ax[0].pcolormesh(X, Y, u_exact)
ax[0].set_title("Exact Solution")
ax[0].set_aspect('equal', 'box')
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
fig.colorbar(ax0, ax=ax[0], fraction=cbar_fraction)

ax1 = ax[1].pcolormesh(X, Y, u_solution)
ax[1].set_title("Numerical Solution")
ax[1].set_aspect('equal', 'box')
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
fig.colorbar(ax1, ax=ax[1], fraction=cbar_fraction)

ax2 = ax[2].pcolormesh(X, Y, err)
ax[2].set_title("Error")
ax[2].set_aspect('equal', 'box')
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
fig.colorbar(ax2, ax=ax[2], fraction=cbar_fraction)

fig.subplots_adjust(wspace=0.3)

plt.show()

#%% Error analysis

from numpy.linalg import norm

N = 6
H = np.zeros(N)
E = np.zeros(N)

for i in range(N):
    
    m = 10*2**i
    
    H[i] = 1/(m+1)
    
    x = np.linspace(0,1,m+2)
    y = np.linspace(0,1,m+2)

    A = poissonA(m)
    b = vec_b(m)

    u_solution = spsolve(A, b)
    u_solution = u_solution.reshape(m, m)

    X,Y = np.meshgrid(x[1:-1],y[1:-1])

    u_exact = exactfunc(X,Y)

    E[i] = norm(u_exact-u_solution,np.inf)
    
    
a,b = np.polyfit(np.log(H), np.log(E), 1)

plt.figure()
plt.plot(np.log(H),np.log(E),"o-")
plt.plot(np.log(H),b+a*np.log(H),color="red")
plt.xlabel(r"$\log(h)$")
plt.ylabel(r"$\log(E)$")
plt.show()







