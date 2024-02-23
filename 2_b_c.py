import numpy as np 
from scipy.sparse import spdiags, kron, eye


def exactfunc(x,y):
    return np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)

def poissonA(m):
    e = np.ones(m)
    S = spdiags([e,-2*e,e], [-1, 0, 1], m, m)
    I = eye(m)
    A = kron(I, S) + kron(S, I)
    #A = (m + 1)**2 * A SKAL VI BRUGE DETTE (DET ER FRA SLIDES??)
    return A 

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

    return A

def is_less_than(t1, t2):
    return t1[0] <= t2[0] and t1[1] <= t2[1]

def vec_b(m):

    h = 1/(m+1)
    
    x = np.linspace(0,1,m+2)
    y = np.linspace(0,1,m+2)
    b = np.zeros(m*m)

    for j in range(1,m+1): # x
        for i in range(1,m+1): # y

            k = i+m*(j-1)

            point = (j,i)    

            # The corners
            if (1,1) == point: # left bottom corner
                b[k-1] = -exactfunc(x[j-1],y[i])/h**2 - exactfunc(x[j],y[i-1])/h**2  
            elif point == (m,1): # right bottem corner
                b[k-1] = -exactfunc(x[j+1],y[i])/h**2 - exactfunc(x[j],y[i-1])/h**2  
            elif (1,m) == point: # left upper corner
                b[k-1] = -exactfunc(x[j-1],y[i])/h**2 - exactfunc(x[j],y[i+1])/h**2 
            elif point == (m,m): # right upper corner
                b[k-1] = -exactfunc(x[j+1],y[i])/h**2 - exactfunc(x[j],y[i+1])/h**2  
            elif is_less_than((1,1),point) and is_less_than(point, (m,1)): # bottem row
                b[k-1] = -exactfunc(x[j],y[i-1])/h**2
            elif is_less_than((1,m),point) and is_less_than(point, (m,m)): # upper row
                b[k-1] = -exactfunc(x[j],y[i+1])/h**2  
            elif is_less_than((1,1),point) and is_less_than(point,(1,m)):
                b[k-1] = -exactfunc(x[j-1],y[i])/h**2 # left side 
            elif is_less_than((m,1),point) and is_less_than(point,(m,m)):
                b[k-1] = -exactfunc(x[j+1],y[i])/h**2 # right side 

    return b


m = 5
A1 = poissonA(m)
A2 = mat_A(m)  
b = vec_b(m)
print(A1.toarray())
print(b)

from numpy import linalg

x = np.linspace(0,1,m*m)
y = np.linspace(0,1,m*m)

u_solution = linalg.solve(A1.toarray(), b)
u_exact = np.zeros([m,m])

for j in range(m):
    for i in range(m):
        u_exact[j,i] = exactfunc(x[j],y[i])

global_err = u_solution - u_exact

import matplotlib as plt
N = 6
H = np.zeros(N)

a,b = np.polyfit(np.log(H), np.log(global_err), 1)

plt.figure()
plt.plot(np.log(H),np.log(global_err),"o-")
plt.plot(np.log(H),b+a*np.log(H),color="red")
plt.xlabel(r"$\log(h)$")
plt.ylabel(r"$\log(E)$")
plt.show()





















