import numpy as np 

def exactfunc(x,y):
    return np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)

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

def vec_b(m):

    h = 1/(m+1)
    
    x = np.linspace(0,1,m+2)
    y = np.linspace(0,1,m+2)
    b = np.zeros(m*m)

    for j in range(m):
        for i in range(m):

            k = i + j*m

            # The corners
            if k == 0: # left bottom corner
                b[k] = -exactfunc(x[j],y[i+1])/h**2 - exactfunc(x[j+1],y[i])/h**2  
            elif k == m-1: # right bottem corner
                b[k] = -exactfunc(x[j+2],y[i+1])/h**2 - exactfunc(x[j+1],y[i])/h**2  
            elif k == m**2-m: # left upper corner
                b[k] = -exactfunc(x[j],y[i+1])/h**2 - exactfunc(x[j+1],y[i+2])/h**2 
            elif k == m**2-1: # right upper corner
                b[k] = -exactfunc(x[j+2],y[i+2])/h**2 - exactfunc(x[j+1],y[i+2])/h**2  
            elif k > 0 and k < (m-1): # bottem row
                b[k] = -exactfunc(x[j+1],y[i])/h**2
            elif k > m**2-m and k < m**2-1: # upper row
                b[k] = -exactfunc(x[j+1],y[i+2])/h**2  

            # These are checked after the first if loop
            # left column
            if k%m == 0 and b[k] == 0:
                b[k] = -exactfunc(x[j-1],y[i])/h**2 
            # right column
            if (k+1)%m and b[k] == 0: 
                b[k] = -exactfunc(x[j+1],y[i])/h**2

    return b

m = 5
A = mat_A(m)  
print(A) 
b = vec_b(m)
print(b)
























