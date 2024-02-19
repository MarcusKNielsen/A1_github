import numpy as np 


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

A = mat_A(3)  
print(mat_A(3))  

def exactfunc(x,y):
    return np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)

m = 5
x = np.linspace(0,1,m+2)
y = np.linspace(0,1,m+2)
boundary = np.zeros([m+2,m+2])

for i in x:
    for j in y:
        boundary[i,j] = 






















