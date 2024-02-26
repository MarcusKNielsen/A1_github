import numpy as np 
from scipy.sparse import spdiags, kron, eye

def exactfunc(x,y):
    return np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)

def f(x,y): 
    term1 = 2*np.sin(4*np.pi*(x+y))
    term2 = (x**2+y**2)*np.cos(4*np.pi*x*y)
    return -(4*np.pi)**2*(term1+term2)

def f_nab2(x,y):
    return 256 * np.pi**2 * (4 * np.pi**2 * np.sin(4 * np.pi * (x + y)) +
                             (-1/4 + (x**2 + y**2)**2 * np.pi**2) * np.cos(4 * np.pi * x * y) +
                             2 * np.sin(4 * np.pi * x * y) * np.pi * x * y)

def is_less_than(t1, t2, place):

    if place == "edge_l_r":
        return t1[0] == t2[0] and t1[1] < t2[1]
    if place == "edge_b_u":
        return t1[0] < t2[0] and t1[1] == t2[1]

def poisson_A5(m):
    e = np.ones(m)
    S = spdiags([e,-2*e,e], [-1, 0, 1], m, m, format="csc")
    I = eye(m, format="csc")
    A = kron(I, S) + kron(S, I)
    A=(m+1)**2*A
    return A

def poisson_A9(m):
    e = np.ones((m, 1))
    S = spdiags([-e.flatten(), -10*e.flatten(), -e.flatten()], [-1, 0, 1], m, m)
    I = spdiags([-1/2*e.flatten(), e.flatten(), -1/2*e.flatten()], [-1, 0, 1], m, m)
    A = 1/6 * (m+1)**2 * (kron(I, S) + kron(S, I))
    return A

def poisson_b5(m):

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
                b[k-1] = -exactfunc(x[i-1],y[j])/(h**2) - exactfunc(x[i],y[j-1])/(h**2)  
            elif point == (m,1): # right bottem corner
                b[k-1] = -exactfunc(x[i+1],y[j])/(h**2) - exactfunc(x[i],y[j-1])/(h**2)  
            elif (1,m) == point: # left upper corner
                b[k-1] = -exactfunc(x[i-1],y[j])/(h**2) - exactfunc(x[i],y[j+1])/(h**2)
            elif point == (m,m): # right upper corner
                b[k-1] = -exactfunc(x[i+1],y[j])/(h**2) - exactfunc(x[i],y[j+1])/(h**2) 
            elif is_less_than((1,1),point,"edge_b_u") & is_less_than(point,(m,1),"edge_b_u"): # bottem row
                b[k-1] = -exactfunc(x[i],y[j-1])/(h**2)
            elif is_less_than((1,m),point,"edge_b_u") & is_less_than(point,(m,m),"edge_b_u"): # upper row
                b[k-1] = -exactfunc(x[i],y[j+1])/(h**2) 
            elif is_less_than((1,1),point,"edge_l_r") & is_less_than(point,(1,m),"edge_l_r"):
                b[k-1] = -exactfunc(x[i-1],y[j])/(h**2) # left side 
            elif is_less_than((m,1),point,"edge_l_r") & is_less_than(point,(m,m),"edge_l_r"):
                b[k-1] = -exactfunc(x[i+1],y[j])/(h**2) # right side 

            b[k-1] += f(x[i],y[j]) # tilføjelse af f i b vektoren

    return b

def poisson_b9(m,correction):

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
                b[k-1] = -4*exactfunc(x[i-1],y[j])/(6*h**2) - 4*exactfunc(x[i],y[j-1])/(6*h**2) 
                -exactfunc(x[i-1],y[j-1])/(6*h**2) -exactfunc(x[i+1],y[j-1])/(6*h**2)
                -exactfunc(x[i-1],y[j+1])/(6*h**2)
            elif point == (m,1): # right bottem corner
                b[k-1] = -4*exactfunc(x[i+1],y[j])/(6*h**2) - 4*exactfunc(x[i],y[j-1])/(6*h**2)
                -exactfunc(x[i+1],y[j-1])/(6*h**2) -exactfunc(x[i+1],y[j+1])/(6*h**2)
                -exactfunc(x[i-1],y[j-1])/(6*h**2)  
            elif (1,m) == point: # left upper corner
                b[k-1] = -4*exactfunc(x[i-1],y[j])/(6*h**2) - 4*exactfunc(x[i],y[j+1])/(6*h**2)
                -exactfunc(x[i-1],y[j-1])/(6*h**2) -exactfunc(x[i+1],y[j+1])/(6*h**2)
                -exactfunc(x[i-1],y[j+1])/(6*h**2)
            elif point == (m,m): # right upper corner
                b[k-1] = -4*exactfunc(x[i+1],y[j])/(6*h**2) - 4*exactfunc(x[i],y[j+1])/(6*h**2) 
                -exactfunc(x[i-1],y[j+1])/(6*h**2) -exactfunc(x[i+1],y[j-1])/(6*h**2)
                -exactfunc(x[i+1],y[j+1])/(6*h**2)

            elif is_less_than((1,1),point,"edge_b_u") & is_less_than(point,(m,1),"edge_b_u"): # bottem row
                b[k-1] = -4*exactfunc(x[i],y[j-1])/(6*h**2) -exactfunc(x[i-1],y[j-1])/(6*h**2) -exactfunc(x[i+1],y[j-1])/(6*h**2)   
            elif is_less_than((1,m),point,"edge_b_u") & is_less_than(point,(m,m),"edge_b_u"): # upper row
                b[k-1] = -4*exactfunc(x[i],y[j+1])/(6*h**2) -exactfunc(x[i-1],y[j+1])/(6*h**2) -exactfunc(x[i+1],y[j+1])/(6*h**2) 
            elif is_less_than((1,1),point,"edge_l_r") & is_less_than(point,(1,m),"edge_l_r"): # left side
                b[k-1] = -4*exactfunc(x[i-1],y[j])/(6*h**2) -exactfunc(x[i-1],y[j-1])/(6*h**2) -exactfunc(x[i-1],y[j+1])/(6*h**2)
            elif is_less_than((m,1),point,"edge_l_r") & is_less_than(point,(m,m),"edge_l_r"): # right side
                b[k-1] = -4*exactfunc(x[i+1],y[j])/(6*h**2) -exactfunc(x[i+1],y[j-1])/(6*h**2) -exactfunc(x[i+1],y[j+1])/(6*h**2)
            
            if correction:
                b[k-1] += f(x[i],y[j]) + h**2/12*f_nab2(x[i],y[j]) # tilføjelse af f i b vektoren
            else:
                b[k-1] += f(x[i],y[j]) # tilføjelse af f i b vektoren
 
    return b 

