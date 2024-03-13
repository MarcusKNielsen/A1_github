import numpy as np 
from scipy.sparse import spdiags, kron, eye,csc_matrix
from scipy.sparse.linalg import inv, spsolve

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
    e = np.ones(m)
    S = spdiags([-e, -10*e, -e], [-1, 0, 1], m, m)#, format="csc")
    I = spdiags([-1/2*e, e, -1/2*e], [-1, 0, 1], m, m)#, format="csc")
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
                b[k-1] = -4*exactfunc(x[i-1],y[j])/(6*h**2) - 4*exactfunc(x[i],y[j-1])/(6*h**2) -exactfunc(x[i-1],y[j-1])/(6*h**2) -exactfunc(x[i+1],y[j-1])/(6*h**2) -exactfunc(x[i-1],y[j+1])/(6*h**2)
            elif point == (m,1): # right bottom corner
                b[k-1] = -4*exactfunc(x[i+1],y[j])/(6*h**2) - 4*exactfunc(x[i],y[j-1])/(6*h**2) -exactfunc(x[i+1],y[j-1])/(6*h**2) -exactfunc(x[i+1],y[j+1])/(6*h**2) -exactfunc(x[i-1],y[j-1])/(6*h**2)  
            elif (1,m) == point: # left upper corner
                b[k-1] = -4*exactfunc(x[i-1],y[j])/(6*h**2) - 4*exactfunc(x[i],y[j+1])/(6*h**2)-exactfunc(x[i-1],y[j-1])/(6*h**2) -exactfunc(x[i+1],y[j+1])/(6*h**2)-exactfunc(x[i-1],y[j+1])/(6*h**2)
            elif point == (m,m): # right upper corner
                b[k-1] = -4*exactfunc(x[i+1],y[j])/(6*h**2) - 4*exactfunc(x[i],y[j+1])/(6*h**2)-exactfunc(x[i-1],y[j+1])/(6*h**2) -exactfunc(x[i+1],y[j-1])/(6*h**2)-exactfunc(x[i+1],y[j+1])/(6*h**2)

            elif is_less_than((1,1),point,"edge_b_u") & is_less_than(point,(m,1),"edge_b_u"): # bottom row
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

def Amult(U,m):

    result = np.zeros(m*m)
    h = 1/(m+1)

    for i in range(1,m+1):
        for j in range(1,m+1):

            point = (i,j)
            k = i+m*(j-1) 
            k_c = k-1 # Current k in vector U
       
            # The corners
            if (1,1) == point: # left bottom corner
                result[k-1] =  -4*U[k_c] + U[k_c+1] + U[k_c+m] 
            elif point == (m,1): # right bottem corner
                result[k-1] = -4*U[k_c] + U[k_c-1] + U[k_c+m]
            elif (1,m) == point: # left upper corner
                result[k-1] = -4*U[k_c] + U[k_c+1] + U[k_c-m]
            elif point == (m,m): # right upper corner
                result[k-1] = -4*U[k_c] + U[k_c-1] + U[k_c-m]
            elif is_less_than((1,1),point,"edge_b_u") & is_less_than(point,(m,1),"edge_b_u"): # bottem row
                result[k-1] = -4*U[k_c] + U[k_c-1] + U[k_c+1] + U[k_c+m] 
            elif is_less_than((1,m),point,"edge_b_u") & is_less_than(point,(m,m),"edge_b_u"): # upper row
                result[k-1] = -4*U[k_c] + U[k_c-1] + U[k_c+1] + U[k_c-m] 
            elif is_less_than((1,1),point,"edge_l_r") & is_less_than(point,(1,m),"edge_l_r"):  # left side 
                result[k-1] = -4*U[k_c] + U[k_c+m] + U[k_c+1] + U[k_c-m] 
            elif is_less_than((m,1),point,"edge_l_r") & is_less_than(point,(m,m),"edge_l_r"): # right side 
                result[k-1] = -4*U[k_c] + U[k_c+m] + U[k_c-1] + U[k_c-m] 
            else:
                result[k-1] = -4*U[k_c] + U[k_c+m] + U[k_c-1] + U[k_c-m] + U[k_c+1] # Rest follows the equation

    # -A**h * U
    return -result/(h**2)


def eigenvalues_5point_relax(h,p,q,omega):

    # Eigenvalue for G (Iteration matrix) eq. (3.15)
    lambda_pq = 2/(h**2)*( (np.cos(p*np.pi*h)-1) + (np.cos(q*np.pi*h)-1) ) 

    # The derived eigenvalue for the G_omega 
    gamma_pq = (1-omega)+omega*(1+h**2/4*lambda_pq) # eq. (4.90)
    
    return gamma_pq

def smooth(U,omega,m,F):

    # Step size
    h = 1/(m+1)

    # Dervied from eq. (4.88)
    Unew = U+omega*h**2/4*F

    return Unew

def matrix_figure(A):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.title("Matrix structure")
    img = plt.imshow(A)
    fig.colorbar(img)
    plt.xlabel("k: node index")
    plt.ylabel("k: node index")
    plt.show()

def generate_R(m):
    e = np.ones(m*m)
    S = spdiags([e,e,e*2,e,e], [-m,-1, 0, 1,m], m, m, format="csc")
    I = eye(m, format="csc")
    R_mat = kron(I, S) + kron(S, I)
    R_mat = R_mat/8
    R_mat = R_mat.toarray()
    R_mat = R_mat[:int((m*m-1)/2),:]
    R_mat = csc_matrix(R_mat)

    return R_mat 

def coarsen(R,res):

    # residual of coarse 
    r_coarse = R@res

    return r_coarse

def generate_P(m):
    x_range = np.arange(1, m+1)
    y_range = np.arange(1, m+1)

    # Create a meshgrid of coordinates
    X, Y = np.meshgrid(x_range, y_range)

    # Flatten the meshgrid and create an array of tuples
    array_points = np.column_stack((X.ravel(), Y.ravel()))
    points = [tuple(row) for row in array_points]

    m_f = int(m*m)
    m_c = int((m_f-1)/2)

    P = np.zeros([m_f,m_c+1])

    for idx,row in enumerate(P):

        point = points[idx]

        # Changing to 1-indexing
        idx += 1

        if idx%2 == 0: # Even numbers
            if (1,1) == point: # left bottom corner
                # Right up
                row[int((idx+m+1)/2)] = 1
                # Left up
                row[int((idx+m-1)/2)] = 1

            elif point == (m,1): # right bottem corner
                # Right up
                row[int((idx+m+1)/2)] = 1
                # Left up
                row[int((idx+m-1)/2)] = 1

            elif (1,m) == point: # left upper corner
                # Right down
                row[int((idx-m+1)/2)] = 1
                # Left down 
                row[int((idx-m-1)/2)] = 1

            elif point == (m,m): # right upper corner
                # Right down
                row[int((idx-m+1)/2)] = 1
                # Left down 
                row[int((idx-m-1)/2)] = 1
                    
            elif is_less_than((1,1),point,"edge_b_u") & is_less_than(point,(m,1),"edge_b_u"): # bottem row
                # Right up
                row[int((idx+m+1)/2)] = 1
                # Left up
                row[int((idx+m-1)/2)] = 1

            elif is_less_than((1,m),point,"edge_b_u") & is_less_than(point,(m,m),"edge_b_u"): # upper row
                # Right down
                row[int((idx-m+1)/2)] = 1
                # Left down 
                row[int((idx-m-1)/2)] = 1

            elif is_less_than((1,1),point,"edge_l_r") & is_less_than(point,(1,m),"edge_l_r"):  # left side 
                # Right up
                row[int((idx+m+1)/2)] = 1
                # Right down
                row[int((idx-m+1)/2)] = 1

            elif is_less_than((m,1),point,"edge_l_r") & is_less_than(point,(m,m),"edge_l_r"): # right side 
                # Left up
                row[int((idx+m-1)/2)] = 1
                # Left down 
                row[int((idx-m-1)/2)] = 1

            else: # Mid points, we use all of the points in the stencil
                # Right up
                row[int((idx+m+1)/2)] = 1
                # Left up
                row[int((idx+m-1)/2)] = 1
                # Right down
                row[int((idx-m+1)/2)] = 1
                # Left down 
                row[int((idx-m-1)/2)] = 1

        else: # If odd number
            if (1,1) == point: # left bottom corner
                # Up 
                row[int((idx+m)/2)] = 1
                # Right 
                row[int((idx+1)/2)] = 1

            elif point == (m,1): # right bottem corner
                # Up 
                row[int((idx+m)/2)] = 1
                #left 
                row[int((idx-1)/2)] = 1

            elif (1,m) == point: # left upper corner
                # Down
                row[int((idx-m)/2)] = 1
                # Right 
                row[int((idx+1)/2)] = 1

            elif point == (m,m): # right upper corner
                # Down
                row[int((idx-m)/2)] = 1
                #left 
                row[int((idx-1)/2)] = 1
                    
            elif is_less_than((1,1),point,"edge_b_u") & is_less_than(point,(m,1),"edge_b_u"): # bottem row
                # Up 
                row[int((idx+m)/2)] = 1
                # Right 
                row[int((idx+1)/2)] = 1
                #left 
                row[int((idx-1)/2)] = 1

            elif is_less_than((1,m),point,"edge_b_u") & is_less_than(point,(m,m),"edge_b_u"): # upper row
                # Down
                row[int((idx-m)/2)] = 1
                # Right 
                row[int((idx+1)/2)] = 1
                #left 
                row[int((idx-1)/2)] = 1

            elif is_less_than((1,1),point,"edge_l_r") & is_less_than(point,(1,m),"edge_l_r"):  # left side 
                # Up 
                row[int((idx+m)/2)] = 1
                # Down
                row[int((idx-m)/2)] = 1
                # Right 
                row[int((idx+1)/2)] = 1

            elif is_less_than((m,1),point,"edge_l_r") & is_less_than(point,(m,m),"edge_l_r"): # right side 
                # Up 
                row[int((idx+m)/2)] = 1
                # Down
                row[int((idx-m)/2)] = 1
                #left 
                row[int((idx-1)/2)] = 1

            else: # Mid points, we use all of the points in the stencil
                # Up 
                row[int((idx+m)/2)] = 1
                # Down
                row[int((idx-m)/2)] = 1
                # Right 
                row[int((idx+1)/2)] = 1
                #left 
                row[int((idx-1)/2)] = 1

    # Deleting first column due to 1-indexing
    P = P[:,1:]

    return csc_matrix(P)

def interpolate(P,ec):    

    e = P@ec
                
    return e

def VCM(A,R,P,u,f,l,m):

    omega = 2/3

    if l == 1:
        u = spsolve(A,f)
    else:
        u = smooth(u,omega,m,f)
        r_f = f+Amult(u,m)
        r_c = coarsen(R,r_f)
        e_c = np.zeros(int((m*m-1)/2))
        e_c = VCM(R@A@P,R,P,e_c,r_c,l-1,m)
        e_f = interpolate(P,e_c)
        u = u + e_f
        u = smooth(u,omega,m,f)
    return u


k=2
m = 2**k - 1
R = generate_P(m) 
print(R.shape)   
#matrix_figure(R)

l = 2
A = poisson_A5(m)
f = poisson_b5(m)
#R = generate_R(m) #Mangler
P = generate_P(m) 
u = np.zeros(m*m)

u_test = VCM(A,R,P,u,f,l,m)



