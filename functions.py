import numpy as np 
from scipy.sparse import spdiags, kron, eye,csc_matrix
from scipy.sparse.linalg import inv, spsolve
import matplotlib.pyplot as plt

def exactfunc(x,y):
    return np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)

def f_func(x,y): 
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

            b[k-1] += f_func(x[i],y[j]) # tilføjelse af f i b vektoren

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
                b[k-1] += f_func(x[i],y[j]) + h**2/12*f_nab2(x[i],y[j]) # tilføjelse af f i b vektoren
            else:
                b[k-1] += f_func(x[i],y[j]) # tilføjelse af f i b vektoren
 
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
    return result/(h**2)

def matrix_figure(A,i):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.title(f"Matrix structure{i}")
    img = plt.imshow(A)
    fig.colorbar(img)
    plt.xlabel("k: node index")
    plt.ylabel("k: node index")
    plt.show()

def plot_multigrid(n, idx_from_zero=True):

    # Define marker size
    marker_size_dot = 5
    marker_size_cross = 7

    # Define x and y ranges
    x = np.linspace(0, 1, n+2)  # Add 2 for boundary points
    y = np.linspace(0, 1, n+2)

    # Create meshgrid
    X, Y = np.meshgrid(x, y)

    # Plotting the grid
    plt.figure(figsize=(6,6))

    # Plotting interior points
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            if (i * (n+2) + j) % 2 == 0:  # Even interior points
                plt.plot(X[i, j], Y[i, j], 'bo', markersize=marker_size_dot)  # Plot as dot
            else:  # Odd interior points
                plt.plot(X[i, j], Y[i, j], 'bx', markersize=marker_size_cross)  # Plot as cross

    # Plotting boundary points
    plt.plot(X[:, 0], Y[:, 0], 'ro', markersize=marker_size_dot)  # left boundary
    plt.plot(X[:, -1], Y[:, -1], 'ro', markersize=marker_size_dot)  # right boundary
    plt.plot(X[0, :], Y[0, :], 'ro', markersize=marker_size_dot)  # bottom boundary
    plt.plot(X[-1, :], Y[-1, :], 'ro', markersize=marker_size_dot)  # top boundary

    # Adding labels for interior points
    if idx_from_zero:
        label1 = 0
        label2 = 0
    else:
        label1 = 1
        label2 = 1
        
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            plt.text(X[i, j] + 0.03, Y[i, j] - 0.02, str(label1), fontsize=10)
            label1 += 1
            if (i * (n+2) + j) % 2 != 0:  # If it's a cross point
                plt.text(X[i, j] + 0.02, Y[i, j] - 0.05, f"({str(label2)})", fontsize=10)
                label2 += 1
            
            

    plt.title('Uniform Grid with Boundary Points and Labels')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
    plt.show()

def plot_u_vec(u):
    m = int(np.sqrt(len(u)))
    u = u.reshape(m, m)
    x = np.linspace(0,1,m+2)
    y = np.linspace(0,1,m+2)
    
    X,Y = np.meshgrid(x[1:-1],y[1:-1])
    
    u_exact = exactfunc(X,Y)
    
    err = u_exact-u
    
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    
    cbar_fraction = 0.045
    
    ax0 = ax[0].pcolormesh(X, Y, u_exact)
    ax[0].set_title("Exact Solution")
    ax[0].set_aspect('equal', 'box')
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    fig.colorbar(ax0, ax=ax[0], fraction=cbar_fraction)
    
    ax1 = ax[1].pcolormesh(X, Y, u)
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

def eigenvalues_5point_relax(h,p,q,omega):

    # Eigenvalue for G (Iteration matrix) eq. (3.15)
    lambda_pq = 2/(h**2)*( (np.cos(p*np.pi*h)-1) + (np.cos(q*np.pi*h)-1) ) 

    # The derived eigenvalue for the G_omega 
    gamma_pq = (1-omega)+omega*(1+h**2/4*lambda_pq) # eq. (4.90)
    
    return gamma_pq


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
        
        # The current point
        point = points[idx]

        # Changing to 1-indexing
        idx += 1

        if idx%2 == 0: # Even number (coarse grid values)

            row[int(idx/2)] = 1  # Passes on the value we know already

        else: # Odd number (fine grid values)      
            if (1,1) == point: # left bottom corner
                # Up 
                row[int((idx+m)/2)] = 1/2
                # Right 
                row[int((idx+1)/2)] = 1/2

            elif point == (m,1): # right bottem corner
                # Up 
                row[int((idx+m)/2)] = 1/2
                #left 
                row[int((idx-1)/2)] = 1/2

            elif (1,m) == point: # left upper corner
                # Down
                row[int((idx-m)/2)] = 1/2
                # Right 
                row[int((idx+1)/2)] = 1/2

            elif point == (m,m): # right upper corner
                # Down
                row[int((idx-m)/2)] = 1/2
                #left 
                row[int((idx-1)/2)] = 1/2
                    
            elif is_less_than((1,1),point,"edge_b_u") & is_less_than(point,(m,1),"edge_b_u"): # bottem row
                # Up 
                row[int((idx+m)/2)] = 1/3
                # Right 
                row[int((idx+1)/2)] = 1/3
                #left 
                row[int((idx-1)/2)] = 1/3

            elif is_less_than((1,m),point,"edge_b_u") & is_less_than(point,(m,m),"edge_b_u"): # upper row
                # Down
                row[int((idx-m)/2)] = 1/3
                # Right 
                row[int((idx+1)/2)] = 1/3
                #left 
                row[int((idx-1)/2)] = 1/3

            elif is_less_than((1,1),point,"edge_l_r") & is_less_than(point,(1,m),"edge_l_r"):  # left side 
                # Up 
                row[int((idx+m)/2)] = 1/3
                # Down
                row[int((idx-m)/2)] = 1/3
                # Right 
                row[int((idx+1)/2)] = 1/3

            elif is_less_than((m,1),point,"edge_l_r") & is_less_than(point,(m,m),"edge_l_r"): # right side 
                # Up 
                row[int((idx+m)/2)] = 1/3
                # Down
                row[int((idx-m)/2)] = 1/3
                #left 
                row[int((idx-1)/2)] = 1/3

            else: # Mid points, we use all of the points in the stencil
                # Up 
                row[int((idx+m)/2)] = 1/4
                # Down
                row[int((idx-m)/2)] = 1/4
                # Right 
                row[int((idx+1)/2)] = 1/4
                #left 
                row[int((idx-1)/2)] = 1/4

    # Deleting first column due to 1-indexing
    P = P[:,1:]

    return csc_matrix(P)

def generate_R(m):

    x_range = np.arange(1, m+1)
    y_range = np.arange(1, m+1)

    # Create a meshgrid of coordinates
    X, Y = np.meshgrid(x_range, y_range)

    # Flatten the meshgrid and create an array of tuples
    array_points = np.column_stack((X.ravel(), Y.ravel()))
    points = [tuple(row) for row in array_points]

    m_f = int(m*m)
    m_c = int((m_f-1)/2)

    R = np.zeros([m_c,m_f+1])

    for idx,row in enumerate(R):

        # The current point
        point = points[idx*2+1]

        # Changing to 1-indexing
        idx += 1
                
        if is_less_than((1,1),point,"edge_b_u") & is_less_than(point,(m,1),"edge_b_u"): # bottem row
            # Up 
            row[int(idx*2+m)] = 1/6
            # Right 
            row[int(idx*2+1)] = 1/6
            #left 
            row[int(idx*2-1)] = 1/6
            # Mid point
            row[int(idx*2)] = 1/2  

        elif is_less_than((1,m),point,"edge_b_u") & is_less_than(point,(m,m),"edge_b_u"): # upper row
            # Down
            row[int(idx*2-m)] = 1/6
            # Right 
            row[int(idx*2+1)] = 1/6
            #left 
            row[int(idx*2-1)] = 1/6
            # Mid point
            row[int(idx*2)] = 1/2  

        elif is_less_than((1,1),point,"edge_l_r") & is_less_than(point,(1,m),"edge_l_r"):  # left side 
            # Up 
            row[int(idx*2+m)] = 1/6
            # Down
            row[int(idx*2-m)] = 1/6
            # Right 
            row[int(idx*2+1)] = 1/6
            # Mid point
            row[int(idx*2)] = 1/2 

        elif is_less_than((m,1),point,"edge_l_r") & is_less_than(point,(m,m),"edge_l_r"): # right side 
            # Up 
            row[int(idx*2+m)] = 1/6
            # Down
            row[int(idx*2-m)] = 1/6
            #left 
            row[int(idx*2-1)] = 1/6
            # Mid point
            row[int(idx*2)] = 1/2 

        else: # Mid points, we use all of the points in the stencil
            # Up 
            row[int(idx*2+m)] = 1/8
            # Down
            row[int(idx*2-m)] = 1/8
            # Right 
            row[int(idx*2+1)] = 1/8
            #left 
            row[int(idx*2-1)] = 1/8
            # Mid point
            row[int(idx*2)] = 1/2 

    # Deleting first column due to 1-indexing
    R = R[:,1:]

    return csc_matrix(R)

def compute_err(u):
    m = int(np.sqrt(len(u)))
    u = u.reshape(m, m)
    x = np.linspace(0,1,m+2)
    y = np.linspace(0,1,m+2)
    
    X,Y = np.meshgrid(x[1:-1],y[1:-1])
    
    u_exact = exactfunc(X,Y)
    
    err = u_exact-u
    return err

def interpolate(P,ec):    

    e = P@ec
                
    return e

def coarsen(R,res):

    # residual of coarse 
    r_coarse = R@res

    return r_coarse

def smooth(U,omega,m,F):

    # Step size
    h = 1/(m+1)

    # Dervied from eq. (4.88)
    Unew = U+(omega*h**2/4)*Amult(U, m) - omega*(h**2/4)*F

    return Unew 

def VCM(A,R,P,u,f,l,m):
    
    omega = 2/3

    if l == 1:
        u = spsolve(A,f)
    else: 
        for _ in range(20):
            u = smooth(u,omega,m,f)
        r_f = f-Amult(u,m)
        r_c = coarsen(R,r_f)
        m_c = int((u.size-1)/2)
        e_c = np.zeros(m_c)
        e_c = VCM(R@A@P,R,P,e_c,r_c,l-1,int(np.sqrt(m_c)))
        e_f = interpolate(P,e_c)
        u = u + e_f
        for _ in range(20):
            u = smooth(u,omega,m,f)
        
    return u

def run_VCM(u0,tol,maxiter):
    
    m = int(np.sqrt(u0.size))
    l = 2
    A = poisson_A5(m)
    f = poisson_b5(m)
    P = generate_P(m) 
    R = generate_R(m)

    uk = u0
    r = Amult(uk,m)-f
    iter = 0
    converge = False

    while not converge or iter > maxiter:
        uk =  VCM(A,R,P,uk,f,l,m)
        r = Amult(uk,m)-f
        iter += 1
        converge = np.max(np.abs(r)) < tol 
    
    res = dict()
    res["converged"] = converge
    res["u"] = uk
    res["iterations"] = iter

    return res




