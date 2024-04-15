import numpy as np 

def Model(t,y):
    # Function from the problem description
    return y**2-y**3

def RK32(Model,h,yn,tn,A,b,c,d):

    # Internal states
    eps = np.zeros(3)
    # Function evaluations (reduces expensive functions evaluations)
    k = np.zeros(3)

    # Internal states
    eps[0] = yn 
    k[0] = Model(tn,eps[0]) # saving function evaluation
    eps[1] = yn + A[1,0]*h*k[0]
    k[1] = Model(tn+c[1]*h,eps[0]) # saving function evaluation
    eps[2] = yn + h*(A[2,0]*k[1] + A[2,1]*k[2])
    k[2] = Model(tn+c[2]*h,eps[1]) # saving function evaluation

    # Next step
    ynp1 = yn + h*np.dot(b,k)

    # Given from problem description, slide 96 week 9, IVP2
    err = h*np.abs(np.dot(d,k))

    # returning error estimate and function value
    return ynp1, err

def control_stepsize(h,tol,p,errp1):

    # Slide 102, week 9 IVP2
    h_opt = h*(tol/errp1)**(1/p)

    return h_opt

def Solve_RK32():

    yn = np.zeros(500000)
    tn = np.zeros(500000)
    delta = 0.2                 # from problem description
    y0 = delta 
    yn[0] = y0
    h = 2 
    reps = 1e-3
    aeps = 1e-3 
    tn[0] = 0 
    T = 2/delta
    iter = 0
    num_accept = 0

    # RK23
    A = np.zeros([3,3])          # internal step weights
    A[1,0] = 1/2                 # non zero values
    A[2,0] = -1                     
    A[2,1] = 2
    c = np.array([0,1/2,1])      # internal time step weights
    b3 = np.array([1/6,2/3,1/6]) # external step weights (third order)  
    b2 = np.array([1,-1,1])      # external step weights (second order)    
    d = b3-b2                    # error weights (The 'good' minus the 'bad')tn

    # Running until the desired time for function evaluation is reached
    while tn[iter] < T:
        
        # Computing the RK using the current h, time tn and function evaluation yn
        ynp1, errp1 = RK32(Model,h,yn[num_accept],tn[num_accept],A,b3,c,d) 

        # Computing the tolerance for error to adjust h
        tol = reps*np.abs(ynp1)+aeps

        # Accepting step
        acceptstep = (errp1 < tol)
        
        if acceptstep:
            # Updating accepted number of steps
            num_accept += 1

            # Updating the values
            yn[num_accept] = ynp1
            tn[num_accept] = tn[num_accept-1]+h
        
        if tn[num_accept] > T:
            return yn,tn,iter,num_accept
        elif iter % 100 == 0:  
            print(f"Progress: {tn[num_accept]/T*100:2f}%",end="\r")
        print(f"Progress: {tn[num_accept]/T*100:2f}%",end="\r")
        # Updating the step size h
        h = control_stepsize(h,tol,2,errp1) 

        # Ensuring that we do not overshoot the time 
        if tn[num_accept]+h>T:
            h = T-tn[num_accept] # Now we are at the final time T
        
        iter += 1
    
    return yn,tn,iter,num_accept

yn,tn,iter,num_accept = Solve_RK32()

True





    






