import numpy as np
import matplotlib.pyplot as plt
from Burgersadvection import *
import time

eps = 0.01/np.pi
T = 1.6037/np.pi

m_test = np.array([200])
m_test = m_test + 1

#%%
"""
Non-uniform
"""

data_non_uni = np.zeros([len(m_test),8])

for i in range(len(m_test)):

    m = m_test[i]+1
    
    start = time.time()
    t,U,x,h,k,n = solve_Burgers_low_memory(T,m,eps,U_initial,non_uni=True)
    end  = time.time()
    
    x_idx = np.argsort(abs(x))[:3]
    x_idx = np.sort(x_idx)
    
    hl = (x[x_idx[1]] - x[x_idx[0]])
    hr = (x[x_idx[2]] - x[x_idx[1]])
    
    
    Du_left    = (U[x_idx[1]] - U[x_idx[0]])/hl
    Du_central = (U[x_idx[2]] - U[x_idx[0]])/(hr+hl)
    Du_right   = (U[x_idx[2]] - U[x_idx[1]])/hr
    
    
    data_non_uni[i,0] = m+2               # Number of points in grid
    data_non_uni[i,1] = h                 # h spatial step
    data_non_uni[i,2] = k                 # k time step
    data_non_uni[i,3] = n                 # n number of time step
    data_non_uni[i,4] = start-end         # execution time
    data_non_uni[i,5] = Du_left
    data_non_uni[i,6] = Du_central
    data_non_uni[i,7] = Du_right
    
    np.savetxt("data_non_uni.txt",data_non_uni,delimiter=",")
    
    #print(Du_left, Du_central, Du_right)

#%%
"""
Uniform
"""
data_uni = np.zeros([len(m_test),8])

for i in range(len(m_test)):

    m = m_test[i]+1
    
    start = time.time()
    t,U,x,h,k,n = solve_Burgers_low_memory(T,m,eps,U_initial,non_uni=False)
    end = time.time()
    
    x_idx = np.argsort(abs(x))[:3]
    x_idx = np.sort(x_idx)
    
    hl = (x[x_idx[1]] - x[x_idx[0]])
    hr = (x[x_idx[2]] - x[x_idx[1]])
    
    
    Du_left    = (U[x_idx[1]] - U[x_idx[0]])/hl
    Du_central = (U[x_idx[2]] - U[x_idx[0]])/(hr+hl)
    Du_right   = (U[x_idx[2]] - U[x_idx[1]])/hr
    
    
    data_uni[i,0] = m+2               # Number of points in grid
    data_uni[i,1] = h                 # h spatial step
    data_uni[i,2] = k                 # k time step
    data_uni[i,3] = n                 # n number of time step
    data_uni[i,4] = start-end         # execution time
    data_uni[i,5] = Du_left
    data_uni[i,6] = Du_central
    data_uni[i,7] = Du_right
    
    np.savetxt("data_uni.txt",data_uni,delimiter=",")


#%%



