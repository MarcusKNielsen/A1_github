import numpy as np
import matplotlib.pyplot as plt

def u(x):
    return np.exp(np.cos(x))

def p(x,x_val,u_val):

    Pn = 0

    for i in range(len(x_val)):
        L = 1
        for j in range(len(x_val)):
            if j != i:
                L*=(x-x_val[j])/(x_val[i]-x_val[j])
        
        Pn += u_val[i]*L

    return Pn

h = 0.5
x_val = [-h/2,h/2,3*h/2]
x = np.linspace(-h/2,3*h/2,100)

plt.plot(x,p(x,x_val,u(x_val)),label="p(x)")
plt.plot(x_val[0],u(x_val[0]),".",color="red")
plt.plot(x_val[1],u(x_val[1]),".",color="red")
plt.plot(x_val[2],u(x_val[2]),".",color="red",label="Interpolations points")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.legend()
plt.title("Interpolation plot")

#%% Error convergence test 
h_list =  np.logspace(-3,-1,10)
x_val_p1 = [-h_list/2,h_list/2]
x_val_p2 = [-h_list/2,h_list/2,3*h_list/2]

exact_value = u(0)
approx_value_p1 = p(0,x_val_p1,u(x_val_p1))
approx_value_p2 = p(0,x_val_p2,u(x_val_p2))
error_p1 = np.abs(exact_value-approx_value_p1)
error_p2 = np.abs(exact_value-approx_value_p2)

plt.figure()
plt.plot(np.log(h_list),np.log(error_p1),"-o",color="purple",label="Error of p1")
plt.plot(np.log(h_list),np.log(error_p2),"-o",color="green",label="Error of p2")
plt.plot(np.log(h_list),2*np.log(h_list),color="blue",label="Helper line of order 2")
plt.plot(np.log(h_list),4*np.log(h_list),color="red",label="Helper line of order 4")
plt.title("Convergence test")
plt.xlabel("h")
plt.ylabel("error")
plt.legend()
plt.show()














