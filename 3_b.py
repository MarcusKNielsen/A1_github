from functions import*

k = 7
m = 2**k - 1 # m*m is the grid
u0 = np.zeros(m*m)
tol = 1e-4
maxiter = 100 
results = run_VCM(u0,tol,maxiter)

print(results["converged"])
print(results["iterations"])
plot_u_vec(results["u"])

omega = 2/3
"""
u_iter = np.zeros(m*m)
errors =  []
res = []
for _ in range(100):
    u_iter = smooth(u_iter,omega,m,f)
    errors.append(np.max(np.abs(compute_err(u_iter))))
    res.append(np.max(np.abs(Amult(u_iter,m)-f)))

plt.figure()
plt.loglog(np.arange(100),res,label="Res")
#plt.plot(np.arange(100),errors,label="Err")
plt.legend()
 """
#plot_u_vec(u_iter)

""" 
plot_multigrid(7, idx_from_zero=False)

I = np.eye(m*m)
h = 1/(m+1)
A = A.toarray()
G = I + omega*h**2*A/4
condition_number = np.linalg.cond(G)
print("cond of G", condition_number)

Gk = np.copy(G)
for k in range(10):
    Gk *= G

vals,vecs = np.linalg.eig(G)
eig_max = np.max(np.abs(vals))

condition_number = np.linalg.cond(vecs)
print("cond of R", condition_number)
print("Max eigenvalue",eig_max) """

#%% Convergence test
from numpy.linalg import norm

N = 7
start = 4
H = np.zeros(N-start)
E_inf = np.zeros(N-start)

for k in range(start,N):
    
    m = 2**k-1
    
    H[k-start] = 1/(m+1)

    u0 = np.zeros(m*m)

    results = run_VCM(u0,tol,maxiter)

    e_i = compute_err(results["u"])

    E_inf[k-start] = np.max(np.abs(e_i))

    
a,b = np.polyfit(np.log(H), np.log(E_inf), 1)
print(a)
plt.figure()
plt.plot(np.log(H),np.log(E_inf),"o-",color="green",label="Inifinity norm of global error")
plt.plot(np.log(H),b+a*np.log(H),color="red",label="Helper line of order 2")
plt.xlabel(r"$\log(h)$")
plt.ylabel(r"$\log(E)$")
plt.legend()
plt.savefig("Convergence plot.png")
