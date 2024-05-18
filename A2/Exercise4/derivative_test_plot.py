import numpy as np
import matplotlib.pyplot as plt

data_non_uni = np.loadtxt("data_non_uni_v1.txt",delimiter=",")
data_uni     = np.loadtxt("data_uni_v1.txt",delimiter=",")

m_test = data_uni[:,0]


#%%

fs = 12

plt.figure()
plt.plot(m_test,data_non_uni[:,5],"-o",label="Non-Uniform")
plt.plot(m_test,data_uni[:,5],"-o",label="Uniform")
plt.hlines(-152.00516,min(m_test),max(m_test)+25,linestyles="--",color="red",label="Theoretical value")
plt.xlabel("m: number of grid points",fontsize=fs)
plt.ylabel(r"$\dfrac{\partial u}{\partial x}(0,T)$",fontsize=fs)
plt.title(r"Derivative Estimate",fontsize=fs+2)
plt.legend(fontsize=fs)
plt.subplots_adjust(left=0.17, bottom=0.125, right=0.925, top=0.90)
plt.show()

#%%

fig,axes = plt.subplots(2,figsize=(8, 8))

fs = 12
plt.suptitle(r"Derivative Estimate", fontsize=fs+4)

ax = axes[0]
ax.plot(m_test,data_non_uni[:,5],"-o",label="Non-Uniform")
ax.plot(m_test,data_uni[:,5],"-o",label="Uniform")
ax.set_xlabel("m: Number of grid points")
ax.set_ylabel(r"$\dfrac{\partial u}{\partial x}(0,T)$")
ax.hlines(-152.00516,min(m_test),max(m_test)+25,linestyles="--",color="red",label="Theoretical value")
ax.legend()


ax = axes[1]
ax.plot(m_test,-data_non_uni[:,4]/60,"-o",label="Non-Uniform")
ax.plot(m_test,-data_uni[:,4]/60,"-o",label="Uniform")
#ax.set_title(r"Derivative Estimate")
ax.set_xlabel("m: Number of grid points")
ax.set_ylabel(r"Compute time [minutes]")
ax.legend()

plt.subplots_adjust(wspace=0.3, hspace=0.2, bottom=0.1,left=0.15, right=0.95, top = 0.93)

plt.show()


