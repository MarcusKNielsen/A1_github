import matplotlib.pyplot as plt
from functions import *


m = 3

P = generate_P(m).todense()
R = generate_R(m).todense()


# Creating a subplot with 1 row and 2 columns
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Plotting matrix structure P
cax1 = ax[0].imshow(P, aspect='equal')
fig.colorbar(cax1, ax=ax[0])
ax[0].set_title("Matrix structure P")
ax[0].set_xlabel("k: node index")
ax[0].set_ylabel("k: node index")

# Plotting matrix structure R
cax2 = ax[1].imshow(R, aspect='equal')
fig.colorbar(cax2, ax=ax[1])
ax[1].set_title("Matrix structure R")
ax[1].set_xlabel("k: node index")
ax[1].set_ylabel("k: node index")

plt.tight_layout()
plt.show()