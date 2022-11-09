from fryd.ercolation import shells_by_cells
import numpy as np
from matplotlib import pyplot as plt
N = 1000000
T = 100
p = 0.1

activation = np.zeros(T)
for p in [0.1, 0.2, 0.3]:
    S = np.ones(N)
    for t in range(T):
        u = np.random.uniform(0,1, size=N)
        S[u<p] = 0
        activation[t] = np.sum(S)/N

    plt.plot(activation)
plt.yscale("log")
plt.show()
