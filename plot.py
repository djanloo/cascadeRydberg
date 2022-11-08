from cryd.ercolation import run_by_cells
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track

M = 50
samples = 10

n = np.zeros((samples, M))
eps = np.linspace(0.01,0.5,M)

for samp in track(range(samples)):
    S = np.random.uniform(0,1 , size=(1000, 3)).astype(np.float32)
    for i, r in enumerate(eps):
        n[samp, i] = run_by_cells(S, r)

up,down = np.quantile(n,[.1, .9], axis = 0)
mean = np.mean(n, axis = 0)

plt.plot(eps, mean, color = "k")
plt.fill_between(eps, down, up, color = 'orange', alpha=0.5)

plt.ylabel(r"$\langle N \rangle$")
plt.xlabel(r"$\epsilon$")
plt.show()