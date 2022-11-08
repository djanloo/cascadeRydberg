from cryd.ercolation import run_by_cells
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track

M = 30
samples = 100
eps = 0.11

n = np.zeros((samples, M))
Ns = 50*np.arange(1,M +1)

for samp in track(range(samples)):
    for i, N in enumerate(Ns):
        S = np.random.uniform(0,1 , size=(N, 3)).astype(np.float32)
        n[samp, i] = run_by_cells(S, eps)/N

up,median,down = np.quantile(n,[.2, .5, .8], axis = 0)

plt.plot(Ns, median, color = "k")
plt.fill_between(Ns, down, up, color = 'orange', alpha=0.5)

plt.ylabel(r"$\langle N_{excited} / N \rangle$")
plt.xlabel(r"$\rho$")
plt.show()