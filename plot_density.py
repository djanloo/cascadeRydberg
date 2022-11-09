from cryd.ercolation import shells_by_cells
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track

M = 50
samples = 500

r = 0.1
d = 0.05

n = np.zeros((samples, M))
Ns = 40*np.arange(1,M +1)

for p, color in zip(np.linspace(0.01, 1, 4), ["r", "g", "b", "orange"]):
    for samp in track(range(samples)):
        for i, N in enumerate(Ns):
            S = np.random.uniform(0,1 , size=(N, 3)).astype(np.float32)
            n[samp, i] = shells_by_cells(S, r, d, p, p)/N

    up,median,down = np.quantile(n,[.25, .5, .75], axis = 0)

    plt.plot(Ns, median, color = "k")
    plt.fill_between(Ns, down, up, color=color , alpha=0.5, label=f"p = {p}")

plt.legend()
plt.title(f"$r = {r}, \delta = {d}$")
plt.ylabel(r"$\langle N_{excited} / N \rangle$")
plt.xlabel(r"$\rho$")
plt.show()