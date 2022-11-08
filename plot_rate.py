from cryd.ercolation import shells_by_cells
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track

N = 1000
samples = 100

r = 0.1
d = 0.1

M = 100
n = np.zeros((samples, M))
rates = np.linspace(0.05, 1, M)

for samp in track(range(samples)):
    for i, gamma in enumerate(rates):
        S = np.random.uniform(0,1 , size=(N, 3)).astype(np.float32)
        n[samp, i] = shells_by_cells(S, r, d, gamma, gamma)/N

up,median,down = np.quantile(n,[.2, .5, .8], axis = 0)

plt.plot(rates, median, color = "k")
plt.fill_between(rates, down, up, color = 'orange', alpha=0.5)

plt.ylabel(r"$\langle N_{excited} / N \rangle$")
plt.xlabel(r"excitation probability")
plt.show()