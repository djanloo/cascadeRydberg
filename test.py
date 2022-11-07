from cryd.ercolation import run_by_cells
import numpy as np
from matplotlib import pyplot as plt

S = np.random.uniform(0,1, size=(1000000, 2)).astype(np.float32)
cells = run_by_cells(S, 0.08)

for i in range(13):
    for j in range(13):
        plt.scatter(S[cells[i][j], 0], S[cells[i][j], 1])

plt.show()