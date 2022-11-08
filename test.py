from cryd.ercolation import run_by_cells
import numpy as np
from matplotlib import pyplot as plt

S = np.random.uniform(0,1, size=(1000, 2)).astype(np.float32)
print(run_by_cells(S, 0.08))

plt.show()