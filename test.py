from cryd.ercolation import shells_by_cells
import numpy as np
from matplotlib import pyplot as plt

S = np.random.uniform(0,1, size=(100, 2)).astype(np.float32)
c = np.array(shells_by_cells(S, 0.1, 0.05))
plt.scatter(S[:, 0], S[:, 1], c=c)


plt.show()