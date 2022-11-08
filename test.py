from cryd.ercolation import shells_by_cells
import numpy as np
from matplotlib import pyplot as plt

S = np.random.uniform(0,1, size=(1000, 2)).astype(np.float32)
c = 0
for _ in range(100):
    c += np.array(shells_by_cells(S, 0.1, 0.1, 0.99, 0.01))
print(c/100)
plt.show()