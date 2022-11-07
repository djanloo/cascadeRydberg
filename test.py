from cryd.ercolation import run_by_cells
import numpy as np

S = np.random.uniform(0,1, size=(100, 2)).astype(np.float32)

run_by_cells(S, 0.1)