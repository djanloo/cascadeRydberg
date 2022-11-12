"""Profiling examle using line_profiler.
"""
from os import chdir
from os.path import dirname, join
from line_profiler import LineProfiler
from fryd import ercolation
import numpy as np
from matplotlib import pyplot as plt

# Sets the working directory as the one with the code inside
# Without this line line_profiler won't find anything
chdir(join(dirname(__file__), "fryd"))


profile = LineProfiler()

profile.add_function(ercolation.shells_by_cells)
# profile.add_function(ercolation.delete_core_from_blob)
S = np.random.uniform(0,1 , size=(6000, 2)).astype(np.float32)

np.random.seed(43)

wrap = profile(ercolation.shells_by_cells)
for j in range(1):
    ercolation.set_seed(42)
    u = wrap(S, 0.02 ,0.05, 1e-2, 1e-4, N_iterations=1000)
profile.print_stats()
print(u)
plt.scatter(S[:,0], S[:,1], c=u["topological_state"], vmin=0, vmax=3)
plt.show()