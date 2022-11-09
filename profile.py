"""Profiling examle using line_profiler.
"""
from os import chdir
from os.path import dirname, join
from line_profiler import LineProfiler
from fryd import ercolation
import numpy as np

# Sets the working directory as the one with the code inside
# Without this line line_profiler won't find anything
chdir(join(dirname(__file__), "fryd"))


profile = LineProfiler()

profile.add_function(ercolation.shells_by_cells)

S = np.random.uniform(0,1 , size=(1000, 3)).astype(np.float32)

np.random.seed(42)

wrap = profile(ercolation.shells_by_cells)
for j in range(1):
    ercolation.set_seed(42)
    print(wrap(S, 0.05, 0.1, 0.8, 0.8, N_iterations=1000)["cores"])

profile.print_stats()
