"""Profiling examle using line_profiler.
"""
from os import chdir
from os.path import dirname, join
from line_profiler import LineProfiler
from cryd import ercolation
import numpy as np

# Sets the working directory as the one with the code inside
# Without this line line_profiler won't find anything
chdir(join(dirname(__file__), "cryd"))


profile = LineProfiler()

profile.add_function(ercolation.run_by_cells)

S = np.random.uniform(0,1 , size=(1000, 3)).astype(np.float32)

np.random.seed(42)

wrap = profile(ercolation.run_by_cells)
for j in range(10):
    ercolation.set_seed(42)
    print(wrap(S, 0.1))

profile.print_stats()
