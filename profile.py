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
profile.add_function(ercolation.run)
wrap = profile(ercolation.run)
wrap(np.random.uniform(0,1 , size=(100, 3)), 0.3)
profile.print_stats()
