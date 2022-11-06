"""Simulation of the process of avalanche excitation of Rydberg atoms.

Uniformly distributed atoms in 3D spacee.
An atom is excited. From the eps-ball around him a random atom is excited.
The process is repeated over the union of the two eps-ball centered in
each atom.

"""
cimport cython
from cython.parallel import prange
from libc.math cimport sqrt, log
from libc.stdlib cimport rand, srand
from libc.time cimport time,time_t

import numpy as np
cimport numpy as np

cdef extern from "limits.h":
    int INT_MAX

"""Uniform between 0-1 in nogil mode"""
cdef float randzerone_nogil() nogil:
  return rand()/ float(INT_MAX)

"""Uniform in 0-1 in normal mode"""
cdef float randzerone():
  return rand()/float(INT_MAX)

cdef void set_time_seed():
  cdef time_t t = time(NULL)
  srand(t)
  return

cdef float dist(np.ndarray a, np.ndarray b):
  return np.sum( (a-b)**2 )

def run(np.ndarray S, float eps):
  """Does the simulation.
  
  Args
  ----
    S : np.array or memoryview
      the array of points (as a memoryview).
    eps: float
      the radius of the ball
  """

  cdef unsigned int N = len(S)
  cdef unsigned int i,j, new_excited_index
  
  set_time_seed()

  excited = [] # The indexes of excited atoms

  # Start with a random excited atom
  excited.append(rand()%N)

  # Given a list of excited atoms
  # Finds the index of every reachable atom
  reachable = []
  cdef int M = 0
  while M < 100:
    for i in range(len(excited)):
      for j in range(N):
        if j not in excited:
          if dist(S[excited[i]], S[j]) <= eps:
            if j not in reachable:
              reachable.append(j)

    if len(reachable) == 0:
      # print("no reachable elements")
      break
    else:
      # print(f"iteration {M}")
      # print(f"reachable = {reachable}")
      # print(f"excited = {excited}")
      # Selects a reachable atom and exites it

      new_excited_index = rand()%len(reachable)
      excited.append(reachable[new_excited_index])
      """Here the algorithm must be revised, 
      since fusing the balls means that all the reachable 
      points at this iteration will be rachable in the next one.
      
      A lot of computing power is wasted trashing everything."""
      reachable = []    
    M +=1 
  return excited
        



