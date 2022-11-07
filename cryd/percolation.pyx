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

cdef float square_dist(float [:] a, float [:] b):
  cdef float sqdist = 0.0
  cdef int k

  for k in range(len(a)):
    sqdist += (a[k] - b[k])**2

  return sqdist

def run(float [:, :] S, float eps):
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
  cdef float square_eps = eps**2
  
  set_time_seed()

  cdef list excited = [] # The indexes of excited atoms
  cdef list reachable = []
  cdef list not_reached_yet = list(range(N))
  cdef list dummy = []

  # Start with a random excited atom
  first_atom_index = rand()%N
  excited.append(first_atom_index)
  not_reached_yet.remove(first_atom_index)

  # Given a list of excited atoms
  # Finds the index of every reachable atom
  while True:
    for i in range(len(excited)):
      for j in range(len(not_reached_yet)):
        if square_dist(S[excited[i]], S[not_reached_yet[j]]) <= square_eps:
          reachable.append(not_reached_yet[j])
          not_reached_yet[j] = -1
    
    for j in range(len(not_reached_yet)):
      if not_reached_yet[j] != -1:
        dummy.append(not_reached_yet[j])
      
    not_reached_yet = dummy.copy()
    dummy = []

    if len(reachable) == 0:
      # print("no reachable elements")
      break
    else:
      # print(f"{reachable}")
      # Selects a reachable atom and exites it
      new_excited_index = rand()%len(reachable)
      excited.append(reachable[new_excited_index])
      
      # Removes the newly excited index from the excitable atoms
      del reachable[new_excited_index]

  return len(excited)

### Starting cell list method

cdef int [:,:] get_neighboring_cells(float [:] point, float eps):
  """Returns the cell indexes of a given point given the point and lattice spacing"""
  cdef int k
  cdef int space_dim = len(points)
  cdef int [:] cell_indexes
  for k in range(space_dim):
    cell_indexes[k] = <Int> (point[k]/eps)

  # The number of neighboring cells is 3**space_dim - 1 and each cell
  # requires 3 indexes
  cdef np.ndarray dummy = np.zeros((3**space_dim - 1, space_dim))
  cdef int [:,:] neighboring_cells_indexes = a

  # Indexing is don by modular operations
  # Spans over every possible delta in each dimension
  # The self-cell is included
  for u in range(3**dim):
    for k in range(dim):
      delta = (u//(3**k))%3 - 1 
      neighboring_cells_indexes[u, k] = cell_indexes[k] + delta

def get_cell_list(float [:,:] S, float eps):
  """Returns the list of elements in each cell.

  For example: cell[1,4,3] = [1, 5, 87, 4]"""
  cdef int N = len(S)
  cdef int M = <int> (1.0/eps)

  shape = (M,)
  for k in range(len(S[0])-2):
    shape += (M,)
  shape += (1,)

  cdef np.ndarray cells_np = - np.ones(shape, dtype=np.dtype('i'))
  cdef list cells = cells_np.tolist()
  for i in range(N):
    indexes = [-1]*dim
    for k in dim:
      indexes[k] = int()
    cells[<int> ]

def run_by_cells(float [:,:] S, float eps):
  """The strategy is to divide the space in hypercubes using a position division:
  for example, if the space is [0,1]x[0,1] and eps == 0.1, a point that has x coordinate equal to 0.2 
  will be placed in the second column: col = int(x/eps).

  Then the eps-search is done only for the neighboring cells.
  """
  



