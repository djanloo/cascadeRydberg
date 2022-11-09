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

cpdef void set_seed(int seed):
  srand(seed)
  return

cdef float square_dist(float [:] a, float [:] b, int space_dim):
  cdef float sqdist = 0.0
  cdef int k

  for k in range(space_dim):
    sqdist += (a[k] - b[k])**2

  return sqdist

cdef list get_cell_list(float [:,:] S, float eps):
  """Returns the list of elements in each cell.

  For example: cell[1,4,3] = [1, 5, 87, 4]"""
  cdef int N = len(S)
  cdef int M = <int> (1.0/eps) + 1 # The number of cells per axis
  cdef int dim = len(S[0])

  ## Builds a list of shape (M,M, ... ,M , 1) where M is repeated dim times
  # So that (e.g. in 3D) cell[1,2,3] is a list of one element
  shape = (M,)
  for k in range(len(S[0])-1):
    shape += (M,)
  shape += (0,)
  cdef np.ndarray cells_np = np.empty(shape=shape)
  cdef list cells = cells_np.tolist()
  cdef list el

  for i in range(N):
    el = cells
    for k in range(dim):
      el = el[<int> (S[i, k]/eps)]
    el.append(i)

  return cells

cdef unsigned int EXTERNAL = 0
cdef unsigned int SHELL = 1
cdef unsigned int INTERNAL = 2
cdef unsigned int CORE = 3

cdef void add_point_to_blob( unsigned int atom_index, float [:,:] S,
                        list cells,
                        float r, float delta,
                        unsigned int [:] current_topological_state):
  # Cell list navigation working variables
  cdef unsigned int neighboring_cell_index_on_axis
  cdef list neighbors
  cdef float [:] atom_position = S[atom_index]
  cdef unsigned int space_dim = len(atom_position)
  cdef unsigned int k

  for i in range(3**space_dim):
    neighbors = cells.copy()
    for j in range(space_dim):
      # CELL NAVIGATION
      # This quirky one-line indexes the (i,j,k) coordinates of the neighbors
      # mapping the discrete interval [0, ... , 3**space_dim] into the set {(-1, -1, -1), (-1, -1, 0), ... (-1, 1, 0), ..}
      # of all the neighboring cells 
      neighboring_cell_index_on_axis = <int> (atom_position[j]/(r+delta/2.0)) + (i//(3**j))%3 - 1
      if neighboring_cell_index_on_axis < 0 or neighboring_cell_index_on_axis >= M:
        neighbors = []
        break
      neighbors = neighbors[neighboring_cell_index_on_axis]      
    # At this point neighbors is the list of neighbors in the i-th cell

    # TOPOLOGY UPDATE
    for k in range(len(neighbors)):
      neighbor_index = neighbors[k]
      neighbor_topological_state = current_topological_state[neighbor_index]
      if neighbor_topological_state != INTERNAL and neighbor_topological_state != CORE:
        # Here enter only SHELL and EXTERNAL points
        sq_dist = square_dist(atom_position, S[neighbor_index], space_dim)
        if neighbor_topological_state == SHELL:
          if sq_dist < square_upper_radius:
            current_topological_state[neighbor_index] = INTERNAL
          # else :
          #   print("\tleaved as SHELL")
        else:
          # Here enter only EXTERNAL points
          if sq_dist < square_lower_radius:
            current_topological_state[neighbor_index] = INTERNAL 
          elif sq_dist < square_upper_radius:
            current_topological_state[neighbor_index] = SHELL
    #     else:
    #       print("\tleaved as EXTERNAL")
    # 
  return 

def shells_by_cells(float [:,:] S, 
                    float r, float delta, 
                    float excitation_probability = 0.1, float decay_probability = 0.1,
                    unsigned int N_iterations = 100
                    ):
  """Use a cell binning to find the atoms in the shell of the whole blob,
  thn simulate the process.

  Iteration start from an already defined blob and relative topology and iteratively adds
  newly excited cores to the blob, updating the topological indicators.
  Iteration uses 3 lists:
    Core lists:
    - cores        (already excited atoms that define the blob)
    - new_cores    (excited in last iteration: must be added to the blob)
    Topology indicator:
    specify the topological relation of a point to the blob
    topological_state has values:
        - INTERNAL    (not excitable:  dist_from_nearest_core < r - delta/2)
        - SHELL       (excitable:      r - delta/2 < dist_from_all_cores < r + delta/2 )
        - EXTERNAL    (undetermined:   dist_from_all_cores > r + delta/2 )
        - CORE

  Suppositions: 
    - since atoms don't shut down, internal points will remain internal points
    - each atom that is connected to more than one atom will be inside the intersection of two shells, 
      so it will be marked as internal. Cores actions must be indipendent their topological state

  Algorithm:
    For each new_core:
    - get adjacent cells
    - for each point in adjacent cells:
      - if internal   do nothing
      - if shell      if dist < r+delta: mark as internal (double shell correction)
                      (here the case dist < r-delta is included)
      - if external   if dist < r-d: mark as internal
                      else if dist < r+d: mark as shell
  """
  # Preliminar operations
  cdef unsigned int N = len(S)
  if N == 0:
    exit("empty points") 
  cdef unsigned int space_dim = len(S[0])
  cdef unsigned int M = <int> (1.0/(r+delta/2.0)) + 1

  # Define main lists of the algorithm
  cdef list cores = [], new_cores = []
  cdef unsigned int [:] topological_state = np.zeros(N, dtype=np.uintc)

  # Topolgy update working variables
  cdef unsigned int i,j,k, nc, el_index, element_topological_state
  cdef float sq_dist
  cdef float square_upper_radius = (r + delta/2.0)**2
  cdef float square_lower_radius = (r - delta/2.0)**2


  ################ CELL LIST CREATION ###########################
  cdef list cells = get_cell_list(S, r+delta/2.0)

  ################ FIRST ATOM EXCITATION ########################
  first_atom_index = rand()%N
  new_cores.append(first_atom_index)
  topological_state[first_atom_index] = CORE

  cdef int number_of_cores = 0, exists_at_least_one_shell_atom = 0
  cdef unsigned int iteration_count = 0
  cdef dict results = {}

  while iteration_count < N_iterations:
    ################ BEGIN TOPOLOGICAL UPDATE ###################
    
    # Adds the new tcore to the blob
    for nc in range(len(new_cores)):
      add_point_to_blob(new_cores[nc], S, cells, r, delta, topological_state)
      
    # Transfer new_cores to cores
    for k in range(len(new_cores)):
      nc = new_cores[k]
      cores.append(nc)
      topological_state[nc] = CORE
      number_of_cores += 1
    ################# END TOPOLOGICAL UPDATE ####################

    # Empties the new_cores list
    new_cores = []
  
    if iteration_count == N_iterations-1:
      print("exited after topological update")
      break

    ################ BEGIN EXCITATION/DECAY #####################

    # Excites each shell atom with a fixed probability
    exists_at_least_one_shell_atom = 0
    for i in range(N):
      # Excite shell atoms with probability `excitation_probability`
      if topological_state[i] == SHELL:
        exists_at_least_one_shell_atom = 1
        if randzerone() < excitation_probability:
          new_cores.append(i) 

    ################## END EXCITATION ############################
    iteration_count += 1
  # Returns the results as a dictionary
  results["state"] = S 
  results["cores"] = cores 
  results["topological_state"] = topological_state

  return results