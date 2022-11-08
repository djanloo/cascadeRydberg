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
  cdef int space_dim = len(S[0])
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
        if square_dist(S[<int> excited[i]],  S[<int> not_reached_yet[j]], space_dim) <= square_eps:
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
cdef np.ndarray cell_indexes_template
cdef np.ndarray neighboring_cells_indexes_template


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

def run_by_cells(float [:,:] S, float eps):
  """The strategy is to divide the space in hypercubes using a position division:
  for example, if the space is [0,1]x[0,1] and eps == 0.1, a point that has x coordinate equal to 0.2 
  will be placed in the second column: col = int(x/eps).

  Then the eps-search is done only for the neighboring cells.
  """
  cdef int N = len(S)
  if N==0:
    exit("Empty array")
  cdef int space_dim = len(S[0])

  cdef int M = <int> (1.0/eps) + 1 # The number of boxes per axis
  cdef list cells = get_cell_list(S, eps)
  cdef float square_eps = eps**2
  cdef int [:,:] neighboring_cells

  cdef int i, k, j, e
  cdef list excited = [], excitables = [], newly_added_excited = [], el
  cdef int [:] is_already_reached = np.zeros(N, dtype=np.dtype("i"))
  cdef int neighboring_cell_index_on_axis
  cdef int [:] current_cell_indexes = np.zeros(space_dim, dtype=np.dtype("i")) 

  first_atom_index = rand()%N
  newly_added_excited.append(first_atom_index)
  is_already_reached[first_atom_index] = 1

  excitable = []

  while True:
    # Since excitable neighbors due to already excited atom are already computed
    # computes only the new excitable ones due to the atoms excited in the
    # previous iteration
    for e in range(len(newly_added_excited)):

      for i in range(3**space_dim):      
        # Starting i-th Cell-neighbors listing
        # where i is one of the 3**space_dim neighboring cells

        # Traveling the cell list
        el = cells.copy()
        for j in range(space_dim):
          # This quirky one-line indexes the (i,j,k) coordinates of the neighbors
          # mapping the discrete interval [0, ... , 3**space_dim] into the set {(-1, -1, -1), (-1, -1, 0), ... (-1, 1, 0), ..}
          # of all the neighboring cells 
          # HINT for future djanloo: try switching the i-for and the j-for to save time in accessing S(j) (does not depend on i, redundancy)
          neighboring_cell_index_on_axis = <int> (S[<int> newly_added_excited[e], j]/eps) + (i//(3**j))%3 - 1
          if neighboring_cell_index_on_axis < 0 or neighboring_cell_index_on_axis >= M:
            el = []
            break
          el = el[neighboring_cell_index_on_axis]
        # End of i-th Cell-neighbors listing
        # At this point el is the list of neighbors in the i-th cell

        # Then checks if the i-th cell elements
        for k in range(len(el)):
          if is_already_reached[<int> el[k]] == 0 and square_dist(S[<int> newly_added_excited[e]],  S[<int> el[k]], space_dim) <= square_eps:
            excitable.append(el[k])
            is_already_reached[<int> el[k]] = 1
      # Since the neighbors of the newly added axcited are computed
      # Simply stores it in the excited list
      excited.append(newly_added_excited[e])

    newly_added_excited = []

    if len(excitable) == 0:
        # print("no reachable elements")
        break
    else:
      # Selects a reachable atom and exites it
      new_excited_index = rand()%len(excitable)
      newly_added_excited.append(excitable[new_excited_index])
      
      # Removes the newly excited index from the excitable atoms
      del excitable[new_excited_index]

  return len(excited)

cdef unsigned int EXTERNAL = 0
cdef unsigned int SHELL = 1
cdef unsigned int INTERNAL = 2
cdef unsigned int CORE = 3


def shells_by_cells(float [:,:] S, float r, float delta):
  """Use a cellular division to find the atoms in the shell of the whole sausage,
  thn simulate the process.

  Iteration start from an already defined sausage and relative topology and iteratively adds
  newly excited cores to the sausage, updating the topological indicators.
  Iteration uses 3 lists:
    Core lists:
    - cores        (already excited atoms that define the sausage)
    - new_cores    (excited in last iteration: must be added to the sausage)
    Topology indicator:
    specify the topological relation of a point to the sausage
    - topology_state:
        has values:
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

  # Main lists of the algorithm
  cdef list cores = [], new_cores = []
  cdef unsigned int [:] topology_state = np.zeros(N, dtype=np.uintc)

  # Topolgy update working variables
  cdef unsigned int i,j,k, nc, el_index, element_topology_state
  cdef float sq_dist
  cdef float square_upper_radius = (r + delta/2.0)**2
  cdef float square_lower_radius = (r - delta/2.0)**2

  # print(f"square_upper_radius = {square_upper_radius}")
  # print(f"square_lower_radius = {square_lower_radius}")

  # Cell list navigation working variables
  cdef unsigned int neighboring_cell_index_on_axis, 
  cdef list el

  ################ CELL LIST CREATION ##########################
  cdef list cells = get_cell_list(S, r+delta/2.0)

  ################ FIRST ATOM EXCITATION #######################
  first_atom_index = rand()%N
  new_cores.append(first_atom_index)
  topology_state[first_atom_index] = CORE

  cdef int number_of_cores = 0, exists_at_least_one_shell_atom = 0, safety_flag=0
  while True:
    ################ BEGIN TOPOLOGICAL UPDATE ####################
    #
    for nc in range(len(new_cores)):
      
      for i in range(3**space_dim):
        el = cells.copy()
        for j in range(space_dim):
          # CELL NAVIGATION
          # This quirky one-line indexes the (i,j,k) coordinates of the neighbors
          # mapping the discrete interval [0, ... , 3**space_dim] into the set {(-1, -1, -1), (-1, -1, 0), ... (-1, 1, 0), ..}
          # of all the neighboring cells 
          neighboring_cell_index_on_axis = <int> (S[<int> new_cores[nc], j]/(r+delta/2)) + (i//(3**j))%3 - 1
          if neighboring_cell_index_on_axis < 0 or neighboring_cell_index_on_axis >= M:
            el = []
            break
          el = el[neighboring_cell_index_on_axis]      
        # At this point el is the list of neighbors in the i-th cell

        # TOPOLOGY UPDATE
        for k in range(len(el)):
          el_index = el[k]
          element_topology_state = topology_state[el_index]
          if element_topology_state != INTERNAL and element_topology_state != CORE:
            # Here enter only SHELL and EXTERNAL points
            sq_dist = square_dist(S[new_cores[nc]], S[el[k]], space_dim)
            if element_topology_state == SHELL:
              if sq_dist < square_upper_radius:
                topology_state[el_index] = INTERNAL
              # else :
              #   print("\tleaved as SHELL")
            else:
              # Here enter only EXTERNAL points
              if sq_dist < square_lower_radius:
                topology_state[el_index] = INTERNAL 
              elif sq_dist < square_upper_radius:
                topology_state[el_index] = SHELL
    #           else:
    #             print("\tleaved as EXTERNAL")
    # 
    ################# END TOPOLOGICAL UPDATE ####################

    # Transfer new_cores to cores
    for k in range(len(new_cores)):
      nc = new_cores[k]
      cores.append(nc)
      topology_state[nc] = CORE
      number_of_cores += 1

    # Empties the new_cores list
    new_cores = []

    ############### BEGIN EXCITATION #############################
    #
    # Excites each shell atom with a fixed probability
    exists_at_least_one_shell_atom = 0
    for i in range(N):
      if topology_state[i] == SHELL:
        exists_at_least_one_shell_atom = 1
        if randzerone() < 0.1:
          new_cores.append(i) 
    #
    ############### END EXCITATION ###############################
    if exists_at_least_one_shell_atom == 0:
      break

  return len(cores)