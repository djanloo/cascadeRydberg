"""Simulation of the process of avalanche excitation of Rydberg atoms.

Uniformly distributed atoms in 3D spacee.
An atom is excited. From the eps-ball around him a random atom is excited.
The process is repeated over the union of the two eps-ball centered in
each atom.

"""
cimport cython
from libc.math cimport sqrt, log
from libc.stdlib cimport rand, srand

import numpy as np
cimport numpy as np

from . utils import square_dist, randzerone

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
cdef get_ij_neighbors(list neighbors, unsigned int i, unsigned int j, float [:] atom_position, unsigned int M, float upper_radius):
  # CELL NAVIGATION
  # This quirky one-line indexes the (i,j,k) coordinates of the neighbors
  # mapping the discrete interval [0, ... , 3**space_dim] into the set {(-1, -1, -1), (-1, -1, 0), ... (-1, 1, 0), ..}
  # of all the neighboring cells 
  cdef int neighboring_cell_index_on_axis
  neighboring_cell_index_on_axis = <int> (atom_position[j]/upper_radius) + (i//(3**j))%3 - 1
  if neighboring_cell_index_on_axis < 0 or neighboring_cell_index_on_axis >= M:
    return []
  else:
    neighbors = neighbors[neighboring_cell_index_on_axis]      
  # At this point neighbors is the list of neighbors in the i-th cell
  return neighbors

cdef add_core_to_blob( unsigned int atom_index, float [:,:] S,
                        list cells, unsigned int M, 
                        float lower_radius, float upper_radius, 
                        float square_lower_radius, float square_upper_radius,
                        unsigned int [:] current_topological_state):
  """
  Given the current topology defined by (positions, topological_state) adds a point to the blob.

  This is done by assigning the topological state of the r-ball of the new core as INTERNAL or SHELL.
  """
  # Cell list navigation working variables
  cdef list neighbors
  cdef float [:] atom_position = S[atom_index]
  cdef unsigned int space_dim = len(atom_position)
  cdef unsigned int k, j, neighbor_index
  cdef float sq_dist
  for i in range(3**space_dim):
    neighbors = cells.copy()
    for j in range(space_dim):
      neighbors = get_ij_neighbors(neighbors, i, j, atom_position,  M, upper_radius)
      if neighbors == []:
        break
    # TOPOLOGY UPDATE
    # - INTERNAL -> INTERNAL
    # - SHELL -> SHELL/INTERNAL
    # - EXTERNAL -> SHELL/INTERNAL
    for k in range(len(neighbors)):
      neighbor_index = <int> neighbors[k]
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
  # Set the core atom as CORE
  current_topological_state[atom_index] = CORE
  return 

cdef int delete_core_from_blob(list decaying_cores, float [:, :] S,
                          list cells, list double_cells, unsigned int M,
                          float lower_radius, float upper_radius, 
                          float square_lower_radius, float square_upper_radius,
                          unsigned int [:] current_topological_state):
  """Given the current topology defined by (positions, topological_state) deletes a core point from the blob.
  
  This is done by assigning every point of the 2r-neighborhood of the deleted point as EXTERN (carving)
  and recomputing the topological state for every core (inflating).

  Cores are rearched in the 2r-neighborhood of the decaying core.
  """
  cdef unsigned int c, dc

  # Cell list navigation working variables
  cdef list neighbors
  cdef float [:] atom_position = S[0]
  cdef unsigned int space_dim = len(atom_position)
  cdef unsigned int k, j, neighbor_index, decaying_core_index 

  cdef float sq_dist
  cdef set cores_to_be_inflated_set = set([])
  cdef list cores_to_be_inflated = []
  if decaying_cores == []:
    return 0
  #################### CARVING ###################
  for dc in range(len(decaying_cores)):
    decaying_core_index = <int> decaying_cores[dc]
    current_topological_state[decaying_core_index] = EXTERNAL

    atom_position = S[decaying_core_index ]

    # Takes all the 2r-cell-neighborhood and sets it as EXTERN if not a core point
    for i in range(3**space_dim):
      neighbors = double_cells.copy()
      for j in range(space_dim):
        neighbors = get_ij_neighbors(neighbors, i, j, atom_position,  M//2 , 2*upper_radius)
        if neighbors == []:
          break
      
      for k in range(len(neighbors)):
        if current_topological_state[<int> neighbors[k]] == CORE:
          # Adds the neighboring core to the list of points to be inflated
          cores_to_be_inflated_set.add(neighbors[k])
        else:
          # Carves a cube from the blob
          ## TEST: DANGEROUS
          if square_dist(atom_position, S[neighbors[k]], space_dim) < square_upper_radius:
            current_topological_state[<int> neighbors[k]] = EXTERNAL

  ############### INFLATING #############
  cores_to_be_inflated = list(cores_to_be_inflated_set)
  # print(f"Decayed: {decaying_cores}")
  # print(f"Cores to be inflated: {cores_to_be_inflated}")
  # For the inflating phase only the r-cell-neighborhood is required
  # It is not possible to call sdd_point_to_blob because shell points are lost
  for c in range(len(cores_to_be_inflated)):
    atom_position = S[<int> cores_to_be_inflated[c]]
    for i in range(3**space_dim):
      neighbors = cells.copy()
      for j in range(space_dim):
        neighbors = get_ij_neighbors(neighbors, i, j, atom_position,  M, upper_radius)
        if neighbors == []:
          break

      # TOPOLOGY UPDATE FOR INFLATING
      # - INTERNAL -> INTERNAL
      # - SHELL -> SHELL (because otherwise the fraction of shell already marked will be lost)
      # - EXTERNAL -> SHELL/INTERNAL
      for k in range(len(neighbors)):
        neighbor_index = <int> neighbors[k]
        neighbor_topological_state = current_topological_state[neighbor_index]
        if neighbor_topological_state == EXTERNAL:
          sq_dist = square_dist(atom_position, S[neighbor_index], space_dim)
          if sq_dist < square_lower_radius:
            current_topological_state[neighbor_index] = INTERNAL 
          elif sq_dist < square_upper_radius:
            current_topological_state[neighbor_index] = SHELL
        # TEST: DANGEROUS LINE
        if neighbor_topological_state == SHELL:
          sq_dist = square_dist(atom_position, S[neighbor_index], space_dim)
          if sq_dist < square_upper_radius:
            # current_topological_state[neighbor_index] = INTERNAL
            pass

  return len(cores_to_be_inflated)

def shells_by_cells(float [:,:] S, 
                    float r, float delta, initial_cores = None,
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
  cdef list cores = [], new_cores = [], this_iteration_decayed_cores = []

  cdef unsigned int [:] topological_state = np.zeros(N, dtype=np.uintc)

  # Topolgy update working variables
  cdef unsigned int i,j,k, nc, el_index, element_topological_state
  cdef float sq_dist

  cdef float upper_radius = (r + delta/2.0)
  cdef float lower_radius = (r - delta/2.0)
  cdef float square_upper_radius = upper_radius**2
  cdef float square_lower_radius = lower_radius**2

  ################ TIME DIAGNOSTIC  #############################
  cdef unsigned int [:] N_cores = np.zeros(N_iterations, dtype=np.uintc)
  cdef unsigned int [:] N_decayed = np.zeros(N_iterations, dtype=np.uintc)
  cdef unsigned int [:] N_inflated = np.zeros(N_iterations, dtype=np.uintc)

  ################ CELL LIST CREATION ###########################
  cdef list cells = get_cell_list(S, upper_radius)
  cdef list double_cells = get_cell_list(S, 2*upper_radius)

  ################ FIRST ATOM(s) EXCITATION ########################
  if initial_cores == None:
    for i in range(5):
      first_atom_index = rand()%N
      new_cores.append(first_atom_index)
      topological_state[first_atom_index] = CORE
  
  else:
    for i in range(len(initial_cores)):
      new_cores.append(initial_cores[i])
      topological_state[initial_cores[i]] = CORE

  cdef int number_of_cores = 0, exists_at_least_one_shell_atom = 0, exists_at_least_one_core_atom = 0
  cdef unsigned int iteration_count = 0
  cdef dict results = {}

  while iteration_count < N_iterations:
    ################ BEGIN TOPOLOGICAL UPDATE ###################
    
    # Adds the new tcore to the blob
    for nc in range(len(new_cores)):
      add_core_to_blob(new_cores[nc], S, cells,M, lower_radius, upper_radius, square_lower_radius, square_upper_radius, topological_state)  
      cores.append(new_cores[nc])
      number_of_cores += 1

    ################# END TOPOLOGICAL UPDATE ####################

    # print(f"it: {iteration_count}: added {len(new_cores)} cores")
    # Empties the new_cores list
    new_cores = []

    ################ BEGIN EXCITATION/DECAY #####################

    # Excites each shell atom with a fixed probability
    exists_at_least_one_shell_atom = 0
    exists_at_least_one_core_atom = 0
    this_iteration_decayed_cores = []
    for i in range(N):
      # Excite shell atoms with probability `excitation_probability`
      if topological_state[i] == SHELL:
        exists_at_least_one_shell_atom = 1
        if randzerone() < excitation_probability:
          new_cores.append(i) 
      if topological_state[i] == CORE:
        exists_at_least_one_core_atom = 1
        if randzerone() < decay_probability:
          topological_state[i] = EXTERNAL
          this_iteration_decayed_cores.append(i)
          cores.remove(i)

    # print(f"it: {iteration_count}: decayed {len(this_iteration_decayed_cores)} cores")
    N_inflated[iteration_count] = delete_core_from_blob(this_iteration_decayed_cores,S, cells, double_cells, M, lower_radius,upper_radius, square_lower_radius, square_upper_radius, topological_state)
    if (exists_at_least_one_core_atom == 0 and exists_at_least_one_shell_atom == 0 ):
      print(f"WARNING: excited population died at iteration {iteration_count}")
      results["died"] = True
      break
    ############### END EXCITATION/DECAY #########################
    N_cores[iteration_count] = len(cores)
    N_decayed[iteration_count] = len(this_iteration_decayed_cores)

    if iteration_count == N_iterations-1:
      # print("exited after topological update")
      results["died"] = False
      break


    iteration_count += 1
  # Returns the results as a dictionary
  results["state"] = S 
  results["cores"] = cores 
  results["topological_state"] = np.array(topological_state)
  results["N_cores_t"] = np.array(N_cores)
  results["N_decayed_t"] = np.array(N_decayed)
  results["N_inflated_t"] = np.array(N_inflated)
  return results