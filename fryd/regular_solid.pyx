from . import utils
from libc.stdlib cimport rand
import numpy as np

cdef extern from "limits.h":
    int INT_MAX

cdef float randzerone():
  return rand()/float(INT_MAX)

cdef unsigned int EXTERNAL = 0
cdef unsigned int SHELL = 1
cdef unsigned int INTERNAL = 2
cdef unsigned int CORE = 3

cdef add_to_blob(unsigned int [:,:] S, unsigned int i, unsigned int j):
    cdef int k,l
    S[i,j] = CORE

    for k in range(5):
        for l in range(5):
            if S[i + k - 2, j + l - 2] != CORE:
                if (k == 1 and l != 0 and l != 4 ) or (k==3 and l !=0 and l !=4 ) or (k == 2 and l !=0 and l!=2 and l!=4) :
                    S[i + k - 2, j + l - 2] = INTERNAL
                elif k == 0 or k == 4 or l == 0 or l == 4:
                    if S[i + k - 2, j + l - 2] != INTERNAL:
                        S[i + k - 2, j + l - 2] = SHELL
                    elif S[i + k - 2, j + l - 2] == SHELL:
                        S[i + k - 2, j + l - 2] = INTERNAL

cdef delete_from_blob(unsigned int [:,:] S, unsigned int i, unsigned int j):
    cdef list neighboring_cores = []
    cdef unsigned int k,l
    S[i,j] = EXTERNAL
    cdef int M = 15
    cdef int offset = M//2
    for k in range(M):
        for l in range(M):
            if (S[i + k - offset, j + l - offset] == CORE):
                neighboring_cores.append([i + k - offset, j + l - offset])
            else:
                S[i + k - offset, j + l - offset] = EXTERNAL
            
    for k in range(len(neighboring_cores)):
        add_to_blob(S, neighboring_cores[k][0], neighboring_cores[k][1])

def regular_lattice(unsigned int [:,:] S, 
                    excitation_probability=0.1, decay_probability=0.1, 
                    unsigned int N_iterations=100, startfrom=None):
    cdef unsigned int i, j ,k, iteration
    cdef unsigned int N = len(S)
    cdef unsigned int [:] N_of_cores = np.zeros(N_iterations, dtype=np.uintc)
    cdef unsigned int [:] N_of_decayed = np.zeros(N_iterations, dtype=np.uintc)
    cdef dict stats = dict([])

    ############## INIT ##############
    if startfrom is None:
        i = rand()%N
        j = rand()%N
        add_to_blob(S, i, j)
    else:
        S = startfrom
    
    for iteration in range(N_iterations):
        for i in range(10,N-10):
            for j in range(10, N-10):
                if S[i,j] == SHELL:
                    if randzerone() < excitation_probability:
                        add_to_blob(S, i, j)
                        N_of_cores[iteration] += 1
                if S[i,j] == CORE:
                    if randzerone() < decay_probability:
                        delete_from_blob(S, i, j)
                        N_of_decayed[iteration] += 1
                        N_of_cores[iteration] =- 1

    stats["state"] = np.array(S)
    stats["cores"] = np.array(N_of_cores)
    stats["decayed"] = np.array(N_of_decayed)
    return stats
