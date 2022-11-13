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
