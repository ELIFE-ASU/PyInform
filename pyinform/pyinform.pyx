cimport cython.array
cimport numpy
import numpy

cdef extern from "stdint.h":
    ctypedef unsigned long uint64_t

cdef extern from "inform/time_series.h":
    double inform_active_info(const uint64_t* series, size_t n, uint64_t base, uint64_t k)

def activeinfo(xs, uint64_t k, uint64_t b = 0):
    from math import isnan

    if len(xs) < k+1 or len(xs) == 0:
        raise ValueError("container is too short ({0}) for history length ({1})".format(len(xs),k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2,max(xs)+1)

    cdef uint64_t [:] ys = numpy.asarray(xs, dtype=numpy.uint64)
    ai = inform_active_info(&ys[0], <uint64_t>len(xs), b, k)

    if isnan(ai):
        raise ValueError("invalid active information computed (NaN)")
    return ai
