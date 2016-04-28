cimport cython.array
cimport numpy
import numpy

cdef extern from "stdint.h":
    ctypedef unsigned long uint64_t

cdef extern from "inform/time_series.h":
    double inform_active_info(const uint64_t* series, size_t n, uint64_t base, uint64_t k)

def activeinfo(xs, uint64_t k, uint64_t b = 0):
    if not xs:
        raise ValueError("active information not defined on empty containers")

    if b < 2:
        b = max(2,max(xs)+1)

    cdef uint64_t [:] ys = numpy.asarray(xs, dtype=numpy.uint64)
    return inform_active_info(&ys[0], <uint64_t>len(xs), b, k)
