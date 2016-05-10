cimport cython.array
cimport numpy
import numpy

cdef extern from "stdint.h":
    ctypedef unsigned long uint64_t

cdef extern from "inform/time_series.h":
    double inform_active_info(const uint64_t* series, size_t n, uint64_t base, uint64_t k)
    double inform_active_info_ensemble(const uint64_t* series, size_t n, size_t m, uint64_t base, uint64_t k)

def activeinfo1d(arr, uint64_t k, uint64_t b):
    from math import isnan

    if len(arr) < k+1 or len(arr) == 0:
        raise ValueError("container is too short ({0}) for history length ({1})".format(len(arr),k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2,max(arr)+1)

    cdef uint64_t [:] ys = arr
    ai = inform_active_info(&ys[0], <uint64_t>len(arr), b, k)

    if isnan(ai):
        raise ValueError("invalid active information computed (NaN)")

    return ai

def activeinfo2d(arr, uint64_t k, uint64_t b):
    from math import isnan

    shape = arr.shape
    if shape[1] < k+1 or shape[0] == 0:
        raise ValueError("container is too short ({0}) for history length ({1})".format(len(arr),k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2,numpy.amax(arr)+1)

    cdef uint64_t [:] ys = arr.ravel()
    ai = inform_active_info_ensemble(&ys[0], <uint64_t>shape[0], <uint64_t>shape[1], b, k)

    if isnan(ai):
        raise ValueError("invalid active information computed (NaN)")

    return ai

def activeinfo(xs, uint64_t k, uint64_t b = 0):
    array = numpy.asarray(xs, dtype=numpy.uint64)
    if array.ndim == 0:
        raise ValueError("active information is ill-defined on empty arrays")
    elif array.ndim == 1:
        return activeinfo1d(array, k, b)
    elif array.ndim == 2:
        return activeinfo2d(array, k, b)
    else:
        raise ValueError("arrays of dimension greater than 2 are not yet supported")
