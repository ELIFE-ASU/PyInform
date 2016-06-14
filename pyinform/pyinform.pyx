cimport cython.array
cimport numpy
import numpy

cdef extern from "stdint.h":
    ctypedef unsigned long uint64_t

cdef extern from "inform/dist.h":
    ctypedef struct inform_dist:
        pass

    inform_dist* inform_dist_alloc(size_t n)
    void inform_dist_free(inform_dist* dist)

    size_t inform_dist_size(const inform_dist* dist);

cdef class Dist:
    cdef inform_dist* _c_dist
    def __cinit__(self, n):
        if n <= 0:
            raise ValueError("distributions require positive, nonzero support")

        self._c_dist = inform_dist_alloc(n)
        if self._c_dist is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_dist is not NULL:
            inform_dist_free(self._c_dist)

    def __len__(self):
        return inform_dist_size(self._c_dist)

cdef extern from "inform/time_series.h":
    double inform_active_info(const uint64_t* series, size_t n, uint64_t base, uint64_t k)
    double inform_active_info_ensemble(const uint64_t* series, size_t n, size_t m, uint64_t base, uint64_t k)
    double inform_transfer_entropy(const uint64_t* seriesy, const uint64_t* seriesx, size_t n, uint64_t base, uint64_t k)
    double inform_transfer_entropy_ensemble(const uint64_t* seriesy, const uint64_t* seriesx, size_t n, size_t m, uint64_t base, uint64_t k)
    double inform_entropy_rate(const uint64_t* series, size_t n, uint64_t b, uint64_t k);
    double inform_entropy_rate_ensemble(const uint64_t* series, size_t n, size_t m, uint64_t b, uint64_t k);

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

def transferentropy1d(ys, xs, uint64_t k, uint64_t b):
    from math import isnan

    if len(ys) < k+1 or len(ys) == 0:
        raise ValueError("containers are too short ({0}) for history length ({1})".format(len(ys),k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2,max(xs)+1,max(ys)+1)

    cdef uint64_t [:] ysarr = ys
    cdef uint64_t [:] xsarr = xs
    te = inform_transfer_entropy(&ysarr[0], &xsarr[0], <uint64_t>len(ys), b, k)

    if isnan(te):
        raise ValueError("invalid transfer entropy computed (NaN)")

    return te

def transferentropy2d(ys, xs, uint64_t k, uint64_t b):
    from math import isnan

    shape = ys.shape
    if shape[1] < k+1 or shape[0] == 0:
        raise ValueError("container is too short ({0}) for history length ({1})".format(shape[1],k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2, numpy.amax(xs)+1, numpy.amax(ys)+1)

    cdef uint64_t [:] ysarr = ys.ravel()
    cdef uint64_t [:] xsarr = xs.ravel()
    te = inform_transfer_entropy_ensemble(&ysarr[0], &xsarr[0], <uint64_t>shape[0], <uint64_t>shape[1], b, k)

    if isnan(te):
        raise ValueError("invalid transfer entropy computed (NaN)")

    return te

def transferentropy(ys, xs, uint64_t k, uint64_t b = 0):
    ysarr = numpy.asarray(ys, dtype=numpy.uint64)
    xsarr = numpy.asarray(xs, dtype=numpy.uint64)
    if ysarr.shape != xsarr.shape:
        raise ValueError("the x and y time series must have the same shape")
    elif ysarr.ndim == 1:
        return transferentropy1d(ysarr, xsarr, k, b)
    elif ysarr.ndim == 2:
        return transferentropy2d(ysarr, xsarr, k, b)
    else:
        raise ValueError("arrays of dimension greater than 2 are not yet supported")

def entropyrate1d(arr, uint64_t k, uint64_t b):
    from math import isnan

    if len(arr) < k+1 or len(arr) == 0:
        raise ValueError("container is too short ({0}) for history length ({1})".format(len(arr),k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2,max(arr)+1)

    cdef uint64_t [:] ys = arr
    er = inform_entropy_rate(&ys[0], <uint64_t>len(arr), b, k)

    if isnan(er):
        raise ValueError("invalid entropy rate computed (NaN)")

    return er

def entropyrate2d(arr, uint64_t k, uint64_t b):
    from math import isnan

    shape = arr.shape
    if shape[1] < k+1 or shape[0] == 0:
        raise ValueError("container is too short ({0}) for history length ({1})".format(shape[1],k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2,numpy.amax(arr)+1)

    cdef uint64_t [:] ys = arr.ravel()
    er = inform_entropy_rate_ensemble(&ys[0], <uint64_t>shape[0], <uint64_t>shape[1], b, k)

    if isnan(er):
        raise ValueError("invalid entropy rate computed (NaN)")

    return er

def entropyrate(xs, uint64_t k, uint64_t b = 0):
    array = numpy.asarray(xs, dtype=numpy.uint64)
    if array.ndim == 0:
        raise ValueError("entropy rate is ill-defined on empty arrays")
    elif array.ndim == 1:
        return entropyrate1d(array, k, b)
    elif array.ndim == 2:
        return entropyrate2d(array, k, b)
    else:
        raise ValueError("arrays of dimension greater than 2 are not yet supported")
