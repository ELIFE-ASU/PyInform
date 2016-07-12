cimport cython.array
cimport numpy
import numpy

cdef extern from "stdint.h":
    ctypedef unsigned long uint64_t

cdef extern from "stdbool.h":
    ctypedef int bool

cdef extern from "inform/dist.h":
    ctypedef struct inform_dist:
        pass

    inform_dist* inform_dist_alloc(size_t n)
    inform_dist* inform_dist_copy(const inform_dist* src, inform_dist* dest)
    inform_dist* inform_dist_realloc(inform_dist* dist, size_t n)
    void inform_dist_free(inform_dist* dist)

    size_t inform_dist_size(const inform_dist* dist)
    uint64_t inform_dist_counts(const inform_dist* dist)
    bool inform_dist_is_valid(const inform_dist* dist)

    uint64_t inform_dist_get(const inform_dist* dist, uint64_t event)
    uint64_t inform_dist_set(inform_dist* dist, uint64_t event, uint64_t value)
    uint64_t inform_dist_tick(inform_dist* dist, uint64_t event)

    double inform_dist_prob(const inform_dist* dist, uint64_t event)
    int inform_dist_dump(const inform_dist* dist, double* probs, size_t n)

cdef class Dist:
    cdef inform_dist* _c_dist

    def __cinit__(self, n):
        """
        Construct an invalid distribution of size n.
        """
        if n <= 0:
            raise ValueError("distributions require positive, nonzero support")
        self._c_dist = inform_dist_alloc(n)
        if self._c_dist is NULL:
            raise MemoryError()

    def __dealloc__(self):
        """
        Deallocate the memory underlying the distribution.
        """
        if self._c_dist is not NULL:
            inform_dist_free(self._c_dist)

    def __len__(self):
        """
        Return the size of the support of the distribution.
        """
        return inform_dist_size(self._c_dist)

    def resize(self, n):
        """
        Resize the support of the distribution in place.
        """
        if n <= 0:
            raise ValueError("distributions require positive, nonzero support")
        self._c_dist = inform_dist_realloc(self._c_dist, n)
        if self._c_dist is NULL:
            raise MemoryError()

    def copy(self):
        """
        Perform a deep copy of the distribution.
        """
        d = Dist(len(self))
        inform_dist_copy(self._c_dist, d._c_dist)
        return d

    def counts(self):
        """
        Return the number of observations made thus far.
        """
        return inform_dist_counts(self._c_dist)

    def valid(self):
        """
        Determine if the distribution is a valid probability distribution, i.e.
        if the support is not empty and at least one observation has been made.
        """
        return inform_dist_is_valid(self._c_dist)

    def __getitem__(self, event):
        """
        Return the number of observations made of event.
        """
        if event >= len(self):
            raise IndexError()
        return inform_dist_get(self._c_dist, event)

    def __setitem__(self, event, value):
        """
        Set the number of observations of event which must be zero at minimum.
        """
        if event >= len(self):
            raise IndexError()
        inform_dist_set(self._c_dist, event, value)

    def tick(self, event):
        """
        Make a single observation of event and return the total number of
        observations of said event.
        """
        if event >= len(self):
            raise IndexError()
        return inform_dist_tick(self._c_dist, event)

    def probability(self, event):
        """
        Return the probability of an event.
        """
        if not self.valid():
            raise ValueError("invalid distribution")
        if event >= len(self):
            raise IndexError()
        return inform_dist_prob(self._c_dist, event)

    def dump(self):
        """
        Return a numpy array containing the probabilities of each of the possible events.
        """
        if not self.valid():
            raise ValueError("invalid distribution")
        n = len(self)
        probs = numpy.empty(n, dtype=numpy.float64)
        cdef double [:] arr = probs
        m = inform_dist_dump(self._c_dist, &arr[0], n)
        if m != n:
            raise RuntimeError("cannot dump the distribution")
        return probs

cdef extern from "inform/active_info.h":
    double inform_active_info(const uint64_t* series, size_t n, size_t m, uint64_t base, uint64_t k)
    int inform_local_active_info(const uint64_t *series, size_t n, size_t m, uint64_t base, uint64_t k, double *ai)

def activeinfo1d(arr, uint64_t k, uint64_t b, local):
    from math import isnan

    if len(arr) < k+1 or len(arr) == 0:
        raise ValueError("container is too short ({0}) for history length ({1})".format(len(arr),k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2,max(arr)+1)

    cdef uint64_t [:] ys = arr
    cdef double [:] a

    if local:
        ai = numpy.empty(len(arr) - k, numpy.float64)
        a = ai
        err = inform_local_active_info(&ys[0], 1, <uint64_t>len(arr), b, k, &a[0]);
        if err != 0:
            raise ValueError("invalid local active information computed")
        return ai
    else:
        ai = inform_active_info(&ys[0], 1, <uint64_t>len(arr), b, k)
        if isnan(ai):
            raise ValueError("invalid active information computed (NaN)")
        return ai

def activeinfo2d(arr, uint64_t k, uint64_t b, local):
    from math import isnan

    shape = arr.shape
    if shape[1] < k+1 or shape[0] == 0:
        raise ValueError("container is too short ({0}) for history length ({1})".format(len(arr),k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2,numpy.amax(arr)+1)

    cdef uint64_t [:] ys = arr.ravel()
    cdef double [:] a;

    if local:
        ai = numpy.empty((shape[0], shape[1]-k), dtype=numpy.float64)
        a = ai.ravel()
        err = inform_local_active_info(&ys[0], <uint64_t>shape[0], <uint64_t>shape[1], b, k, &a[0])
        if err != 0:
            raise ValueError("invalid local active information computed")
        return ai
    else:
        ai = inform_active_info(&ys[0], <uint64_t>shape[0], <uint64_t>shape[1], b, k)
        if isnan(ai):
            raise ValueError("invalid active information computed (NaN)")
        return ai

def activeinfo(xs, uint64_t k, uint64_t b = 0, local = False):
    array = numpy.asarray(xs, dtype=numpy.uint64)
    if array.ndim == 0:
        raise ValueError("active information is ill-defined on empty arrays")
    elif array.ndim == 1:
        return activeinfo1d(array, k, b, local)
    elif array.ndim == 2:
        return activeinfo2d(array, k, b, local)
    else:
        raise ValueError("arrays of dimension greater than 2 are not yet supported")

cdef extern from "inform/transfer_entropy.h":
    double inform_transfer_entropy(const uint64_t* seriesy, const uint64_t* seriesx, size_t n, size_t m, uint64_t base, uint64_t k)
    int inform_local_transfer_entropy(const uint64_t *seriesy, const uint64_t* seriesx, size_t n, size_t m, uint64_t base, uint64_t k, double *te)

def transferentropy1d(ys, xs, uint64_t k, uint64_t b, local):
    from math import isnan

    if len(ys) < k+1 or len(ys) == 0:
        raise ValueError("containers are too short ({0}) for history length ({1})".format(len(ys),k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2,max(xs)+1,max(ys)+1)

    cdef uint64_t [:] ysarr = ys
    cdef uint64_t [:] xsarr = xs
    cdef double [:] t;

    if local:
        te = numpy.empty(len(ys)-k, dtype=numpy.float64)
        t = te;
        err = inform_local_transfer_entropy(&ysarr[0], &xsarr[0], 1, <uint64_t>len(ys), b, k, &t[0])
        if err != 0:
            raise ValueError("invalid local transfer entropy computed")
        return te
    else:
        te = inform_transfer_entropy(&ysarr[0], &xsarr[0], 1, <uint64_t>len(ys), b, k)
        if isnan(te):
            raise ValueError("invalid transfer entropy computed (NaN)")
        return te

def transferentropy2d(ys, xs, uint64_t k, uint64_t b, local):
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
    cdef double [:] t

    if local:
        te = numpy.empty((shape[0], shape[1]-k), dtype=numpy.float64)
        t = te.ravel()
        err = inform_local_transfer_entropy(&ysarr[0], &xsarr[0], <uint64_t>shape[0], <uint64_t>shape[1], b, k, &t[0])
        if err != 0:
            raise ValueError("invalid local transfer entropy computed")
        return te
    else: 
        te = inform_transfer_entropy(&ysarr[0], &xsarr[0], <uint64_t>shape[0], <uint64_t>shape[1], b, k)
        if isnan(te):
            raise ValueError("invalid transfer entropy computed (NaN)")
        return te

def transferentropy(ys, xs, uint64_t k, uint64_t b = 0, local = False):
    ysarr = numpy.asarray(ys, dtype=numpy.uint64)
    xsarr = numpy.asarray(xs, dtype=numpy.uint64)
    if ysarr.shape != xsarr.shape:
        raise ValueError("the x and y time series must have the same shape")
    elif ysarr.ndim == 1:
        return transferentropy1d(ysarr, xsarr, k, b, local)
    elif ysarr.ndim == 2:
        return transferentropy2d(ysarr, xsarr, k, b, local)
    else:
        raise ValueError("arrays of dimension greater than 2 are not yet supported")

cdef extern from "inform/entropy_rate.h":
    double inform_entropy_rate(const uint64_t* series, size_t n, size_t m, uint64_t base, uint64_t k)
    double inform_local_entropy_rate(const uint64_t* series, size_t n, size_t m, uint64_t base, uint64_t k, double *er)

def entropyrate1d(arr, uint64_t k, uint64_t b, local):
    from math import isnan

    if len(arr) < k+1 or len(arr) == 0:
        raise ValueError("container is too short ({0}) for history length ({1})".format(len(arr),k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2,max(arr)+1)

    cdef uint64_t [:] ys = arr
    cdef double [:] e

    if local:
        er = numpy.empty(len(arr)-k, dtype=numpy.float64)
        e = er
        err = inform_local_entropy_rate(&ys[0], 1, <uint64_t>len(arr), b, k, &e[0])
        if err != 0:
            raise ValueError("invalid local entropy rate computed")
        return er
    else:
        er = inform_entropy_rate(&ys[0], 1, <uint64_t>len(arr), b, k)
        if isnan(er):
            raise ValueError("invalid entropy rate computed (NaN)")
        return er

def entropyrate2d(arr, uint64_t k, uint64_t b, local):
    from math import isnan

    shape = arr.shape
    if shape[1] < k+1 or shape[0] == 0:
        raise ValueError("container is too short ({0}) for history length ({1})".format(shape[1],k))

    if k == 0:
        raise ValueError("history length is too short")

    if b < 2:
        b = max(2,numpy.amax(arr)+1)

    cdef uint64_t [:] ys = arr.ravel()
    cdef double [:] e

    if local:
        er = numpy.empty((shape[0], shape[1]-k), dtype=numpy.float64)
        e = er.ravel()
        err = inform_local_entropy_rate(&ys[0], <uint64_t>shape[0], <uint64_t>shape[1], b, k, &e[0])
        if err != 0:
            raise ValueError("invalid local entropy rate computed")
        return er
    else:
        er = inform_entropy_rate(&ys[0], <uint64_t>shape[0], <uint64_t>shape[1], b, k)
        if isnan(er):
            raise ValueError("invalid entropy rate computed (NaN)")
        return er

def entropyrate(xs, uint64_t k, uint64_t b = 0, local = False):
    array = numpy.asarray(xs, dtype=numpy.uint64)
    if array.ndim == 0:
        raise ValueError("entropy rate is ill-defined on empty arrays")
    elif array.ndim == 1:
        return entropyrate1d(array, k, b, local)
    elif array.ndim == 2:
        return entropyrate2d(array, k, b, local)
    else:
        raise ValueError("arrays of dimension greater than 2 are not yet supported")
