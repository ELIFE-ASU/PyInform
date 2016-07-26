# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import c_bool, c_double, c_uint, c_ulong, c_void_p, POINTER
from pyinform import _inform

class Dist:
    def __init__(self, n):
        """
        Construct a distribution.

        If `n` is a list or `numpy` array, then the histogram is populated
        with the values in the array. Otherwise `n` is assumed to be a
        number and an invalid distribution is created with `n` as the size
        of its support.
        """
        if isinstance(n, list) or isinstance(n, np.ndarray):
            xs = np.ascontiguousarray(n, dtype=np.uint32)
            if xs.ndim != 1:
                raise ValueError("support is multi-dimenstional")
            elif xs.size == 0:
                raise ValueError("support is empty")
            data = xs.ctypes.data_as(POINTER(c_uint))
            self._dist = _dist_create(data, xs.size)
        else:
            if n <= 0:
                raise ValueError("support is zero")
            self._dist = _dist_alloc(c_ulong(n))

        if not self._dist:
            raise MemoryError()

    def __dealloc__(self,n):
        """
        Deallocate the memory underlying the distribution.
        """
        if self._dist:
            _dist_free(self._dist)

    def __len__(self):
        """
        Return the size of the support of the distribution.
        """
        return int(_dist_size(self._dist))

    def resize(self, n):
        """
        Resize the support of the distribution in place.
        """
        if n <= 0:
            raise ValueError("support is zero")
        self._dist = _dist_realloc(self._dist, c_ulong(n))
        if not self._dist:
            raise MemoryError()

    def copy(self):
        """
        Perform a deep copy of the distribution.
        """
        d = Dist(len(self))
        _dist_copy(self._dist, d._dist)
        return d

    def counts(self):
        """
        Return the number of observations made thus far.
        """
        return _dist_counts(self._dist)

    def valid(self):
        """
        Determine if the distribution is a valid probability distribution, i.e.
        if the support is not empty and at least one observation has been made.
        """
        return _dist_is_valid(self._dist)

    def __getitem__(self, event):
        """
        Return the number of observations made of `event`.
        """
        if event < 0 or event >= len(self):
            raise IndexError()
        return _dist_get(self._dist, c_ulong(event))

    def __setitem__(self, event, value):
        """
        Set the number of observations of `event`.
        """
        if event < 0 or event >= len(self):
            raise IndexError()
        value = max(0, value)
        return _dist_set(self._dist, c_ulong(event), c_uint(value))

    def tick(self, event):
        """
        Make a single observation of `event` and return the total number
        of observations of said `event`.
        """
        if event < 0 or event >= len(self):
            raise IndexError()
        return _dist_tick(self._dist, c_ulong(event))

    def probability(self, event):
        """
        Return the probability of `event`.
        """
        if not self.valid():
            raise ValueError("invalid distribution")
        elif event < 0 or event >= len(self):
            raise IndexError()
        return _dist_prob(self._dist, c_ulong(event))

    def dump(self):
        """
        Return a numpy array containing the probabilities of each of the
        possible events.
        """
        if not self.valid():
            raise ValueError("invalid distribution")
        n = len(self)
        probs = np.empty(n, dtype=np.float64)
        data = probs.ctypes.data_as(POINTER(c_double))
        m = _dist_dump(self._dist, data, c_ulong(n))
        if m != n:
            raise RuntimeError("cannot dump the distribution")
        return probs

_dist_alloc = _inform.inform_dist_alloc
_dist_alloc.argtypes = [c_ulong]
_dist_alloc.restype = c_void_p

_dist_realloc = _inform.inform_dist_realloc
_dist_realloc.argtypes = [c_void_p, c_ulong]
_dist_realloc.restype = c_void_p

_dist_copy = _inform.inform_dist_copy
_dist_copy.argtypes = [c_void_p, c_void_p]
_dist_copy.restype = c_void_p

_dist_create = _inform.inform_dist_create
_dist_create.argtypes = [POINTER(c_uint), c_ulong]
_dist_create.restype = c_void_p

_dist_free = _inform.inform_dist_free
_dist_free.argtypes = [c_void_p]
_dist_free.restype = None

_dist_size = _inform.inform_dist_size
_dist_size.argtypes = [c_void_p]
_dist_size.restype = c_ulong

_dist_counts = _inform.inform_dist_counts
_dist_counts.argtypes = [c_void_p]
_dist_counts.restype = c_uint

_dist_is_valid = _inform.inform_dist_is_valid
_dist_is_valid.argtypes = [c_void_p]
_dist_is_valid.restype = c_bool

_dist_get = _inform.inform_dist_get
_dist_get.argtypes = [c_void_p, c_ulong]
_dist_get.restype = c_uint

_dist_set = _inform.inform_dist_set
_dist_set.argtypes = [c_void_p, c_ulong, c_uint]
_dist_set.restype = c_uint

_dist_tick = _inform.inform_dist_tick
_dist_tick.argtypes = [c_void_p, c_ulong]
_dist_tick.restype = c_uint

_dist_prob = _inform.inform_dist_prob
_dist_prob.argtypes = [c_void_p, c_ulong]
_dist_prob.restype = c_double

_dist_dump = _inform.inform_dist_dump
_dist_dump.argtypes = [c_void_p, POINTER(c_double), c_ulong]
_dist_dump.restype = c_ulong
