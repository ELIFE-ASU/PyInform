# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import c_bool, c_double, c_uint, c_ulong, c_void_p, POINTER
from pyinform import _inform

class Dist:
    """
    Dist is class designed to represent empirical probability distributions,
    i.e. histograms, for cleanly logging observations of time series data.

    The premise behind this class is that it allows **PyInform** to define
    the standard entropy measures on distributions. This reduces functions
    such as :py:func:`pyinform.activeinfo.active_info` to building
    distributions and then applying standard entropy measures.
    """

    def __init__(self, n):
        """
        Construct a distribution.

        If the parameter *n* is an integer, the distribution is constructed
        with a zeroed support of size *n*. If *n* is a list or
        ``numpy.ndarray``, the sequence is treated as the underlying support.

        Examples: ::

            d = Dist(5)
            d = Dist([0,0,1,2])

        :param n: the support for the distribution
        :type n: int, list or ``numpy.ndarray``
        :raises ValueError: if support is empty or multidimensional
        :raises MemoryError: if memory allocation fails within the C call
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

    def __dealloc__(self):
        """
        Deallocate the memory underlying the distribution.
        """
        if self._dist:
            _dist_free(self._dist)

    def __len__(self):
        """
        Determine the size of the support of the distribution.

        Examples: ::

            assert(len(Dist(5)) == 5)
            assert(len(Dist[0,1,5]) == 3)

        See also :py:meth:`.counts`.

        :return: the size of the support
        :rtype: int
        """
        return int(_dist_size(self._dist))

    def resize(self, n):
        """
        Resize the support of the distribution in place.

        If the distribution...

        - **strinks** - the last ``len(self) - n`` elements are lost, the rest are preserved
        - **grows** - the last ``n - len(self)`` elements are zeroed
        - **is unchanged** - well, that sorta says it all, doesn't it?

        Examples: ::

            d = Dist(5)
            d.resize(3)
            assert(len(d) == 3)
            d.resize(8)
            assert(len(d) == 8)

            d = Dist([1,2,3,4])
            d.resize(2)
            assert(len(d) == 2) # [1,2]
            d.resize(4)
            assert(len(d) == 4) # [1,2,0,0]

        :param int n: the desired size of the support
        :raises ValueError: if the requested size is zero
        :raises MemoryError: if memory allocation fails in the C call
        """
        if n <= 0:
            raise ValueError("support is zero")
        self._dist = _dist_realloc(self._dist, c_ulong(n))
        if not self._dist:
            raise MemoryError()

    def copy(self):
        """
        Perform a deep copy of the distribution.

        :returns: the copied distribution
        :rtype: :py:class:`pyinform.dist.Dist`
        """
        d = Dist(len(self))
        _dist_copy(self._dist, d._dist)
        return d

    def counts(self):
        """
        Return the number of observations made thus far.

        Examples: ::
    
            d = Dist(5)
            assert(d.counts() == 0)

            d = Dist([1,0,3,2])
            assert(d.counts() == 6)

        See also :py:meth:`.__len__`.

        :return: the number of observations
        :rtype: int
        """
        return _dist_counts(self._dist)

    def valid(self):
        """
        Determine if the distribution is a valid probability distribution, i.e.
        if the support is not empty and at least one observation has been made.

        Examples: ::

            d = Dist(5)
            assert(not d.valid())

            d = Dist([0,0,0,1])
            assert(d.valid())

        See also :py:meth:`.__len__` and :py:meth:`.counts`.
    
        :return: a boolean signifying that the distribution is valid
        :rtype: bool
        """
        return _dist_is_valid(self._dist)

    def __getitem__(self, event):
        """
        Get the number of observations made of *event*.

        Examples: ::

            d = Dist(5)
            for i, n in enumerate(d):
                assert(d[i] != 0)
                assert(n == 0)

            d = Dist([0,1,2,3])
            for i, n in enumerate(d):
                assert(d[i] == i)
                assert(n == i)

        See also :py:meth:`.__setitem__`, :py:meth:`.tick` and :py:meth:`.probability`.

        :param int event: the observed event
        :return: the number of observations of *event*
        :rtype: int
        :raises IndexError: if ``event < 0 or len(self) <= event``
        """
        if event < 0 or event >= len(self):
            raise IndexError()
        return _dist_get(self._dist, c_ulong(event))

    def __setitem__(self, event, value):
        """
        Set the number of observations of *event* to *value*.

        If *value* is negative, then the observation count is set to zero.

        Examples: ::

            d = Dist(5)
            for i, _ in enumerate(d):
                d[i] = i*i

            d = Dist([0,1,2,3])
            for i, n in enumerate(d):
                d[i] = 2 * n
            assert(list(d) == [0,2,4,6])
        
        See also :py:meth:`.__getitem__` and :py:meth:`.tick`.
        
        :param int event: the observed event
        :param int value: the number of observations
        :raises IndexError: if ``event < 0 or len(self) <= event``
        """
        if event < 0 or event >= len(self):
            raise IndexError()
        value = max(0, value)
        return _dist_set(self._dist, c_ulong(event), c_uint(value))

    def tick(self, event):
        """
        Make a single observation of *event*, and return the total number
        of observations of said *event*.

        Examples: ::

            d = Dist(5)
            for i, _ in enumerate(d):
                assert(d.tick(i) == 1)
            assert(list(d) == [1,1,1,1,1])

            d = Dist([0,1,2,3])
            for i, _ in enumerate(d):
                assert(d.tick(i) == i + 1)
            assert(list(d) == [1,2,3,4])

        See also :py:meth:`.__getitem__` and :py:meth:`.__setitem__`.
        
        :param int event: the observed event
        :return: the total number of observations of *event*
        :rtype: int
        :raises IndexError: if ``event < 0 or len(self) <= event``
        """
        if event < 0 or event >= len(self):
            raise IndexError()
        return _dist_tick(self._dist, c_ulong(event))

    def probability(self, event):
        """
        Compute the empiricial probability of an *event*.

        Examples: ::

            d = Dist([1,1,1,1])
            for i, _ in enumerate(d):
                assert(d.probability(i) == 1./4)

        See also :py:meth:`.__getitem__` and :py:meth:`.dump`.
        
        :param int event: the observed event
        :return: the empirical probability *event*
        :rtype: float
        :raises RuntimeError: if ``not self.valid()``
        :raises IndexError: if ``event < 0 or len(self) <= event``
        """
        if not self.valid():
            raise RuntimeError("invalid distribution")
        elif event < 0 or event >= len(self):
            raise IndexError()
        return _dist_prob(self._dist, c_ulong(event))

    def dump(self):
        """
        Compute the empirical probability of each observable event and return
        the result as an array.

        Examples: ::

            d = Dist([1,2,2,1])
            assert(list(d.dump()) == [1./6, 1./3, 1./3, 1./6])

        See also :py:meth:`.probability`.

        :return: the empirical probabilities of all o
        :rtype: ``numpy.ndarray``
        :raises RuntimeError: if ``not self.valid()`` or the dump fails in the C call
        :raises IndexError: if ``event < 0 or len(self) <= event``
        """
        if not self.valid():
            raise RuntimeError("invalid distribution")
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
