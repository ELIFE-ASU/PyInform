# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_double, c_int, c_ulong, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def coalesce_series(series):
    """
    Coalesce a timeseries into as few contiguous states as possible.
    
    The magic of information measures is that the actual values of a time series
    are irrelavent. For example, :math:`\{0,1,0,1,1\}` has the same entropy as
    :math:`\{2,9,2,9,9\}` (possibly up to a rescaling). This give us the freedom
    to shift around the values of a time series as long as we do not change the
    relative number of states.
    
    This function thus provides a way of "compressing" a time series into as
    small a base as possible. For example ::
    
        >>> utils.coalesce_series([2,9,2,9,9])
        (array([0, 1, 0, 1, 1], dtype=int32), 2)
        
    Why is this useful? Many of the measures use the base of the time series to
    determine how much memory to allocate; the larger the base, the higher the
    memory usage. It also affects the overall performance as the combinatorics
    climb exponentially with the base.
    
    The two standard usage cases for this function are to reduce the base of a
    time series
    
        >>> utils.coalesce_series([0,2,0,2,0,2])
        (array([0, 1, 0, 1, 0, 1], dtype=int32), 2)
        
    or ensure that the states are non-negative
        
        >>> utils.coalesce_series([-8,2,6,-2,4])
        (array([0, 2, 4, 1, 3], dtype=int32), 5)
        
    Notice that the encoding that is used ensures that the ordering of the
    states stays the same, e.g.
    :math:`\{-8 \\rightarrow 0, -2 \\rightarrow 1, 2 \\rightarrow 2, 4 \\rightarrow 3, 6 \\rightarrow 4\}`.
    This isn't strictly necessary, so we are going to call it a "feature".
    
    :param sequence series: the time series to coalesce
    :return: the coalesced time series and its base
    :rtype: the 2-tuple (``numpy.ndarray``, int)
    :raises InformError: if an error occurs in the ``inform`` C call
    """
    xs = np.ascontiguousarray(series, dtype=np.int32)
    data = xs.ctypes.data_as(POINTER(c_int))

    cs = np.empty(xs.shape, dtype=np.int32)
    coal = cs.ctypes.data_as(POINTER(c_int))

    e = ErrorCode(0)
    b = _inform_coalesce(data, c_ulong(xs.size), coal, byref(e))
    error_guard(e)

    return cs, b

_inform_coalesce = _inform.inform_coalesce
_inform_coalesce.argtypes = [POINTER(c_int), c_ulong, POINTER(c_int), POINTER(c_int)]
_inform_coalesce.restype = c_int
