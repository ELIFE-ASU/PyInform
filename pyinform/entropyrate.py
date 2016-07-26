# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def entropy_rate(series, k, b=0, local=False):
    """
    Compute the entropy rate of a timeseries
    """
    xs = np.ascontiguousarray(series, np.int32)

    if xs.ndim == 0:
        raise ValueError("empty timeseries")
    elif xs.ndim > 2:
        raise ValueError("dimension greater than 2")

    if b == 0:
        b = max(2, np.amax(xs)+1)

    data = xs.ctypes.data_as(POINTER(c_int))
    if xs.ndim == 1:
        n, m = 1, xs.shape[0]
    else:
        n, m = xs.shape

    e = ErrorCode(0)

    if local is True:
        q = max(0, m - k)
        er = np.empty((n,q), dtype=np.float64)
        out = er.ctypes.data_as(POINTER(c_double))
        _local_entropy_rate(data, c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), out, byref(e))
    else:
        er = _entropy_rate(data, c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), byref(e))

    error_guard(e)

    return er

_entropy_rate = _inform.inform_entropy_rate
_entropy_rate.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_int, c_ulong, POINTER(c_int)]
_entropy_rate.restype = c_double

_local_entropy_rate = _inform.inform_local_entropy_rate
_local_entropy_rate.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_int, c_ulong, POINTER(c_double), POINTER(c_int)]
_local_entropy_rate.restype = POINTER(c_double)
