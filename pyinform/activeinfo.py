# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

_active_info = _inform.inform_active_info
_active_info.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_int, c_ulong, POINTER(c_int)]
_active_info.restype = c_double

def active_info(series, k, b=0):
    """
    Compute the active information of a timeseries
    """
    xs = np.asarray(series, np.int32)

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
    ai = _active_info(data, c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), byref(e))
    error_guard(e)
    return ai
