# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_double, c_int, c_ulong, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def coalesce_series(series):
    """
    Coalesce a timeseries into as few contiguous states as possible
    """
    xs = np.asarray(series, dtype=np.int32)
    data = xs.ravel().ctypes.data_as(POINTER(c_int))

    cs = np.empty(xs.size, dtype=np.int32)
    coal = cs.ctypes.data_as(POINTER(c_int))

    e = ErrorCode(0)
    b = _inform_coalesce(data, c_ulong(xs.size), coal, byref(e))
    error_guard(e)

    cs = np.reshape(cs, xs.shape)

    return cs, b

_inform_coalesce = _inform.inform_coalesce
_inform_coalesce.argtypes = [POINTER(c_int), c_ulong, POINTER(c_int), POINTER(c_int)]
_inform_coalesce.restype = c_int