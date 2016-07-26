# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def mutual_info(xs, ys, bx=0, by=0, b=2.0, local=False):
    """
    Compute the mutual information between two timeseries
    """
    us = np.ascontiguousarray(xs, dtype=np.int32)
    vs = np.ascontiguousarray(ys, dtype=np.int32)
    if us.ndim != 1 or vs.ndim != 1:
        raise ValueError("dimension greater than 1")

    if len(us) != len(ys):
        raise ValueError("timeseries lengths do not match")

    if bx == 0:
        bx = max(2, np.amax(us)+1)

    if by == 0:
        by = max(2, np.amax(vs)+1)

    xdata = us.ctypes.data_as(POINTER(c_int))
    ydata = vs.ctypes.data_as(POINTER(c_int))
    n = len(us)

    e = ErrorCode(0)

    if local is True:
        mi = np.empty(n, dtype=np.float64)
        out = mi.ctypes.data_as(POINTER(c_double))
        _local_mutual_info(xdata, ydata, c_ulong(n), c_int(bx), c_int(by), c_double(b), out, byref(e))
    else:
        mi = _mutual_info(xdata, ydata, c_ulong(n), c_int(bx), c_int(by), c_double(b), byref(e))

    error_guard(e)

    return mi

_mutual_info = _inform.inform_mutual_info
_mutual_info.argtypes = [POINTER(c_int), POINTER(c_int), c_ulong, c_int, c_int, c_double, POINTER(c_int)]
_mutual_info.restype = c_double

_local_mutual_info = _inform.inform_local_mutual_info
_local_mutual_info.argtypes = [POINTER(c_int), POINTER(c_int), c_ulong, c_int, c_int, c_double, POINTER(c_double), POINTER(c_int)]
_local_mutual_info.restype = c_double
