# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def conditional_entropy(xs, ys, bx=0, by=0, b=2.0, local=False):
    """
    Compute the conditional entropy between two timeseries
    """
    us = np.ascontiguousarray(xs, dtype=np.int32)
    vs = np.ascontiguousarray(ys, dtype=np.int32)
    if us.shape != vs.shape:
        raise ValueError("timeseries lengths do not match")

    if bx == 0:
        bx = max(2, np.amax(us)+1)

    if by == 0:
        by = max(2, np.amax(vs)+1)

    xdata = us.ctypes.data_as(POINTER(c_int))
    ydata = vs.ctypes.data_as(POINTER(c_int))
    n = us.size

    e = ErrorCode(0)

    if local is True:
        ce = np.empty(us.shape, dtype=np.float64)
        out = ce.ctypes.data_as(POINTER(c_double))
        _local_conditional_entropy(xdata, ydata, c_ulong(n), c_int(bx), c_int(by), c_double(b), out, byref(e))
    else:
        ce = _conditional_entropy(xdata, ydata, c_ulong(n), c_int(bx), c_int(by), c_double(b), byref(e))

    error_guard(e)

    return ce

_conditional_entropy = _inform.inform_conditional_entropy
_conditional_entropy.argtypes = [POINTER(c_int), POINTER(c_int), c_ulong, c_int, c_int, c_double, POINTER(c_int)]
_conditional_entropy.restype = c_double

_local_conditional_entropy = _inform.inform_local_conditional_entropy
_local_conditional_entropy.argtypes = [POINTER(c_int), POINTER(c_int), c_ulong, c_int, c_int, c_double, POINTER(c_double), POINTER(c_int)]
_local_conditional_entropy.restype = c_double
