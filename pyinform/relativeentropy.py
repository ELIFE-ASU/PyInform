# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def relative_entropy(xs, ys, b=0, base=2.0, local=False):
    """
    Compute the relative entropy between two timeseries treating each as
    observations from a distribution.
    """
    us = np.ascontiguousarray(xs, dtype=np.int32)
    vs = np.ascontiguousarray(ys, dtype=np.int32)
    if us.shape != vs.shape:
        raise ValueError("timeseries lengths do not match")

    if b == 0:
        b = max(2, np.amax(us)+1, np.amax(vs)+1)

    xdata = us.ctypes.data_as(POINTER(c_int))
    ydata = vs.ctypes.data_as(POINTER(c_int))
    n = us.size

    e = ErrorCode(0)

    if local is True:
        re = np.empty(b, dtype=np.float64)
        out = re.ctypes.data_as(POINTER(c_double))
        _local_relative_entropy(xdata, ydata, c_ulong(n), c_int(b), c_double(base), out, byref(e))
    else:
        re = _relative_entropy(xdata, ydata, c_ulong(n), c_int(b), c_double(base), byref(e))

    error_guard(e)

    return re

_relative_entropy = _inform.inform_relative_entropy
_relative_entropy.argtypes = [POINTER(c_int), POINTER(c_int), c_ulong, c_int, c_double, POINTER(c_int)]
_relative_entropy.restype = c_double

_local_relative_entropy = _inform.inform_local_relative_entropy
_local_relative_entropy.argtypes = [POINTER(c_int), POINTER(c_int), c_ulong, c_int, c_double, POINTER(c_double), POINTER(c_int)]
_local_relative_entropy.restype = POINTER(c_double)
