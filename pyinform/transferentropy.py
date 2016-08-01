# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def transfer_entropy(source, target, k, b=0, local=False):
    """
    Compute the transfer entropy from one timeseries to another
    """
    ys = np.asarray(source, np.int32)
    xs = np.asarray(target, np.int32)

    if xs.shape != ys.shape:
        raise ValueError("source and target timeseries are different shapes")
    elif xs.ndim == 0:
        raise ValueError("empty timeseries")
    elif xs.ndim > 2:
        raise ValueError("dimension greater than 2")

    if b == 0:
        b = max(2, max(np.amax(xs),np.amax(ys)) + 1)

    ydata = ys.ctypes.data_as(POINTER(c_int))
    xdata = xs.ctypes.data_as(POINTER(c_int))
    if xs.ndim == 1:
        n, m = 1, xs.shape[0]
    else:
        n, m = xs.shape

    e = ErrorCode(0)

    if local is True:
        q = max(0, m - k)
        ai = np.empty((n,q), dtype=np.float64)
        out = ai.ctypes.data_as(POINTER(c_double))
        _local_transfer_entropy(ydata, xdata, c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), out, byref(e))
    else:
        ai = _transfer_entropy(ydata, xdata, c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), byref(e))

    error_guard(e)

    return ai

_transfer_entropy = _inform.inform_transfer_entropy
_transfer_entropy.argtypes = [POINTER(c_int), POINTER(c_int), c_ulong, c_ulong, c_int, c_ulong, POINTER(c_int)]
_transfer_entropy.restype = c_double

_local_transfer_entropy = _inform.inform_local_transfer_entropy
_local_transfer_entropy.argtypes = [POINTER(c_int), POINTER(c_int), c_ulong, c_ulong, c_int, c_ulong, POINTER(c_double), POINTER(c_int)]
_local_transfer_entropy.restype = POINTER(c_double)
