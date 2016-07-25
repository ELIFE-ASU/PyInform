# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_int, c_ulong, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def encode(state, b):
    """
    Encode a base-`b` array of integers into a single integer
    """
    xs = np.asarray(state, dtype=np.int32)
    data = xs.ravel().ctypes.data_as(POINTER(c_int))

    if xs.size == 0:
        raise ValueError("cannot encode an empty array")

    if b is None:
        b = min(2, np.amin(xs)+1)

    e = ErrorCode(0)
    encoding = _inform_encode(data, c_ulong(xs.size), c_int(b), byref(e))
    error_guard(e)

    return encoding

_inform_encode = _inform.inform_encode
_inform_encode.argtypes = [POINTER(c_int), c_ulong, c_int, POINTER(c_int)]
_inform_encode.restype = c_int
