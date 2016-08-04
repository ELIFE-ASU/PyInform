# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def block_entropy(series, k, b=0, local=False):
    """
    Compute the (local) active information of a timeseries with block size *k*.

    .. math::

        H_k(X) = -\sum_{x^{(k)}} p(x^{(k)}) \\log_b p(x^{(k)})

    Examples: ::

        >>> block_entropy([0,0,1,1,1,1,0,0,0], k=1)
        0.9910760598382222
        >>> block_entropy([0,0,1,1,1,1,0,0,0], k=1, local=True)
        array([[ 0.84799691,  0.84799691,  1.169925  ,  1.169925  ,  1.169925  ,
                1.169925  ,  0.84799691,  0.84799691,  0.84799691]])

        >>> block_entropy([0,0,1,1,1,1,0,0,0], k=2)
        1.811278124459133
        >>> block_entropy([0,0,1,1,1,1,0,0,0], k=2, local=True)
        array([[ 1.4150375,  3.       ,  1.4150375,  1.4150375,  1.4150375,
                3.       ,  1.4150375,  1.4150375]])

    :param series: the time series
    :type series: sequence or `numpy.ndarray`
    :param int k: the block size
    :param int b: the base of the logarithm
    :param bool local: compute the local block entropy
    :returns: the average or local block entropy
    :rtype: float or `numpy.ndarray`
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
        q = max(0, m - k + 1)
        ai = np.empty((n,q), dtype=np.float64)
        out = ai.ctypes.data_as(POINTER(c_double))
        _local_block_entropy(data, c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), out, byref(e))
    else:
        ai = _block_entropy(data, c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), byref(e))

    error_guard(e)

    return ai

_block_entropy = _inform.inform_block_entropy
_block_entropy.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_int, c_ulong, POINTER(c_int)]
_block_entropy.restype = c_double

_local_block_entropy = _inform.inform_local_block_entropy
_local_block_entropy.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_int, c_ulong, POINTER(c_double), POINTER(c_int)]
_local_block_entropy.restype = POINTER(c_double)
