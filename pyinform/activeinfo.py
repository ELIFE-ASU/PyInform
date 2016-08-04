# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def active_info(series, k, b=0, local=False):
    """
    Compute the (local) active information of a timeseries with history length
    *k*.

    Active information was introduced in [Lizier2012]_ to quantify information
    storage in distributed computation. If :math:`\\{x_i\\}_{i \\geq 0}` are time
    series data of a discrete-time process and :math:`x^k_i` is the
    :math:`k`-length history at timestep :math:`i`, the local active information
    of the process is defined by

    .. math::

        A_x(k) &= \sum_{x^k_i,\, x_{i+1}} p(x^k_i, x_{i+1}) \\log_b \\frac{p(x^k_i, x_{i+1})}{p(x^k_i)p(x_{i+1})}.
    
    Examples: ::

        >>> active_info([0,0,1,1,1,1,0,0,0], k=2)
        0.3059584928680419

        >>> active_info([0,0,1,1,1,1,0,0,0], k=2, b=3)
        0.19303831650832826

        >>> active_info([0,0,1,1,1,1,0,0,0], k=2, local=True)
        array([[-0.19264508,  0.80735492,  0.22239242,  0.22239242, -0.36257008,
                1.22239242,  0.22239242]])

    :param series: the time series
    :type series: sequence or `numpy.ndarray`
    :param int k: the history length
    :param int b: the base of the time series and logarithm
    :param bool local: compute the local active information
    :returns: the average or local active information
    :rtype: float or `numpy.ndarray`

    .. [Lizier2012] J.T. Lizier, M. Prokopenko and A.Y. Zomaya, "`Local measures of information storage in complex distributed computation`__" Information Sciences, vol. 208, pp. 39-54, 2012.

    .. __: http://dx.doi.org/10.1016/j.ins.2012.04.016
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
        ai = np.empty((n,q), dtype=np.float64)
        out = ai.ctypes.data_as(POINTER(c_double))
        _local_active_info(data, c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), out, byref(e))
    else:
        ai = _active_info(data, c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), byref(e))

    error_guard(e)

    return ai

_active_info = _inform.inform_active_info
_active_info.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_int, c_ulong, POINTER(c_int)]
_active_info.restype = c_double

_local_active_info = _inform.inform_local_active_info
_local_active_info.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_int, c_ulong, POINTER(c_double), POINTER(c_int)]
_local_active_info.restype = POINTER(c_double)
