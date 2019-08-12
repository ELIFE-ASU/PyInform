# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
`Mutual information`_ (MI) is a measure of the amount of mutual dependence
between two random variables. When applied to time series, two time series are
used to construct the empirical distributions and then
:py:func:`~.shannon.mutual_info` can be applied. Locally MI is defined as

.. math::

    i_{i}(X,Y) = -\\log_2 \\frac{p(x_i, y_i)}{p(x_i)p(y_i)}.

The mutual information is then just the time average of :math:`i_{i}(X,Y)`.

.. math::

    I(X,Y) = -\\sum_{x_i, y_i} p(x_i, y_i) \\log_2 \\frac{p(x_i, y_i)}{p(x_i)p(y_i)}.


See [Cover1991]_ for more details.

.. _Mutual information: https://en.wikipedia.org/wiki/Mutual_information

Examples
--------

.. doctest:: mutual_info

    >>> xs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]
    >>> ys = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1]
    >>> mutual_info(xs, ys)
    0.21417094500762912
    >>> mutual_info(xs, ys, local=True)
    array([-1.        , -1.        ,  0.22239242,  0.22239242,  0.22239242,
            0.22239242,  0.22239242,  0.22239242,  0.22239242,  0.22239242,
            0.22239242,  0.22239242,  0.22239242,  0.22239242,  0.22239242,
            0.22239242,  1.5849625 ,  1.5849625 ,  1.5849625 , -1.5849625 ])
"""
import numpy as np

from ctypes import byref, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard


def mutual_info(xs, ys, local=False):
    """
    Compute the (local) mutual information between two time series.

    This function explicitly takes the logarithmic base *b* as an argument.

    :param xs: a time series
    :type xs: a sequence or ``numpy.ndarray``
    :param ys: a time series
    :type ys: a sequence or ``numpy.ndarray``
    :param bool local: compute the local mutual information
    :return: the local or average mutual information
    :rtype: float or ``numpy.ndarray``
    :raises ValueError: if the time series have different shapes
    :raises InformError: if an error occurs within the ``inform`` C call
    """
    us = np.ascontiguousarray(xs, dtype=np.int32)
    vs = np.ascontiguousarray(ys, dtype=np.int32)
    if us.shape != vs.shape:
        raise ValueError("timeseries lengths do not match")

    series = np.ascontiguousarray([us.flatten(), vs.flatten()], dtype=np.int32)

    bx = max(2, np.amax(us) + 1)
    by = max(2, np.amax(vs) + 1)

    bs = np.ascontiguousarray([bx, by], dtype=np.int32)

    seriesdata = series.ctypes.data_as(POINTER(c_int))
    bsdata = bs.ctypes.data_as(POINTER(c_int))
    l, n = series.shape

    e = ErrorCode(0)

    if local is True:
        mi = np.empty(us.shape, dtype=np.float64)
        out = mi.ctypes.data_as(POINTER(c_double))
        _local_mutual_info(seriesdata, c_ulong(l), c_ulong(n), bsdata, out, byref(e))
    else:
        mi = _mutual_info(seriesdata, c_ulong(l), c_ulong(n), bsdata, byref(e))

    error_guard(e)

    return mi


_mutual_info = _inform.inform_mutual_info
_mutual_info.argtypes = [POINTER(c_int), c_ulong, c_ulong, POINTER(c_int), POINTER(c_int)]
_mutual_info.restype = c_double

_local_mutual_info = _inform.inform_local_mutual_info
_local_mutual_info.argtypes = [POINTER(c_int), c_ulong, c_ulong, POINTER(c_int), POINTER(c_double), POINTER(c_int)]
_local_mutual_info.restype = c_double
