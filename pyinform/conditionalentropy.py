# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
`Conditional entropy`_ is a measure of the amount of information
required to describe a random variable :math:`Y` given knowledge of another
random variable :math:`X`. When applied to time series, two time series are used
to construct the empirical distributions and then
:py:func:`~.shannon.conditional_entropy` can be applied to yield

.. math::

    H_b(Y|X) = -\sum_{x_i, y_i} p(x_i, y_i) \\log_b \\frac{p(x_i, y_i)}{p(x_i)}.
    
This can be viewed as the time-average of the local conditional entropy

.. math::

    h_{b,i}(Y|X) = -\\log_b \\frac{p(x_i, y_i)}{p(x_i)}.


See [Cover1991]_ for more information.

.. _Conditional entropy: https://en.wikipedia.org/wiki/Conditional_entropy

Examples
--------

::

    >>> xs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]
    >>> ys = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1]
    >>> conditional_entropy(xs,ys)      # H(Y|X)
    0.5971071794515037
    >>> conditional_entropy(ys,xs)      # H(X|Y)
    0.5077571498797332
    >>> conditional_entropy(xs, ys, local=True)
    array([ 3.        ,  3.        ,  0.19264508,  0.19264508,  0.19264508,
            0.19264508,  0.19264508,  0.19264508,  0.19264508,  0.19264508,
            0.19264508,  0.19264508,  0.19264508,  0.19264508,  0.19264508,
            0.19264508,  0.4150375 ,  0.4150375 ,  0.4150375 ,  2.        ])
    >>> conditional_entropy(ys, xs, local=True)
    array([ 1.32192809,  1.32192809,  0.09953567,  0.09953567,  0.09953567,
            0.09953567,  0.09953567,  0.09953567,  0.09953567,  0.09953567,
            0.09953567,  0.09953567,  0.09953567,  0.09953567,  0.09953567,
            0.09953567,  0.73696559,  0.73696559,  0.73696559,  3.9068906 ])
"""
import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def conditional_entropy(xs, ys, bx=0, by=0, b=2.0, local=False):
    """
    Compute the (local) conditional entropy between two time series.
    
    This function expects the **condition** to be the first argument.
    
    The bases *bx* and *by* are inferred from their respective time series if
    they are not provided (or are 0). The minimum value in both cases is 2.
    
    This function explicitly takes the logarithmic base *b* as an argument.
    
    :param xs: the time series drawn from the conditional distribution
    :type xs: a sequence or ``numpy.ndarray``
    :param ys: the time series drawn from the target distribution
    :type ys: a sequence or ``numpy.ndarray``
    :param int bx: the base of the conditional time series
    :param int by: the base of the target time series
    :param double b: the logarithmic base
    :param bool local: compute the local conditional entropy
    :return: the local or average conditional entropy
    :rtype: float or ``numpy.ndarray``
    :raises ValueError: if the time series have different shapes
    :raises InformError: if an error occurs within the ``inform`` C call
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
