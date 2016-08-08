# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
`Relative entropy`_, also known as the Kullback-Leibler divergence, measures the
amount of information gained in switching from a pior :math:`q_X` to a posterior
distribution :math:`p_X` *over the same support*. That is :math:`q_X` and
:math:`P` represent hypotheses of the distribution of some random variable
:math:`X.` Time series data sampled from the posterior and prior can be used to
estiamte those distributions, and the relative entropy can the be computed via a
call to :py:func:`~.shannon.relative_entropy`. The result is

.. math::

    D_{KL}(p||q) = \\sum_{x_i} p(x_i) \\log_b \\frac{p(x_i)}{q(x_i)}

which has as its local counterpart

.. math::

    d_{KL, i}(p||q) = \\log_b \\frac{p(x_i)}{q(x_i)}.
    
Note that the average in moving from the local to the non-local relative entropy
is taken over the posterior distribution.

See [Kullback1951]_ and [Cover1991]_ for more information.

.. _Relative entropy: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

Examples
--------

::

        >>> xs = [0,1,0,0,0,0,0,0,0,1]
        >>> ys = [0,1,1,1,1,0,0,1,0,0]
        >>> relative_entropy(xs, ys)
        0.27807190511263774
        >>> relative_entropy(ys, xs)
        0.3219280948873624

::

        >>> xs = [0,0,0,0]
        >>> ys = [0,1,1,0]
        >>> relative_entropy(xs, ys)
        1.0
        >>> relative_entropy(ys, xs)
        nan
"""

import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def relative_entropy(xs, ys, b=0, base=2.0, local=False):
    """
    Compute the local or global relative entropy between two time series
    treating each as observations from a distribution.
    
    The base *b* is inferred from the time series if it is not provided (or is
    0). The minimum value is 2.
    
    This function explicitly takes the logarithmic base *base* as an argument.
    
    :param xs: the time series sampled from the posterior distribution
    :type xs: a sequence or ``numpy.ndarray``
    :param ys: the time series sampled from the prior distribution
    :type ys: a sequence or ``numpy.ndarray``
    :param int b: the base of the time series
    :param double b: the logarithmic base
    :param bool local: compute the local relative entropy
    :return: the local or global relative entropy
    :rtype: float or ``numpy.ndarray``
    :raises ValueError: if the time series have different shapes
    :raises InformError: if an error occurs within the ``inform`` C call
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
