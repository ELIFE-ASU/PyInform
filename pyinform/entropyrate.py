# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
`Entropy rate`_ (ER) quantifies the amount of information needed to describe the
:math:`X` given observations of :math:`X^{(k)}`. In other words, it is the
entropy of the time series conditioned on the :math:`k`-histories. The local
entropy rate

.. math::

    h_{X,i}(k) = \\log_2 \\frac{p(x^{(k)}_i, x_{i+1})}{p(x^{(k)}_i)}

can be averaged to obtain the global entropy rate

.. math::

    H_X(k) = \\langle h_{X,i}(k) \\rangle_{i}
             = \\sum_{x^{(k)}_i,\\, x_{i+1}} p(x^{(k)}_i, x_{i+1}) \\log_2 \\frac{p(x^{(k)}_i, x_{i+1})}{p(x^{(k)}_i)}.

Much as with :ref:`active-information`, the local and average entropy rates are
formally obtained in the limit

.. math::

    h_{X,i} = \\lim_{k \\rightarrow \\infty} h_{X,i}(k)
    \\quad \\textrm{and} \\quad
    H_X = \\lim_{k \\rightarrow \\infty} H_X(k),

but we do not provide limiting functionality in this library (yet!).

See [Cover1991]_ for more details.

.. _Entropy rate: https://en.wikipedia.org/wiki/Entropy_rate

Examples
--------

A Single Initial Condition
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's apply the entropy rate to a single initial condition. Typically, you will
just provide the time series and the history length, and let
:py:func:`.entropy_rate` take care of the rest:

.. doctest:: entropy_rate

    >>> entropy_rate([0,0,1,1,1,1,0,0,0], k=2)
    0.6792696431662095
    >>> entropy_rate([0,0,1,1,1,1,0,0,0], k=2, local=True)
    array([[1.       , 0.       , 0.5849625, 0.5849625, 1.5849625, 0.       ,
            1.       ]])
    >>> entropy_rate([0,0,1,1,1,1,2,2,2], k=2)
    0.39355535745192416

Multiple Initial Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Of course multiple initial conditions are handled.

.. doctest:: entropy_rate

    >>> series = [[0,0,1,1,1,1,0,0,0], [1,0,0,1,0,0,1,0,0]]
    >>> entropy_rate(series, k=2)
    0.6253491072973907
    >>> entropy_rate(series, k=2, local=True)
    array([[0.4150375, 1.5849625, 0.5849625, 0.5849625, 1.5849625, 0.       ,
            2.       ],
           [0.       , 0.4150375, 0.5849625, 0.       , 0.4150375, 0.5849625,
            0.       ]])
"""

import numpy as np

from ctypes import byref, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard


def entropy_rate(series, k, local=False):
    """
    Compute the average or local entropy rate of a time series with history
    length *k*.

    :param series: the time series
    :type series: sequence or ``numpy.ndarray``
    :param int k: the history length
    :param bool local: compute the local active information
    :returns: the average or local entropy rate
    :rtype: float or ``numpy.ndarray``
    :raises ValueError: if the time series has no initial conditions
    :raises ValueError: if the time series is greater than 2-D
    :raises InformError: if an error occurs within the ``inform`` C call
    """
    xs = np.ascontiguousarray(series, np.int32)

    if xs.ndim == 0:
        raise ValueError("empty timeseries")
    elif xs.ndim > 2:
        raise ValueError("dimension greater than 2")

    b = max(2, np.amax(xs) + 1)

    data = xs.ctypes.data_as(POINTER(c_int))
    if xs.ndim == 1:
        n, m = 1, xs.shape[0]
    else:
        n, m = xs.shape

    e = ErrorCode(0)

    if local is True:
        q = max(0, m - k)
        er = np.empty((n, q), dtype=np.float64)
        out = er.ctypes.data_as(POINTER(c_double))
        _local_entropy_rate(data, c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), out, byref(e))
    else:
        er = _entropy_rate(data, c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), byref(e))

    error_guard(e)

    return er


_entropy_rate = _inform.inform_entropy_rate
_entropy_rate.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_int, c_ulong, POINTER(c_int)]
_entropy_rate.restype = c_double

_local_entropy_rate = _inform.inform_local_entropy_rate
_local_entropy_rate.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_int, c_ulong, POINTER(c_double), POINTER(c_int)]
_local_entropy_rate.restype = POINTER(c_double)
