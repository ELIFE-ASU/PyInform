# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
Active information (AI) was introduced in [Lizier2012]_ to quantify information
storage in distributed computation. Active information is defined in terms of
a temporally local variant

.. math::

    a_{X,i}(k) = \\log_2 \\frac{p(x^{(k)}_i, x_{i+1})}{p(x^{(k)}_i)p(x_{i+1})}.

where the probabilities are constructed empirically from the *entire* time
series. From the local variant, the temporally global active information as

.. math::

    A_X(k) = \\langle a_{X,i}(k) \\rangle_{i}
             = \\sum_{x^{(k)}_i,\\, x_{i+1}} p(x^{(k)}_i, x_{i+1}) \\log_2 \\frac{p(x^{(k)}_i, x_{i+1})}{p(x^{(k)}_i)p(x_{i+1})}.

Strictly speaking, the local and average active information are defined as

.. math::

    a_{X,i} = \\lim_{k \\rightarrow \\infty} a_{X,i}(k)
    \\quad \\textrm{and} \\quad
    A_X = \\lim_{k \\rightarrow \\infty} A_X(k),

but we do not provide limiting functionality in this library (yet!).

Examples
--------

A Single Initial Condition
^^^^^^^^^^^^^^^^^^^^^^^^^^

The typical usage is to provide the time series as a sequence (or
``numpy.ndarray``) and the history length as an integer and let the
:py:func:`active_info` sort out the rest:

.. doctest:: active_info

    >>> active_info([0,0,1,1,1,1,0,0,0], k=2)
    0.3059584928680418
    >>> active_info([0,0,1,1,1,1,0,0,0], k=2, local=True)
    array([[-0.19264508,  0.80735492,  0.22239242,  0.22239242, -0.36257008,
             1.22239242,  0.22239242]])

Multiple Initial Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

What about multiple initial conditions? We've got that covered!

.. doctest:: active_info

    >>> active_info([[0,0,1,1,1,1,0,0,0], [1,0,0,1,0,0,1,0,0]], k=2)
    0.35987902873686084
    >>> active_info([[0,0,1,1,1,1,0,0,0], [1,0,0,1,0,0,1,0,0]], k=2, local=True)
    array([[ 0.80735492, -0.36257008,  0.63742992,  0.63742992, -0.77760758,
             0.80735492, -1.19264508],
           [ 0.80735492,  0.80735492,  0.22239242,  0.80735492,  0.80735492,
             0.22239242,  0.80735492]])

As mentioned in :ref:`subtle-details`, averaging the AI for over the initial
conditions does not give the same result as constructing the distributions using
all of the initial conditions together.

.. doctest:: active_info

    >>> import numpy as np
    >>> series = np.asarray([[0,0,1,1,1,1,0,0,0], [1,0,0,1,0,0,1,0,0]])
    >>> np.apply_along_axis(active_info, 1, series, 2).mean()
    0.5845395307173363

Or if you are feeling verbose:

.. doctest:: active_info

    >>> ai = np.empty(len(series))
    >>> for i, xs in enumerate(series):
    ...     ai[i] = active_info(xs, k=2)
    ...
    >>> ai
    array([0.30595849, 0.86312057])
    >>> ai.mean()
    0.5845395307173363
"""

import numpy as np

from ctypes import byref, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard


def active_info(series, k, local=False):
    """
    Compute the average or local active information of a timeseries with history
    length *k*.

    :param series: the time series
    :type series: sequence or ``numpy.ndarray``
    :param int k: the history length
    :param bool local: compute the local active information
    :returns: the average or local active information
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
        ai = np.empty((n, q), dtype=np.float64)
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
