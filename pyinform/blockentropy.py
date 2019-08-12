# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
Block entropy, also known as N-gram entropy [Shannon1948]_, is the the standard
Shannon entropy applied to the time series (or sequence) of :math:`k`-histories
of a time series (or sequence):

.. math::

    H(X^{(k)}) = -\\sum_{x^{(k)}_i} p(x^{(k)}_i) \\log_2 p(x^{(k)}_i)

which of course reduces to the traditional Shannon entropy for ``k == 1``. Much
as with :ref:`active-information`, the ideal usage is to take
:math:`k \\rightarrow \\infty`.

Examples
--------

A Single Initial Condition
^^^^^^^^^^^^^^^^^^^^^^^^^^

The typical usage is to provide the time series as a sequence (or
``numpy.ndarray``) and the block size as an integer and let the
:py:func:`block_entropy` sort out the rest:

.. doctest:: block_entropy

    >>> block_entropy([0,0,1,1,1,1,0,0,0], k=1)
    0.9910760598382222
    >>> block_entropy([0,0,1,1,1,1,0,0,0], k=1, local=True)
    array([[0.84799691, 0.84799691, 1.169925  , 1.169925  , 1.169925  ,
            1.169925  , 0.84799691, 0.84799691, 0.84799691]])

.. doctest:: block_entropy

    >>> block_entropy([0,0,1,1,1,1,0,0,0], k=2)
    1.811278124459133
    >>> block_entropy([0,0,1,1,1,1,0,0,0], k=2, local=True)
    array([[1.4150375, 3.       , 1.4150375, 1.4150375, 1.4150375, 3.       ,
            1.4150375, 1.4150375]])

Multiple Initial Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Do we support multiple initial conditions? Of course we do!

.. doctest:: block_entropy

    >>> series = [[0,0,1,1,1,1,0,0,0], [1,0,0,1,0,0,1,0,0]]
    >>> block_entropy(series, k=2)
    1.936278124459133
    >>> block_entropy(series, k=2, local=True)
    array([[1.4150375, 2.4150375, 2.4150375, 2.4150375, 2.4150375, 2.       ,
            1.4150375, 1.4150375],
           [2.       , 1.4150375, 2.4150375, 2.       , 1.4150375, 2.4150375,
            2.       , 1.4150375]])

Or you can compute the block entropy on each initial condition and average:

.. doctest:: block_entropy

    >>> import numpy as np
    >>> np.apply_along_axis(block_entropy, 1, series, 2).mean()
    1.686278124459133
"""
import numpy as np

from ctypes import byref, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard


def block_entropy(series, k, local=False):
    """
    Compute the (local) block entropy of a time series with block size *k*.

    :param series: the time series
    :type series: sequence or `numpy.ndarray`
    :param int k: the block size
    :param bool local: compute the local block entropy
    :returns: the average or local block entropy
    :rtype: float or `numpy.ndarray`
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
        q = max(0, m - k + 1)
        ai = np.empty((n, q), dtype=np.float64)
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
