# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
`Transfer entropy`_ (TE) measures the amount of directed transfer of information
between two random processes. The local variant of TE is defined as

.. math::

    t_{Y \\rightarrow X, i}(k) = \\log_2 \\frac{p(x_{i+1}, y_i | x^{(k)}_i)}{p(x_{i+1} | x^{(k)}_i)p(y_i | x^{(k)}_i)}.

Averaging in time we have

.. math::

    T_{Y \\rightarrow X}(k) = \\sum_{x^{(k)}_i,\\, x_{i+1},\\, y_i} p(x_{i+1}, y_i, x^{(k)}_i) \\log_2 \\frac{p(x_{i+1}, y_i | x^{(k)}_i)}{p(x_{i+1} | x^{(k)}_i)p(y_i | x^{(k)}_i)}.

As in the case of :ref:`active-information` and :ref:`entropy-rate`, the
transfer entropy is formally defined as the limit of the :math:`k`-history
transfer entropy as :math:`k \\rightarrow \\infty`:

.. math::

    t_{Y \\rightarrow X,i} = \\lim_{k \\rightarrow \\infty} t_{Y \\rightarrow X,i}(k)
    \\quad \\textrm{and} \\quad
    T_{Y \\rightarrow X} = \\lim_{k \\rightarrow \\infty} T_{Y \\rightarrow X}(k),

but we do not provide limiting functionality in this library (yet!).

.. note::

    What we call "transfer entropy" is referred to as "apparent transfer
    entropy" in the parlance of [Lizier2008]_. A related quantity, complete
    transfer entropy, also considers the semi-infinite histories of all other
    random processes associated with the system. An implementation of
    complete transfer entropy is planned for a future release of
    `Inform <http://github.com/elife-asu/inform>`_/PyInform.


See [Schreiber2000]_, [Kraiser2002]_ and [Lizier2008]_ for more details.

.. _Transfer entropy: https://en.wikipedia.org/wiki/Transfer_entropy

Examples
--------

A Single Initial Condition
^^^^^^^^^^^^^^^^^^^^^^^^^^

Just give us a couple of time series and tell us the history length and we'll
give you a number

.. doctest:: transfer_entropy

    >>> xs = [0,0,1,1,1,1,0,0,0]
    >>> ys = [0,1,1,1,1,0,0,0,1]
    >>> transfer_entropy(ys, xs, k=1)
    0.8112781244591327
    >>> transfer_entropy(ys, xs, k=2)
    0.6792696431662097
    >>> transfer_entropy(xs, ys, k=1)
    0.21691718668869922
    >>> transfer_entropy(xs, ys, k=2) # pesky floating-point math
    0.0

or an array if you ask for it

.. doctest:: transfer_entropy

    >>> transfer_entropy(ys, xs, k=1, local=True)
    array([[0.4150375, 2.       , 0.4150375, 0.4150375, 0.4150375, 2.       ,
            0.4150375, 0.4150375]])
    >>> transfer_entropy(ys, xs, k=2, local=True)
    array([[1.       , 0.       , 0.5849625, 0.5849625, 1.5849625, 0.       ,
            1.       ]])
    >>> transfer_entropy(xs, ys, k=1, local=True)
    array([[ 0.4150375,  0.4150375, -0.169925 , -0.169925 ,  0.4150375,
             1.       , -0.5849625,  0.4150375]])
    >>> transfer_entropy(xs, ys, k=2, local=True)
    array([[0., 0., 0., 0., 0., 0., 0.]])

Multiple Initial Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uhm, yes we can! (Did you really expect anything less?)

.. doctest:: transfer_entropy

    >>> xs = [[0,0,1,1,1,1,0,0,0], [1,0,0,0,0,1,1,1,0]]
    >>> ys = [[1,0,0,0,0,1,1,1,1], [1,1,1,1,0,0,0,1,1]]
    >>> transfer_entropy(ys, xs, k=1)
    0.8828560636920488
    >>> transfer_entropy(ys, xs, k=2)
    0.693536138896192
    >>> transfer_entropy(xs, ys, k=1)
    0.15969728512148243
    >>> transfer_entropy(xs, ys, k=2)
    0.0


And local too

.. doctest:: transfer_entropy

    >>> transfer_entropy(ys, xs, k=1, local=True)
    array([[0.4150375 , 2.        , 0.67807191, 0.67807191, 0.67807191,
            1.4150375 , 0.4150375 , 0.4150375 ],
           [1.4150375 , 0.4150375 , 0.4150375 , 0.4150375 , 2.        ,
            0.67807191, 0.67807191, 1.4150375 ]])
    >>> transfer_entropy(ys, xs, k=2, local=True)
    array([[1.32192809, 0.        , 0.73696559, 0.73696559, 1.32192809,
            0.        , 0.73696559],
           [0.        , 0.73696559, 0.73696559, 1.32192809, 0.        ,
            0.73696559, 1.32192809]])
    >>> transfer_entropy(xs, ys, k=1, local=True)
    array([[ 0.5849625 ,  0.48542683, -0.25153877, -0.25153877,  0.48542683,
             0.36257008, -0.22239242, -0.22239242],
           [ 0.36257008, -0.22239242, -0.22239242,  0.5849625 ,  0.48542683,
            -0.25153877,  0.48542683,  0.36257008]])
    >>> transfer_entropy(xs, ys, k=2, local=True)
    array([[0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.]])
"""
import numpy as np

from ctypes import byref, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard


def transfer_entropy(source, target, k, local=False):
    """
    Compute the local or average transfer entropy from one time series to
    another with target history length *k*.

    :param source: the source time series
    :type source: sequence or ``numpy.ndarray``
    :param target: the target time series
    :type target: sequence or ``numpy.ndarray``
    :param int k: the history length
    :param bool local: compute the local transfer entropy
    :returns: the average or local transfer entropy
    :rtype: float or ``numpy.ndarray``
    :raises ValueError: if the time series have different shapes
    :raises ValueError: if either time series has no initial conditions
    :raises ValueError: if either time series is greater than 2-D
    :raises InformError: if an error occurs within the ``inform`` C call
    """
    ys = np.ascontiguousarray(source, np.int32)
    xs = np.ascontiguousarray(target, np.int32)

    if xs.shape != ys.shape:
        raise ValueError("source and target timeseries are different shapes")
    elif xs.ndim == 0:
        raise ValueError("empty timeseries")
    elif xs.ndim > 2:
        raise ValueError("dimension greater than 2")

    b = max(2, max(np.amax(xs), np.amax(ys)) + 1)

    ydata = ys.ctypes.data_as(POINTER(c_int))
    xdata = xs.ctypes.data_as(POINTER(c_int))
    if xs.ndim == 1:
        n, m = 1, xs.shape[0]
    else:
        n, m = xs.shape

    e = ErrorCode(0)

    if local is True:
        q = max(0, m - k)
        te = np.empty((n, q), dtype=np.float64)
        out = te.ctypes.data_as(POINTER(c_double))
        _local_transfer_entropy(ydata, xdata, None, c_ulong(0), c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), out, byref(e))
    else:
        te = _transfer_entropy(ydata, xdata, None, c_ulong(0), c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), byref(e))

    error_guard(e)

    return te


_transfer_entropy = _inform.inform_transfer_entropy
_transfer_entropy.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int), c_ulong, c_ulong, c_ulong, c_int, c_ulong, POINTER(c_int)]
_transfer_entropy.restype = c_double

_local_transfer_entropy = _inform.inform_local_transfer_entropy
_local_transfer_entropy.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int), c_ulong, c_ulong, c_ulong, c_int, c_ulong, POINTER(c_double), POINTER(c_int)]
_local_transfer_entropy.restype = POINTER(c_double)
