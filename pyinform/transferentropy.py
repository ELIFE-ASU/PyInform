# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
`Transfer entropy`_ (TE) was introduced by [Schreiber2000]_ to quantify
information transfer between an information source and destination,
conditioning out shared history effects. TE was originally formulated
considering only the source and destination; however, many systems of interest
have more than just those two components. As such, it may be necessary to
condition the probabilities on the states of all "background" components in the
system. These two forms are sometimes called _apparent_ and _complete_ transfer
entropy, respectively ([Lizier2008]_).

This implementation of TE allows the user to condition the probabilities on
any number of background processes, within hardware limits of course. For
the subsequent description, take :math:`X` to be the source, :math:`Y`
the target, and :math:`\\mathcal{W}=\\left\\{W_1, \\ldots, W_l\\right\\}`
to be the background processes against which we'd like to condition. For
example, we might take the state of two nodes in a dynamical network as the
source and target, while all other nodes in the network are treated as the
background. Transfer entropy is then defined in terms of a time-local variant:

.. math::

    t_{X \\rightarrow Y,\\mathcal{W},i}(k) = \\log_2{\\frac{p(y_{i+1}, x_i~|~y^{(k)}_i, W_{\\{1,i\\}},\\ldots,W_{\\{l,i\\}})}{p(y_{i+1}~|~y^{(k)}_i, W_{\\{1,i\\}},\\ldots,W_{\\{l,i\\}})p(x_i~|~y^{(k)}_i,W_{\\{1,i\\}},\\ldots,W_{\\{l,i\\}})}}

Averaging in time we have

.. math::

    T_{Y \\rightarrow X,\\mathcal{W}}(k) = \\langle t_{X \\rightarrow Y,\\mathcal{W},i}(k) \\rangle_i

As in the case of :ref:`active-information` and :ref:`entropy-rate`, the
transfer entropy is formally defined as the limit of the :math:`k`-history
transfer entropy as :math:`k \\rightarrow \\infty`:

.. math::

    t_{Y \\rightarrow X,\\mathcal{W},i} = \\lim_{k \\rightarrow \\infty} t_{Y \\rightarrow X,\\mathcal{W},i}(k)
    \\quad \\textrm{and} \\quad
    T_{Y \\rightarrow X,\\mathcal{W}} = \\lim_{k \\rightarrow \\infty} T_{Y \\rightarrow X,\\mathcal{W}}(k),

but we do not provide limiting functionality in this library (yet!).

See [Schreiber2000]_, [Kraiser2002]_ and [Lizier2008]_ for more details.

.. _Transfer entropy: https://en.wikipedia.org/wiki/Transfer_entropy

Examples
--------

One initial condition, no background
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Just give us a couple of time series and tell us the history length and we'll
give you a number

.. doctest:: transfer_entropy

    >>> xs = [0,1,1,1,1,0,0,0,0]
    >>> ys = [0,0,1,1,1,1,0,0,0]
    >>> transfer_entropy(xs, ys, k=2)
    0.6792696431662097
    >>> transfer_entropy(ys, xs, k=2)
    0.0

or an array if you ask for it

.. doctest:: transfer_entropy

    >>> transfer_entropy(xs, ys, k=2, local=True)
    array([[1.       , 0.       , 0.5849625, 0.5849625, 1.5849625, 0.       ,
            1.       ]])
    >>> transfer_entropy(ys, xs, k=2, local=True)
    array([[0., 0., 0., 0., 0., 0., 0.]])

Two initial conditions, no background
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uhm, yes we can! (Did you really expect anything less?)

.. doctest:: transfer_entropy

    >>> xs = [[1,0,0,0,0,1,1,1,1], [1,1,1,1,0,0,0,1,1]]
    >>> ys = [[0,0,1,1,1,1,0,0,0], [1,0,0,0,0,1,1,1,0]]
    >>> transfer_entropy(xs, ys, k=2)
    0.693536138896192
    >>> transfer_entropy(xs, ys, k=2, local=True)
    array([[1.32192809, 0.        , 0.73696559, 0.73696559, 1.32192809,
            0.        , 0.73696559],
           [0.        , 0.73696559, 0.73696559, 1.32192809, 0.        ,
            0.73696559, 1.32192809]])

One initial condition, one background process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doctest:: transfer_entropy

    >>> xs = [0,1,1,1,1,0,0,0,0]
    >>> ys = [0,0,1,1,1,1,0,0,0]
    >>> ws = [0,1,1,1,1,0,1,1,1]
    >>> transfer_entropy(xs, ys, k=2, condition=ws)
    0.2857142857142857
    >>> transfer_entropy(xs, ys, k=2, condition=ws, local=True)
    array([[1., 0., 0., 0., 0., 0., 1.]])

One initial condition, two background processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doctest:: transfer_entropy

    >>> xs = [0,1,1,1,1,0,0,0,0]
    >>> ys = [0,0,1,1,1,1,0,0,0]
    >>> ws = [[1,0,1,0,1,1,1,1,1], [1,1,0,1,0,1,1,1,1]]
    >>> transfer_entropy(xs, ys, k=2, condition=ws)
    0.0
    >>> transfer_entropy(xs, ys, k=2, condition=ws, local=True)
    array([[0., 0., 0., 0., 0., 0., 0.]])

Two initial conditions, two background processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doctest:: transfer_entropy

    >>> xs = [[1,1,0,1,0,1,1,0,0],[0,1,0,1,1,1,0,0,1]]
    >>> ys = [[1,1,1,0,1,1,1,0,0],[0,0,1,0,1,1,1,0,0]]
    >>> ws = [[[1,1,0,1,1,0,1,0,1],[1,1,1,0,1,1,1,1,0]],
    ...       [[1,1,1,1,0,0,0,0,1],[0,0,0,1,1,1,1,0,1]]]
    >>> transfer_entropy(xs, ys, k=2)
    0.5364125003090668
    >>> transfer_entropy(xs, ys, k=2, condition=ws)
    0.3396348215831049
    >>> transfer_entropy(xs, ys, k=2, condition=ws, local=True)
    array([[ 1.       ,  0.       ,  0.       , -0.4150375,  0.       ,
             0.       ,  1.       ],
           [ 0.       ,  0.5849625,  1.       ,  0.5849625,  0.       ,
             1.       ,  0.       ]])
"""
import numpy as np

from ctypes import byref, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard


def transfer_entropy(source, target, k, condition=None, local=False):
    """
    Compute the local or average transfer entropy from one time series to
    another with target history length *k*. Optionally, time series can be
    provided against which to *condition*.

    :param source: the source time series
    :type source: sequence or ``numpy.ndarray``
    :param target: the target time series
    :type target: sequence or ``numpy.ndarray``
    :param int k: the history length
    :param condition: time series of any conditions
    :type condition: sequence or ``numpy.ndarray``
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
    cs = np.ascontiguousarray(condition, np.int32) if condition is not None else None

    if xs.shape != ys.shape:
        raise ValueError("source and target timeseries are different shapes")
    elif xs.ndim > 2:
        raise ValueError("source and target have too great a dimension; must be 2 or less")

    if cs is None:
        pass
    elif cs.ndim == 1 and cs.shape != xs.shape:
        raise ValueError("condition has a shape that's inconsistent with the source and target")
    elif cs.ndim == 2 and xs.ndim == 1 and cs.shape[1:] != xs.shape:
        raise ValueError("condition has a shape that's inconsistent with the source and target")
    elif cs.ndim == 2 and xs.ndim == 2 and cs.shape != xs.shape:
        raise ValueError("condition has a shape that's inconsistent with the source and target")
    elif cs.ndim == 3 and cs.shape[1:] != xs.shape:
        raise ValueError("condition has a shape that's inconsistent with the source and target")
    elif cs.ndim > 3:
        raise ValueError("condition has too great a dimension; must be 3 or less")

    ydata = ys.ctypes.data_as(POINTER(c_int))
    xdata = xs.ctypes.data_as(POINTER(c_int))
    cdata = cs.ctypes.data_as(POINTER(c_int)) if cs is not None else None

    if cs is None:
        b = max(2, max(np.amax(xs), np.amax(ys)) + 1)
    else:
        b = max(2, max(np.amax(xs), np.amax(ys), np.amax(cs)) + 1)

    if cs is None:
        z = 0
    elif cs.ndim == 1 or (cs.ndim == 2 and xs.ndim == 2):
        z = 1
    elif cs.ndim == 3 or (cs.ndim == 2 and xs.ndim == 1):
        z = cs.shape[0]
    else:
        raise RuntimeError("unexpected state: condition and source are inconsistent shapes")

    if xs.ndim == 1:
        n, m = 1, xs.shape[0]
    else:
        n, m = xs.shape

    e = ErrorCode(0)

    if local is True:
        q = max(0, m - k)
        te = np.empty((n, q), dtype=np.float64)
        out = te.ctypes.data_as(POINTER(c_double))
        _local_transfer_entropy(ydata, xdata, cdata, c_ulong(z), c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), out, byref(e))
    else:
        te = _transfer_entropy(ydata, xdata, cdata, c_ulong(z), c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), byref(e))

    error_guard(e)

    return te


_transfer_entropy = _inform.inform_transfer_entropy
_transfer_entropy.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int), c_ulong, c_ulong, c_ulong, c_int, c_ulong, POINTER(c_int)]
_transfer_entropy.restype = c_double

_local_transfer_entropy = _inform.inform_local_transfer_entropy
_local_transfer_entropy.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int), c_ulong, c_ulong, c_ulong, c_int, c_ulong, POINTER(c_double), POINTER(c_int)]
_local_transfer_entropy.restype = POINTER(c_double)
