# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
`Entropy rate`_ (ER) quantifies the amount of information needed to describe the
:math:`X` given observations of :math:`X^{(k)}`. In other words, it is the
entropy of the time series conditioned on the :math:`k`-histories. The local
entropy rate

.. math::

    h_{X,i}(k,b) = \\log_b \\frac{p(x^{(k)}_i, x_{i+1})}{p(x^{(k)}_i)}

can be averaged to obtain the global entropy rate

.. math::

    H_X(k,b) = \\langle h_{X,i}(k,b) \\rangle_{i}
             = \\sum_{x^{(k)}_i,\\, x_{i+1}} p(x^{(k)}_i, x_{i+1}) \\log_b \\frac{p(x^{(k)}_i, x_{i+1})}{p(x^{(k)}_i)}.

Much as with :ref:`active-information`, the local and average entropy rates are
formally obtained in the limit

.. math::

    h_{X,i}(b) = \\lim_{k \\rightarrow \infty} h_{X,i}(k,b)
    \\quad \\textrm{and} \\quad
    H_X(b) = \\lim_{k \\rightarrow \infty} H_X(k,b),

but we do not provide limiting functionality in this library (yet!).

See [Cover1991]_ for more details.

.. _Entropy rate: https://en.wikipedia.org/wiki/Entropy_rate

Examples
--------

A Single Initial Condition
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's apply the entropy rate to a single initial condition. Typically, you will
just provide the time series and the history length, and let
:py:func:`.entropy_rate` take care of the rest: ::

    >>> entropy_rate([0,0,1,1,1,1,0,0,0], k=2)
    0.6792696431662095
    >>> entropy_rate([0,0,1,1,1,1,0,0,0], k=2, local=True)
    array([[ 1.       ,  0.       ,  0.5849625,  0.5849625,  1.5849625,
             0.       ,  1.       ]])
             
As with all of the time series measures, you can override the default base. ::

    >>> entropy_rate([0,0,1,1,1,1,2,2,2], k=2)
    0.24830578469386944
    >>> entropy_rate([0,0,1,1,1,1,2,2,2], k=2, b=4)
    0.19677767872596208
    >>> entropy_rate([0,0,1,1,1,1,2,2,2], k=2, b=2)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/home/ubuntu/workspace/pyinform/entropyrate.py", line 79, in entropy_rate
        error_guard(e)
      File "/home/ubuntu/workspace/pyinform/error.py", line 57, in error_guard
        raise InformError(e,func)
    pyinform.error.InformError: an inform error occurred - "unexpected state in timeseries"

Multiple Initial Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Of course multiple initial conditions are handled. ::

    >>> series = [[0,0,1,1,1,1,0,0,0], [1,0,0,1,0,0,1,0,0]]
    >>> entropy_rate(series, k=2)
    0.6253491072973907
    >>> entropy_rate(series, k=2, local=True)
    array([[ 0.4150375,  1.5849625,  0.5849625,  0.5849625,  1.5849625,
             0.       ,  2.       ],
           [ 0.       ,  0.4150375,  0.5849625,  0.       ,  0.4150375,
             0.5849625,  0.       ]])
"""

import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def entropy_rate(series, k, b=0, local=False):
    """
    Compute the average or local entropy rate of a time series with history
    length *k*.
    
    If the base *b* is not specified (or is 0), then it is inferred from the
    time series (with 2) as a minimum. *b* must be at least the base of the time
    series and is used a the base of the logarithm.

    :param series: the time series
    :type series: sequence or ``numpy.ndarray``
    :param int k: the history length
    :param int b: the base of the time series and logarithm
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
        er = np.empty((n,q), dtype=np.float64)
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
