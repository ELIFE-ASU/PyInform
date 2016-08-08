# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
All of the currently implemented time series measures are only defined on
discretely-valued time series. However, in practice continuously-valued time
series are ubiquitous. There are two approaches to accomodating continuous
values.

The simplest is to *bin* the time series, forcing the values into discrete
states. This method has its downsides, namely that the binning is often a bit
unphysical and it can introduce bias. What's more, without some kind of guiding
principle it can be difficult to decide exactly which binning approach.

The second approach attempts to infer condinuous probability distributions from
continuous data. This is potentially more robust, but more technically
difficult. Unfortunately, PyInform does not yet have an implementation of
information measures on continous distributions.

This module (:py:mod:`pyinform.utils.binning`) provides a basic binning facility
via the :py:func:`.bin_series` function.
"""

import numpy as np

from ctypes import byref, c_double, c_int, c_ulong, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def series_range(series):
    """
    Compute the range of a continuously-valued time series.
    
    Examples: ::
    
        >>> from pyinform import utils
        >>> utils.series_range([0,1,2,3,4,5])
        (5, 0, 5)
        >>> utils.series_range([-0.1, 8.5, 0.02, -6.3])
        (14.8, -6.3, 8.5)

    :param sequence series: the time series
    :returns: the range and the minimum/maximum values
    :rtype: 3-tuple (float, float, float)
    :raises InformError: if an error occurs within the ``inform`` C call
    """
    xs = np.ascontiguousarray(series, dtype=np.float64)
    data = xs.ctypes.data_as(POINTER(c_double))

    min, max = c_double(), c_double()
    
    e = ErrorCode(0)
    rng = _inform_range(data, c_ulong(xs.size), byref(min), byref(max), byref(e))
    error_guard(e)
    
    return rng, min.value, max.value

def bin_series(series, b=None, step=None, bounds=None):
    """
    Bin a continously-valued times series.
    
    The binning can be performed in any one of three ways.
    
    .. rubric:: 1. Specified Number of Bins
    
    The first is binning the time series into *b* uniform bins (with *b* an
    integer). ::
    
        >>> from pyinform import utils
        >>> import numpy as np
        >>> xs = 10 * np.random.rand(20)
        >>> xs
        array([ 6.62004974,  7.24471972,  0.76670198,  2.66306833,  4.32200795,
                8.84902227,  6.83491844,  7.05008074,  3.79287646,  6.50844032,
                8.68804879,  6.79543773,  0.3222078 ,  7.39576325,  7.54150189,
                1.06422897,  1.91958431,  2.34760945,  3.90139184,  3.08885353])
        >>> utils.bin_series(xs, b=2)
        (array([1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], dtype=int32), 2, 4.263407236635026)
        >>> utils.bin_series(xs, b=3)
        (array([2, 2, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 0, 0, 0, 1, 0], dtype=int32), 3, 2.8422714910900173)

    
    With this approach the binned sequence (as an ``numpy.ndarray``), the number
    of bins, and the size of each bin are returned.
    
    This binning method is useful if, for example, the user wants to bin several
    time series to the same base.
    
    .. rubric:: 2. Fixed Size Bins
    
    The second type of binning produces bins of a specific size *step*.::
    
        >>> utils.bin_series(xs, step=4.0)
        (array([1, 1, 0, 0, 0, 2, 1, 1, 0, 1, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0], dtype=int32), 3, 4.0)
        >>> utils.bin_series(xs, step=2.0)
        (array([3, 3, 0, 1, 1, 4, 3, 3, 1, 3, 4, 3, 0, 3, 3, 0, 0, 1, 1, 1], dtype=int32), 5, 2.0)
        
    As in the previous case the binned sequence, the number of bins, and the
    size of each bin are returned.
    
    This approach is appropriate when the system at hand has a particular
    sensitivity or precision, e.g. if the system is sensitive down to 5.0mV
    changes in potential.
    
    .. rubric:: 3. Thresholds
    
    The third type of binning is breaks the real number line into segments with
    specified boundaries or thresholds, and the time series is binned according
    to this partitioning. The bounds are expected to be provided in ascending
    order.::
    
        >>> utils.bin_series(xs, bounds=[2.0, 7.5])
        (array([1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 2, 1, 0, 1, 2, 0, 0, 1, 1, 1], dtype=int32), 3, [2.0, 7.5])
        >>> utils.bin_series(xs, bounds=[2.0])
        (array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1], dtype=int32), 2, [2.0])

    Unlike the previous two types of binning, this approach returns the specific
    bounds rather than the bin sizes. The other two returns, the binned
    sequence and the number of bins, are returned as before.
    
    This approach is useful in situations where the system has natural
    thesholds, e.g. the polarized/hyperpolarized states of a neuron.

    :param sequence series: the continuously-valued time series
    :param int b: the desired number of uniform bins
    :param float step: the desired size of each uniform bin
    :param sequence bounds: the (finite) bounds of each bin
    :return: the binned sequence, the number of bins and either the bin sizes or bin bounds
    :rtype: either (``numpy.ndarray``, int, float) or (``numpy.ndarray``, int, sequence)
    :raises ValueError: if no keyword argument is provided
    :raises ValueError: if more than one keyword argument is provided
    :raises InformError: if an error occurs in the ``inform`` C call
    """
    if b is None and step is None and bounds is None:
        raise ValueError("must provide either number of bins, step size, or bin boundaries")
    elif b is not None and step is not None:
        raise ValueError("cannot provide both number of bins and step size")
    elif b is not None and bounds is not None:
        raise ValueError("cannot provide both number of bins and bin boundaries")
    elif step is not None and bounds is not None:
        raise ValueError("cannot provide both step size and bin boundaries")
    
    xs = np.ascontiguousarray(series, dtype=np.float64)
    data = xs.ctypes.data_as(POINTER(c_double))

    binned = np.empty(xs.shape, dtype=np.int32)
    out = binned.ctypes.data_as(POINTER(c_int))

    e = ErrorCode(0)
    if b is not None:
        spec = _inform_bin(data, c_ulong(xs.size), c_int(b), out, byref(e))
    elif step is not None:
        spec = step
        b = _inform_bin_step(data, c_ulong(xs.size), c_double(step), out, byref(e))
    elif bounds is not None:
        boundaries = np.ascontiguousarray(bounds, dtype=np.float64)
        bnds = boundaries.ctypes.data_as(POINTER(c_double))
        spec = bounds
        b = _inform_bin_bounds(data, c_ulong(xs.size), bnds, c_ulong(boundaries.size), out, byref(e))
    error_guard(e)

    return binned, b, spec

_inform_range = _inform.inform_range
_inform_range.argtypes = [POINTER(c_double), c_ulong, POINTER(c_double), POINTER(c_double), POINTER(c_int)]
_inform_range.restype = c_double

_inform_bin = _inform.inform_bin
_inform_bin.argtypes = [POINTER(c_double), c_ulong, c_int, POINTER(c_int), POINTER(c_int)]
_inform_bin.restype = c_double

_inform_bin_step = _inform.inform_bin_step
_inform_bin_step.argtypes = [POINTER(c_double), c_ulong, c_double, POINTER(c_int), POINTER(c_int)]
_inform_bin_step.restype = c_int

_inform_bin_bounds = _inform.inform_bin_bounds
_inform_bin_bounds.argtypes = [POINTER(c_double), c_ulong, POINTER(c_double), c_ulong, POINTER(c_int), POINTER(c_int)]
_inform_bin_bounds.restype = c_int
