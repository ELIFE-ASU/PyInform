# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_double, c_int, c_ulong, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def series_range(series):
    """
    Compute the range of a continuously-valued timeseries.
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
    Bin a continuously-valued timeseries into `b` uniform bins
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

    binned = np.empty(xs.size, dtype=np.int32)
    out = binned.ctypes.data_as(POINTER(c_int))

    e = ErrorCode(0)
    if b is not None:
        spec = _inform_bin(data, c_ulong(xs.size), c_int(b), out, byref(e))
    elif step is not None:
        spec = step
        b = _inform_bin_step(data, c_ulong(xs.size), c_double(step), out, byref(e))
    elif bounds is not None:
        boundaries = np.ascontiguousarray(bounds, dtype=np.float64)
        if boundaries.ndim != 1:
            raise ValueError("boundaries array must be one-dimensional")
        bnds = boundaries.ctypes.data_as(POINTER(c_double))
        spec = bounds
        b = _inform_bin_bounds(data, c_ulong(xs.size), bnds, c_ulong(boundaries.size), out, byref(e))
    error_guard(e)

    binned = np.reshape(binned, xs.shape)

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
