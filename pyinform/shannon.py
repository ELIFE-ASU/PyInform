# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import c_double, c_void_p
from pyinform import _inform
from pyinform.dist import Dist

def entropy(dist, b=2.0):
    """
    Compute the base-`b` shannon entropy of a distribution `dist`.
    """
    return _entropy(dist._dist, c_double(b))

def mutual_info(joint, marginal_x, marginal_y, b=2.0):
    """
    Compute the base-`b` mututal information between two random variables
    """
    return _mutual_info(joint._dist, marginal_x._dist, marginal_y._dist, c_double(b))

_entropy = _inform.inform_shannon
_entropy.argtypes = [c_void_p, c_double]
_entropy.restype = c_double

_mutual_info = _inform.inform_shannon_mi
_mutual_info.argtypes = [c_void_p, c_void_p, c_void_p, c_double]
_mutual_info.restype = c_double