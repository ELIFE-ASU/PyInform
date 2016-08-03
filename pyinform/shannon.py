# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import c_double, c_void_p
from pyinform import _inform
from pyinform.dist import Dist

def entropy(dist, b=2.0):
    """
    Compute the base-*b* shannon entropy of the distribution *dist*.

    Examples: ::
    
        d = Dist([1,1,1,1])
        assert(entropy(d) == 2.)
        assert(entropy(d,4.) == 1.)

    :param dist: the distribution
    :type dist: :py:class:`pyinform.dist.Dist`
    :param float b: the logarithmic base
    :return: the shannon entropy of the distribution
    :rtype: float
    """
    return _entropy(dist._dist, c_double(b))

def mutual_info(joint, marginal_x, marginal_y, b=2.0):
    """
    Compute the base-*b* mututal information between two random variables.

    Example: ::

        joint = Dist([10,70,15,5])
        x = Dist([80,20])
        y = Dist([25,75])
        mutual_info(joint, x, y) # 0.214171

    :param joint: the joint distribution
    :type joint: :py:class:`pyinform.dist.Dist`
    :param marginal_x: the *x*-marginal distribution
    :type marginal_x: :py:class:`pyinform.dist.Dist`
    :param marginal_y: the *y*-marginal distribution
    :type marginal_y: :py:class:`pyinform.dist.Dist`
    :param float b: the logarithmic base
    :return: the mutual information
    :rtype: float
    """
    return _mutual_info(joint._dist, marginal_x._dist, marginal_y._dist, c_double(b))

def conditional_entropy(joint, marginal, b=2.0):
    """
    Compute the base-*b* conditional entropy given joint and marginal
    distributions.

    Example: ::

        joint = Dist([10,70,15,5])
        x = Dist([80,20])
        y = Dist([25,75])
        conditional_entropy(joint, x) # 0.597107
        conditional_entropy(joint, y) # 0.507757

    :param joint: the joint distribution
    :type joint: :py:class:`pyinform.dist.Dist`
    :param marginal: the marginal distribution
    :type marginal: :py:class:`pyinform.dist.Dist`
    :param float b: the logarithmic base
    :return: the conditional entropy
    :rtype: float
    """
    return _conditional_entropy(joint._dist, marginal._dist, c_double(b))

def conditional_mutual_info(joint, marginal_xz, marginal_yz, marginal_z, b=2.0):
    """
    Compute the base-*b* conditional mutual information given joint and
    marginal distributions.

    :param joint: the joint distribution
    :type joint: :py:class:`pyinform.dist.Dist`
    :param marginal_xz: the *x,z*-marginal distribution
    :type marginal_xz: :py:class:`pyinform.dist.Dist`
    :param marginal_yz: the *y,z*-marginal distribution
    :type marginal_yz: :py:class:`pyinform.dist.Dist`
    :param marginal_z: the *z*-marginal distribution
    :type marginal_z: :py:class:`pyinform.dist.Dist`
    :param float b: the logarithmic base
    :return: the conditional mutual information
    :rtype: float
    """
    return _conditional_mutual_info(joint._dist, marginal_xz._dist,
        marginal_yz._dist, marginal_z._dist, c_double(b))

def relative_entropy(p, q, b=2.0):
    """
    Compute the base-*b* relative entropy between posterior (*p*) and prior
    (*q*) distributions.

    :param p: the *posterior* distribution
    :type p: :py:class:`pyinform.dist.Dist`
    :param q: the *prior* distribution
    :type q: :py:class:`pyinform.dist.Dist`
    :param float b: the logarithmic base
    :return: the relative entropy
    :rtype: float
    """
    return _relative_entropy(p._dist, q._dist, c_double(b))

_entropy = _inform.inform_shannon
_entropy.argtypes = [c_void_p, c_double]
_entropy.restype = c_double

_mutual_info = _inform.inform_shannon_mi
_mutual_info.argtypes = [c_void_p, c_void_p, c_void_p, c_double]
_mutual_info.restype = c_double

_conditional_entropy = _inform.inform_shannon_ce
_conditional_entropy.argtypes = [c_void_p, c_void_p, c_double]
_conditional_entropy.restype = c_double

_conditional_mutual_info = _inform.inform_shannon_cmi
_conditional_mutual_info.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_double]
_conditional_mutual_info.restype = c_double

_relative_entropy = _inform.inform_shannon_re
_relative_entropy.argtypes = [c_void_p, c_void_p, c_double]
_relative_entropy.restype = c_double