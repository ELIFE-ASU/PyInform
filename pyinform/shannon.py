# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from ctypes import c_double, c_void_p
from pyinform import _inform


def entropy(p, b=2.0):
    """
    Compute the base-*b* shannon entropy of the distribution *p*.

    Taking :math:`X` to be a random variable with :math:`p_X` a probability
    distribution on :math:`X`, the base-:math:`b` Shannon entropy is defined
    as

    .. math::

        H(X) = -\\sum_{x} p_X(x) \\log_b p_X(x).

    .. rubric:: Examples:

    .. doctest:: shannon

        >>> d = Dist([1,1,1,1])
        >>> shannon.entropy(d)
        2.0
        >>> shannon.entropy(d, 4)
        1.0

    .. doctest:: shannon

        >>> d = Dist([2,1])
        >>> shannon.entropy(d)
        0.9182958340544896
        >>> shannon.entropy(d, b=3)
        0.579380164285695

    See [Shannon1948a]_ for more details.

    :param p: the distribution
    :type p: :py:class:`pyinform.dist.Dist`
    :param float b: the logarithmic base
    :return: the shannon entropy of the distribution
    :rtype: float
    """
    return _entropy(p._dist, c_double(b))


def mutual_info(p_xy, p_x, p_y, b=2.0):
    """
    Compute the base-*b* mututal information between two random variables.

    `Mutual information`_ provides a measure of the mutual dependence between
    two random variables. Let :math:`X` and :math:`Y` be random variables with
    probability distributions :math:`p_X` and :math:`p_Y` respectively, and
    :math:`p_{X,Y}` the joint probability distribution over :math:`(X,Y)`. The
    base-:math:`b` mutual information between :math:`X` and :math:`Y` is
    defined as

    .. math::

        I(X;Y) &= \\sum_{x,y} p_{X,Y}(x,y) \\log_b \\frac{p_{X,Y}(x,y)}{p_X(x)p_Y(y)}\\\\
               &= H(X) + H(Y) - H(X,Y).

    Here the second line takes advantage of the properties of logarithms and
    the definition of Shannon entropy, :py:func:`.entropy`.

    To some degree one can think of mutual information as a measure of the
    (linear and non-linear) coorelations between random variables.

    See [Cover1991a]_ for more details.

    .. rubric:: Examples:

    .. doctest:: shannon

        >>> xy = Dist([10,70,15,5])
        >>> x = Dist([80,20])
        >>> y = Dist([25,75])
        >>> shannon.mutual_info(xy, x, y)
        0.21417094500762912

    :param p_xy: the joint distribution
    :type p_xy: :py:class:`pyinform.dist.Dist`
    :param p_x: the *x*-marginal distribution
    :type p_x: :py:class:`pyinform.dist.Dist`
    :param p_y: the *y*-marginal distribution
    :type p_y: :py:class:`pyinform.dist.Dist`
    :param float b: the logarithmic base
    :return: the mutual information
    :rtype: float

    .. _Mutual Information: https://en.wikipedia.org/wiki/Mutual_information
    """
    return _mutual_info(p_xy._dist, p_x._dist, p_y._dist, c_double(b))


def conditional_entropy(p_xy, p_y, b=2.0):
    """
    Compute the base-*b* conditional entropy given joint (*p_xy*) and marginal
    (*p_y*) distributions.

    `Conditional entropy`_ quantifies the amount of information required to
    describe a random variable :math:`X` given knowledge of a random variable
    :math:`Y`. With :math:`p_Y` the probability distribution of :math:`Y`, and
    :math:`p_{X,Y}` the distribution for the joint distribution :math:`(X,Y)`,
    the base-:math:`b` conditional entropy is defined as

    .. math::

        H(X|Y) &= -\\sum_{x,y} p_{X,Y}(x,y) \\log_b \\frac{p_{X,Y}(x,y)}{p_Y(y)}\\\\
               &= H(X,Y) - H(Y).

    See [Cover1991a]_ for more details.

    .. rubric:: Examples:

    .. doctest:: shannon

        >>> xy = Dist([10,70,15,5])
        >>> x = Dist([80,20])
        >>> y = Dist([25,75])
        >>> shannon.conditional_entropy(xy, x)
        0.5971071794515037
        >>> shannon.conditional_entropy(xy, y)
        0.5077571498797332

    :param p_xy: the joint distribution
    :type p_xy: :py:class:`pyinform.dist.Dist`
    :param p_y: the marginal distribution
    :type p_y: :py:class:`pyinform.dist.Dist`
    :param float b: the logarithmic base
    :return: the conditional entropy
    :rtype: float

    .. _Conditional entropy: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return _conditional_entropy(p_xy._dist, p_y._dist, c_double(b))


def conditional_mutual_info(p_xyz, p_xz, p_yz, p_z, b=2.0):
    """
    Compute the base-*b* conditional mutual information the given joint
    (*p_xyz*) and marginal (*p_xz*, *p_yz*, *p_z*) distributions.

    `Conditional mutual information`_ was introduced by [Dobrushin1959]_ and
    [Wyner1978]_, and more or less quantifies the average mutual information
    between random variables :math:`X` and :math:`Y` given knowledge of a third
    :math:`Z`. Following the same notations as in
    :py:func:`.conditional_entropy`, the base-:math:`b` conditional mutual
    information is defined as

    .. math::

        I(X;Y|Z) &= -\\sum_{x,y,z} p_{X,Y,Z}(x,y,z) \\log_b \\frac{p_{X,Y|Z}(x,y|z)}{p_{X|Z}(x|z)p_{Y|Z}(y|z)}\\\\
                 &= -\\sum_{x,y,z} p_{X,Y,Z}(x,y,z) \\log_b \\frac{p_{X,Y,Z}(x,y,z)p_{Z}(z)}{p_{X,Z}(x,z)p_{Y,Z}(y,z)}\\\\
                 &= H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)


    .. _Conditional mutual information: https://en.wikipedia.org/wiki/Conditional_mutual_information

    .. rubric:: Examples:

    .. doctest:: shannon

        >>> xyz = Dist([24,24,9,6,25,15,10,5])
        >>> xz = Dist([15,9,5,10])
        >>> yz = Dist([9,15,10,15])
        >>> z = Dist([3,5])
        >>> shannon.conditional_mutual_info(xyz, xz, yz, z)
        0.12594942727460334

    :param p_xyz: the joint distribution
    :type p_xyz: :py:class:`pyinform.dist.Dist`
    :param p_xz: the *x,z*-marginal distribution
    :type p_xz: :py:class:`pyinform.dist.Dist`
    :param p_yz: the *y,z*-marginal distribution
    :type p_yz: :py:class:`pyinform.dist.Dist`
    :param p_z: the *z*-marginal distribution
    :type p_z: :py:class:`pyinform.dist.Dist`
    :param float b: the logarithmic base
    :return: the conditional mutual information
    :rtype: float
    """
    return _conditional_mutual_info(p_xyz._dist, p_xz._dist, p_yz._dist,
                                    p_z._dist, c_double(b))


def relative_entropy(p, q, b=2.0):
    """
    Compute the base-*b* relative entropy between posterior (*p*) and prior
    (*q*) distributions.

    `Relative entropy`, also known as the Kullback-Leibler divergence, was
    introduced by Kullback and Leiber in 1951 ([Kullback1951a]_). Given a random
    variable :math:`X`, two probability distributions :math:`p_X` and
    :math:`q_X`, relative entropy measures the information gained in switching
    from the prior :math:`q_X` to the posterior :math:`p_X`:

    .. math::

        D_{KL}(p_X || q_X) = \\sum_x p_X(x) \\log_b \\frac{p_X(x)}{q_X(x)}.

    Many of the information measures, e.g. :py:func:`.mutual_info`,
    :py:func:`.conditional_entropy`, etc..., amount to applications of relative
    entropy for various prior and posterior distributions.

    .. rubric:: Examples:

    .. doctest:: shannon

        >>> p = Dist([4,1])
        >>> q = Dist([1,1])
        >>> shannon.relative_entropy(p,q)
        0.27807190511263774
        >>> shannon.relative_entropy(q,p)
        0.3219280948873624

    .. doctest:: shannon

        >>> p = Dist([1,0])
        >>> q = Dist([1,1])
        >>> shannon.relative_entropy(p,q)
        1.0
        >>> shannon.relative_entropy(q,p)
        nan

    :param p: the *posterior* distribution
    :type p: :py:class:`pyinform.dist.Dist`
    :param q: the *prior* distribution
    :type q: :py:class:`pyinform.dist.Dist`
    :param float b: the logarithmic base
    :return: the relative entropy
    :rtype: float
    """
    return _relative_entropy(p._dist, q._dist, c_double(b))


_entropy = _inform.inform_shannon_entropy
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
