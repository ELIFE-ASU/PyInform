# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
State encoding is a necessity when complex time series are being analyzed. For
example, :math:`k`-history must be encoded as an integer in order to "observe"
it using a :py:class:`~.dist.Dist`. What if you are interested in correlating
the aggragate state of one group of nodes with that of another? You'd need to
encode the groups' states as integers. This module
(:py:mod:`pyinform.utils.encoding`)provides such functionality.

.. attention::

    As a practical matter, these utility functions should only be used as a
    stop-gap while a solution for your problem is implemented in the core
    `Inform <https://github.com/elife-asu/inform>`_ library. "Why?" you ask?
    Well, these functions are about as efficient as they can be for one-off
    state encodings, but most of the time you are interested in encoding
    sequences of states. This can be done much more efficiently if you encode
    the entire sequence at once. You need domain-specific information to make
    that happen.

    This being said, these functions aren't bad just be aware that they may turn
    into a bottleneck in whatever you are implementing.

"""
import numpy as np

from ctypes import byref, c_int, c_ulong, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard


def encode(state, b=None):
    """
    Encode a base-*b* array of integers into a single integer.

    This function uses a `big-endian`__ encoding scheme. That is, the most
    significant bits of the encoded integer are determined by the left-most
    end of the unencoded state.

    .. doctest:: utils

        >>> utils.encode([0,0,1], b=2)
        1
        >>> utils.encode([0,1,0], b=3)
        3
        >>> utils.encode([1,0,0], b=4)
        16
        >>> utils.encode([1,0,4], b=5)
        29

    If *b* is not provided (or is None), the base is inferred from the state
    with a minimum value of 2.

    .. doctest:: utils

        >>> utils.encode([0,0,2])
        2
        >>> utils.encode([0,2,0])
        6
        >>> utils.encode([1,2,1])
        16

    See also :py:func:`.decode`.

    .. __: https://en.wikipedia.org/wiki/Endianness#Examples

    :param sequence state: the state to encode
    :param int b: the base in which to encode
    :return: the encoded state
    :rtype: int
    :raises ValueError: if the state is empty
    :raises InformError: if an error occurs in the ``inform`` C call
    """
    xs = np.ascontiguousarray(state, dtype=np.int32)
    data = xs.ctypes.data_as(POINTER(c_int))

    if xs.size == 0:
        raise ValueError("cannot encode an empty array")

    if b is None:
        b = max(2, np.amax(xs) + 1)

    e = ErrorCode(0)
    encoding = _inform_encode(data, c_ulong(xs.size), c_int(b), byref(e))
    error_guard(e)

    return encoding


def decode(encoding, b, n=None):
    """
    Decode an integer into a base-*b* array with *n* digits.

    The provided encoded state is decoded using the `big-endian`__ encoding
    scheme.

    .. doctest:: utils

        >>> utils.decode(2, b=2, n=2)
        array([1, 0], dtype=int32)
        >>> utils.decode(6, b=2, n=3)
        array([1, 1, 0], dtype=int32)
        >>> utils.decode(6, b=3, n=2)
        array([2, 0], dtype=int32)

    Note that the base *b* must be provided, but the number of digits *n* is
    optional. If it is provided then the decoded state will have exactly that
    many elements.

    .. doctest:: utils

        >>> utils.decode(2, b=2, n=4)
        array([0, 0, 1, 0], dtype=int32)

    However, if *n* is too small to contain a full representation of the state,
    an error will be raised.

    .. doctest:: utils

        >>> utils.decode(6, b=2, n=2)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/home/ubuntu/workspace/pyinform/utils/encoding.py", line 126, in decode
            error_guard(e)
          File "/home/ubuntu/workspace/pyinform/error.py", line 57, in error_guard
            raise InformError(e,func)
        pyinform.error.InformError: an inform error occurred - "encoding/decoding failed"

    If *n* is not provided, the length of the decoded state is as small as
    possible:

    .. doctest:: utils

        >>> utils.decode(1, b=2)
        array([1], dtype=int32)
        >>> utils.decode(1, b=3)
        array([1], dtype=int32)
        >>> utils.decode(3, b=2)
        array([1, 1], dtype=int32)
        >>> utils.decode(3, b=3)
        array([1, 0], dtype=int32)
        >>> utils.decode(3, b=4)
        array([3], dtype=int32)

    Of course :py:func:`.encode` and :py:func:`.decode` play well together.

    .. doctest:: utils

        >>> for i in range(100):
        ...     assert(utils.encode(utils.decode(i, b=2)) == i)
        ...
        >>>

    See also :py:func:`.encode`.

    .. __: https://en.wikipedia.org/wiki/Endianness#Examples

    :param int encoding: the encoded state
    :param int b: the desired base
    :param int n: the desired number of digits
    :return: the decoded state
    :rtype: ``numpy.ndarray``
    :raises InformError: if *n* is too small to contain the decoding
    :raises InformError: if an error occurs within the ``inform`` C call
    """
    if n is None:
        state = np.empty(32, dtype=np.int32)
    else:
        state = np.empty(n, dtype=np.int32)
    out = state.ctypes.data_as(POINTER(c_int))

    e = ErrorCode(0)
    _inform_decode(c_int(encoding), c_int(b), out,
                   c_ulong(state.size), byref(e))
    error_guard(e)

    if n is None:
        for i in range(32):
            if state[i] != 0:
                break
        state = state[i:]

    return state


_inform_encode = _inform.inform_encode
_inform_encode.argtypes = [POINTER(c_int), c_ulong, c_int, POINTER(c_int)]
_inform_encode.restype = c_int

_inform_decode = _inform.inform_decode
_inform_decode.argtypes = [c_int, c_int,
                           POINTER(c_int), c_ulong, POINTER(c_int)]
_inform_decode.restype = None
