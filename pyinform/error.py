# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from ctypes import byref, c_bool, c_char_p, c_int, POINTER
from pyinform import _inform

ErrorCode = c_int


def error_string(e):
    """
    Generate an error message from an integral error code `e`.
    """
    if not isinstance(e, ErrorCode):
        return error_string(ErrorCode(e))
    else:
        return _strerror(byref(e)).decode("utf-8")


def is_success(e):
    """
    Determine if an error code represents a success.
    """
    if not isinstance(e, ErrorCode):
        return is_success(ErrorCode(e))
    else:
        return _is_success(byref(e))


def is_failure(e):
    """
    Determine if an error code represents a failure.
    """
    if not isinstance(e, ErrorCode):
        return is_failure(ErrorCode(e))
    else:
        return _is_failure(byref(e))


class InformError(Exception):
    """
    InformError signifies an error occurred in a call to inform.
    """

    def __init__(self, e=-1, func=None):
        msg = error_string(e)

        if func is None:
            msg = "an inform error occurred - \"{}\"".format(msg)
        else:
            msg = "an inform error occurred in `{}` - \"{}\"".format(func, msg)

        super(InformError, self).__init__(msg)

        self.error_code = e if isinstance(e, ErrorCode) else ErrorCode(e)


def error_guard(e, func=None):
    """
    Raise an appropriately formated error if `e` is a failure
    """
    if is_failure(e):
        raise InformError(e, func)


_strerror = _inform.inform_strerror
_strerror.argtypes = [POINTER(c_int)]
_strerror.restype = c_char_p

_is_success = _inform.inform_succeeded
_is_success.argtypes = [POINTER(c_int)]
_is_success.restype = c_bool

_is_failure = _inform.inform_failed
_is_failure.argtypes = [POINTER(c_int)]
_is_failure.restype = c_bool
