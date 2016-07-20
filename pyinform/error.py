# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from pyinform import _inform
from ctypes import byref, c_bool, c_char_p, c_int, POINTER

_strerror = _inform.inform_strerror
_strerror.argtypes = [POINTER(c_int)]
_strerror.restype = c_char_p

_is_success = _inform.inform_succeeded
_is_success.argtypes = [POINTER(c_int)]
_is_success.restype = c_bool

_is_failure = _inform.inform_failed
_is_failure.argtypes = [POINTER(c_int)]
_is_failure.restype = c_bool

def error_string(e):
    """
    Generate an error message from an integral error code `e`.
    """
    if not isinstance(e, c_int):
        return error_string(c_int(e))
    else:
        return _strerror(byref(e)).decode("utf-8")

def is_success(e):
    """
    Determine if an error code represents a success.
    """
    if not isinstance(e, c_int):
        return is_success(c_int(e))
    else:
        return _is_success(byref(e))

def is_failure(e):
    """
    Determine if an error code represents a failure.
    """
    if not isinstance(e, c_int):
        return is_failure(c_int(e))
    else:
        return _is_failure(byref(e))
