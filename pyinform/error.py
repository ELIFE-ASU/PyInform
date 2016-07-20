# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from pyinform import _inform
from ctypes import byref, c_char_p, c_int, POINTER

_strerror = _inform.inform_strerror
_strerror.argtypes = [POINTER(c_int)]
_strerror.restype = c_char_p

def error_string(e):
    """
    Generate an error message from an integral error code `e`.
    """
    if not isinstance(e, c_int):
        return error_string(c_int(e))
    else:
        return _strerror(byref(e)).decode("utf-8")
