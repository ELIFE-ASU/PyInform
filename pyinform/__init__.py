# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from ctypes import CDLL

def get_libpath():
    """
    Get the library path of the the distributed inform binary.
    """
    import os
    import re
    from os.path import dirname, abspath, realpath, join
    from platform import system

    libre = re.compile(r"^inform-(\d+)\.(\d+)\.(\d+)$")

    root = dirname(abspath(realpath(__file__)))

    libdir = None
    major, minor, revision = 0, 0, 0
    for _, dirnames, _ in os.walk(root):
        for dirname in dirnames:
            match = libre.match(dirname) 
            if match:
                a, b, c = tuple(int(x) for x in match.group(1,2,3))
                if (major, minor, revision) < (a,b,c):
                    major, minor, revision = a, b, c
                    libdir = join(root, match.group())
                    break
        break

    if libdir is None:
        raise ImportError("cannot find libinform")

    if system() is 'Linux':
        return "{}/lib/libinform.so.{}.{}.{}".format(libdir,major,minor,revision)
    elif system() is 'Windows':
        return "{}/lib/inform.dll".format(libdir)
    else:
        raise RuntimeError("unsupported platform {}".system())

_inform = CDLL(get_libpath())
