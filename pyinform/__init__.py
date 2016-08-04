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

    if libdir is None:
        raise ImportError("cannot find libinform")

    if system() == 'Linux':
        platform = "linux-x86_64"
        library = "libinform.so.{}.{}.{}".format(major,minor,revision)
    elif system() == 'Darwin':
        platform = "macosx-x86_64"
        library = "libinform.{}.{}.{}.dylib".format(major,minor,revision)
    elif system() == 'Windows':
        platform = "win-amd64"
        library = "inform.dll"
    else:
        raise RuntimeError("unsupported platform - \"{}\"".format(system()))

    return os.path.join(libdir, "lib", platform, library)

_inform = CDLL(get_libpath())

from .activeinfo import active_info
from .blockentropy import block_entropy
from .conditionalentropy import conditional_entropy
from .dist import Dist
from .entropyrate import entropy_rate
from .error import InformError
from .mutualinfo import mutual_info
from .relativeentropy import relative_entropy
from .transferentropy import transfer_entropy

from . import shannon
from . import utils