# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
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
        for dir in dirnames:
            match = libre.match(dir)
            if match:
                a, b, c = tuple(int(x) for x in match.group(1, 2, 3))
                if (major, minor, revision) < (a, b, c):
                    major, minor, revision = a, b, c
                    libdir = join(root, match.group())
        break

    if libdir is None:
        raise ImportError("cannot find libinform")

    if system() == 'Linux':
        platform = "linux-x86_64"
        library = "libinform.so.{}.{}.{}".format(major, minor, revision)
    elif system() == 'Darwin':
        platform = "macosx-x86_64"
        library = "libinform.{}.{}.{}.dylib".format(major, minor, revision)
    elif system() == 'Windows':
        platform = "win-amd64"
        library = "inform.dll"
    else:
        raise RuntimeError("unsupported platform - \"{}\"".format(system()))

    return os.path.join(libdir, "lib", platform, library)


_inform = CDLL(get_libpath())

from . import utils                                  # noqa: F401
from . import shannon                                # noqa: F401
from .transferentropy import transfer_entropy        # noqa: F401
from .relativeentropy import relative_entropy        # noqa: F401
from .mutualinfo import mutual_info                  # noqa: F401
from .error import InformError                       # noqa: F401
from .entropyrate import entropy_rate                # noqa: F401
from .dist import Dist                               # noqa: F401
from .conditionalentropy import conditional_entropy  # noqa: F401
from .blockentropy import block_entropy              # noqa: F401
from .activeinfo import active_info                  # noqa: F401
