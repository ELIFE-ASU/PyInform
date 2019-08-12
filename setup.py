# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from setuptools import setup

long_description="""
PyInform is a python wrapper for the C `inform <https://github.com/elife-asu/inform>`_ library. You can find live API documentation at https://elife-asu.github.io/PyInform. 

So far the python wrapper has been tested under :code:`python2.7`, :code:`python3.4` and :code:`python3.5`, and on the following platforms:

* Debian 8
* Mac OS X 10.11 (El Capitan)
* Windows 10
"""

with open('LICENSE') as f:
    license = f.read()

inform_version = "1.0.0"
inform_files = [
    "README.rst",
    "LICENSE",
    "inform-{}/lib/linux-x86_64/libinform.so.{}".format(inform_version, inform_version),
    "inform-{}/lib/macosx-x86_64/libinform.{}.dylib".format(inform_version, inform_version),
    "inform-{}/lib/win-amd64/inform.dll".format(inform_version),
]

setup(
    name='pyinform',
    version='0.1.0',
    description='A wrapper for the Inform library',
    long_description=long_description,
    maintainer='Douglas G. Moore',
    maintainer_email='doug@dglmoore.com',
    url='https://github.com/elife-asu/pyinform',
    license=license,
    install_requires=['numpy'],
    setup_requires=['green'],
    packages=['pyinform', 'pyinform.utils'],
    package_data={'pyinform': inform_files},
    test_suite="test",
    platforms=["Windows", "OS X", "Linux"]
)
