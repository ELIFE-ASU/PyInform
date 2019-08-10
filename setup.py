# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from setuptools import setup
from platform import system

with open('README.rst') as f:
    readme = f.read()

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
    version='0.0.6',
    description='A wrapper for the Inform library',
    long_description=readme,
    maintainer='Douglas G. Moore',
    maintainer_email='douglas.g.moore@asu.edu',
    url='https://github.com/elife-asu/pyinform',
    license=license,
    requires=['numpy'],
    packages=['pyinform', 'pyinform.utils'],
    package_data = { 'pyinform' : inform_files },
    test_suite = "test",
    platforms = ["Windows", "OS X", "Linux"],
)
