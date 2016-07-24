# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from setuptools import setup
from platform import system

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

inform_version = "0.0.3"
inform_files = [
    "inform-{}/lib/libinform.so.{}".format(inform_version, inform_version),
	"inform-{}/lib/inform.dll".format(inform_version)
]

setup(
    name='pyinform',
    version='0.0.2',
    description='A wrapper for the Inform library',
    long_description=readme,
    maintainer='Douglas G. Moore',
    maintainer_email='douglas.g.moore@asu.edu',
    url='https://github.com/elife-asu/pyinform',
    license=license,
    requires=['numpy'],
    packages=['pyinform'],
    package_data = { 'pyinform' : inform_files },
    test_suite = "test",
    platforms = ["Windows", "Linux"],
)
