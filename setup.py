# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from distutils.cmd import Command
from distutils.core import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

class TestCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import os, sys, subprocess
        from sysconfig import get_python_version
        from distutils.util import get_platform

        raise SystemExit(subprocess.call([sys.executable, '-m', 'unittest', 'discover']))

inform_version = "0.0.3"
inform_files = ["inform-{}/*/*".format(inform_version)]

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
    cmdclass= { 'test': TestCommand }
)
