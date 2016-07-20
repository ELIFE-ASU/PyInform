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

setup(
    name='pyinform',
    version='0.0.2',
    description='A wrapper for the Inform library',
    long_description=readme,
    url='https://github.com/elife-asu/pyinform',
    license=license,
    requires=['numpy'],
    py_modules=['pyinform'],
    cmdclass={
        'test': TestCommand
    }
)
