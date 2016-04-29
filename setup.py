from distutils.cmd import Command
from distutils.core import setup
from distutils.extension import Extension
import numpy

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

        path = "build/lib.{0}-{1}".format(get_platform(), get_python_version())
        os.environ['PYTHONPATH'] = path
        raise SystemExit(subprocess.call([sys.executable, '-m', 'unittest', 'discover']))

extensions = []
try:
    from Cython.Build import cythonize
    extensions = cythonize([Extension("pyinform", ["pyinform/pyinform.pyx"],
                            libraries=["inform"],
                            include_dirs=[numpy.get_include()])])
except ImportError:
    extensions = [Extension("pyinform", ["pyinform/pyinform.c"],
                  libraries=["inform"],
                  include_dirs=[numpy.get_include()])]

setup(
    name='pyinform',
    version='0.0.1',
    description='A wrapper for the Inform library',
    long_description=readme,
    url='https://github.com/elife-asu/pyinform',
    license=license,
    requires=['numpy'],
    ext_modules=extensions,
    cmdclass={
        'test': TestCommand
    }
)
