# PyInform

PyInform is a python wrapper for the C [inform](https://github.com/elife-asu/inform) library. At present it is woefully undertested, and it only provides active information.

## Building and Installation

To install the wrapper, you will first need to install [inform](https://github.com/elife-asu/inform). PyInform has three python dependencies:
- [Cython](http://cython.org)
- [DistUtils](https://docs.python.org/2/library/distutils.html)
- [NumPy](http://www.numpy.org)

Once these have been installed, you can build, test and install.

    $ python setup.py build
    $ python setup.py test
    $ python setup.py install --user

## System Support

So far the python wrapper has been tested under with both `python2.7` and `python3.4`, and on the following platforms:
- Debian 8
- Mac OS X 10.11 (El Capitan)

PyInform does not yet support Windows as the [inform](https://github.com/elife-asu/inform) library does not yet build on taht platform.
