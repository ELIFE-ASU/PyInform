========
PyInform
========

PyInform is a python wrapper for the C `inform <https://github.com/elife-asu/inform>`_ library.

-------------------------
Building and Installation
-------------------------

To install the wrapper, you will first need to install `inform <https://github.com/elife-asu/inform`_. PyInform has one python dependency: `NumPy <http://www.numpy.org>`_.

Once `NumPy <http://www.numpy.org>`_ been installed, you can test and install with `setup.py`::

    $ python setup.py test
    $ python setup.py install --user

or using `pip` on your local copy:::

    $ python setup.py test
    $ pip install --user .

--------------
System Support
--------------

So far the python wrapper has been tested under `python2.7`, `python3.4` and `pythonn3.5`, and on the following platforms:

* Debian 8
* Mac OS X 10.11 (El Capitan)
* Windows 10
