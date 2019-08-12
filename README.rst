========
PyInform
========

PyInform is a python wrapper for the C `inform <https://github.com/elife-asu/inform>`_ library.

+--------------+
| Build Status |
+==============+
| |TravisCI|_  |
| |Appveyor|_  |
| |Codecov|_   |
+--------------+

.. |TravisCI| image:: https://travis-ci.org/elife-asu/pyinform.svg?branch=master
.. _TravisCI: https://travis-ci.org/elife-asu/pyinform

.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/txd9atm8m852b8ns/branch/master?svg=true
.. _Appveyor: https://ci.appveyor.com/project/dglmoore/pyinform-o2fv2/branch/master

.. |Codecov| image:: https://codecov.io/gh/elife-asu/pyinform/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/elife-asu/pyinform

-------------------------
Building and Installation
-------------------------

Via Pip
^^^^^^^

To install via :code:`pip`, you can run the following ::

    $ pip install pyinform
    
Note that on some systems this will require administrative privileges. If you don't have admin privileges or would prefer to install PyInform for your user only, you do so via the :code:`--user` flag: ::

    $ pip install --user pyinform

That's it! You're ready to go.

Manually
^^^^^^^^

If you plan on contributing to PyInform, you likely want to build the project manually. Once you have clone the repository, you will need to manually retrieve the most recent version of the `inform binaries <https://github.com/ELIFE-ASU/Inform/releases/download/v1.0.0/inform-1.0.0_mixed.zip>`_. The associated zip file should be extracted in the :code:`pyinform` subdirectory.

PyInform has one python dependency: `NumPy <http://www.numpy.org>`_. This should be easy to fulfill.

Once `NumPy <http://www.numpy.org>`_ been installed, you can test and install with :code:`setup.py`::

    $ python setup.py test
    $ python setup.py install --user

or using :code:`pip` on your local copy: ::

    $ python setup.py test
    $ pip install --user .

It may also be useful to install an "editable" version of PyInform. This means that any changes you make to your local copy will immediately be accessible in python. To do that just add the :code:`--editable` flag to :code:`pip`::

    $ pip install --editable --user .

And you are ready to go!

--------------------------
Building the Documentation
--------------------------

You can find live API documentation at https://elife-asu.github.io/PyInform. However, you might want to build the documentation yourself — in PDF format, for example. To build the documentation you will have to be able to run PyInform as the `sphinx <http://www.sphinx-doc.org/en/stable/>`_ documentation generator (also required) uses the docstrings in the source to build the documentation.

Once you have PyInform running and Sphinx installed you can build the HTML and PDF documentation via :code:`make`::

    $ make -C docs html
    $ make -C docs latexpdf
    
or via :code:`make.bat` on Windows::

    $ cd docs
    $ make.bat html
    $ make.bat latexpd

Of course, to build the PDF documentation you will need have LaTeX installed.

--------------
System Support
--------------

So far the python wrapper has been tested under :code:`python2.7`, :code:`python3.4` and :code:`python3.5`, and on the following platforms:

* Debian 8
* Mac OS X 10.11 (El Capitan)
* Windows 10

-------------
Grant Support
-------------
This project is supported in part by a grant provided by the Templeton World Charity Foundation as part of the `Power Of Information Initiative <http://www.templetonworldcharity.org/what-we-fund/themes-of-interest/power-of-information>`_.
