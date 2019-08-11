PyInform
========

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

PyInform is a python library of information-theoretic measures for time series
data. PyInform is backed by the `Inform <https://github.com/elife-asu/inform>`_
C library.

The library is built out of three primary components.

1. The :py:class:`pyinform.dist.Dist` class provides discrete, emperical
probability distributions. These form the basis for all of the
information-theoretic measures.

2. A collection of information measures built upon the distribution class
provide the core algorithms for the library and are implemented in the
:py:mod:`pyinform.shannon` submodule.

3. A host of measures of information dynamics on time series are built upon the
core information measures. Each measure is housed in its own submodule, but are
exposed for convenience by the root packge, :py:mod:`pyinform`.

In addition to the core components, a small collection of utilities are
provided by the :py:mod:`pyinform` module.

.. toctree::
   :maxdepth: 2

   starting
   dist
   shannon
   timeseries
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

