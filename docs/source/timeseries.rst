.. _timeseries:

Time Series Measures
====================

The :ref:`dist` and :ref:`shannon` come together to make information measures
on time series almost trivial to implement. Every such measure amounts to
constructing distributions and applying an information measure.

.. _notation:

Notation
--------

Throughout this section we will denote random variables as :math:`X, Y, \ldots`,
and let :math:`x_i, y_i, \ldots` represent the :math:`i`-th time step of a time
series drawn a random variable. Many of the measures consider
:math:`k`-histories (a.k.a :math:`k`-blocks) of the time series, e.g.
:math:`x^{(k)}_i = \{x_{i-k+1}, x_{i-k+2}, \ldots, x_i\}`.

For the sake of conciseness, when denoting probability distributions, we will
only make the random variable explicit in situations where the notation is
ambiguous. Generally, we will write :math:`p(x_i)`, :math:`p(x^{(k)}_i)` and
:math:`p(x^{(k)}_i, x_{i+1})` to denote the empirical probability of obseriving
the :math:`x_i` state, the :math:`x^{(k)}_i` :math:`k`-history, and the joint
probability of observing :math:`(x^{(k)}_i, x_{i+1})`.

**Please report any notational ambiguities as an**
`issue <https://github.com/ELIFE-ASU/PyInform/issues>`_.


.. _subtle-details:

Subtle Details
--------------

The library take several liberties in the way in which the time series measures
are implemented.

The Base: States and Logarithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The word "base" has two different meanings in the context of the information
measures on time series. It could refer to the base of the time series itself,
that is the number of unique states in the time series. For example, the
time series :math:`\{0,2,1,0,0\}` has a base of 3. On the other handle
it could refer to the base of the logarithm used in computing the information
content of the emipirical distributions. The problem is that these two meanings
clash. The base of the time series affects the range of values the measure can
produce, and the base of the logarithm represents a rescaling of those values.

The following measures use one of two conventions. The measures of information
dynamics (e.g. :ref:`active-information`, :ref:`entropy-rate` and
:ref:`transfer-entropy`) take as an argument the **base of the state** and use
that as the logarithm of the logarithm. The result is that the time-averaged
values of those measures are in the unit range. An exception to this rule is
the block entropy. It two uses this convention, but its value will not be in the
unit range unless the block size :math:`k` is 1 or the specified base is
:math:`2^k` (or you could just divide by :math:`k`). The second convention is to
take both the base of the time series and the base of the logarithm. This is
about as unambiguous as it gets. This approach is used for the measures that do
not make explicit use of a history length (or block size), e.g.
:ref:`mutual-information`, :ref:`conditional-entropy`, etc...

Coming releases may revise the handling of the bases, but until then each
function's documentation will specify how the base is used.

Multiple Initial Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyInform tries to provide handling of multiple initial conditions. The "proper"
way to handle initial conditions is a bit contested. One completely reasonable
approach is to apply the information measures to each initial condition's time
series independently and then average. One can think of this approach as
conditioning the measure on the inital condition. The second approach is to
independently use all of the initial conditions to construct the various
probability distributions. You can think of this approach as rolling the
uncertainty of the initial condition into the measure. [#Averaging]_

The current implementation takes the second approach. The accpeted time series
can be up to 2-D with each row representing the time series for a different
initial condition. We chose to take the second approach because the "measure
then average" method can still be done with the current implimentation. For
an example of this, see the example section of :ref:`active-information`.

Subsequent releases may provide a mechanism for specifying a how the user
prefers the initial conditions to be handled, but at the moment the user has to
make it happen manually.

.. [#Averaging] There is actually at least three ways to handle multiple initial
    conditions, but the third method is related to the first described in the text
    by the addition of the entropy of the distribution over initial conditions. In
    this approach, the initial condition is considered as a random variable.


.. _active-information:

Active Information
------------------
.. automodule:: pyinform.activeinfo

    API Documentation
    -----------------

    .. autofunction:: pyinform.activeinfo.active_info

.. _block-entropy:

Block Entropy
-------------
.. automodule:: pyinform.blockentropy

    API Documentation
    -----------------

    .. autofunction:: pyinform.blockentropy.block_entropy

.. _conditional-entropy:

Conditional Entropy
-------------------
.. automodule:: pyinform.conditionalentropy

    API Documentation
    -----------------

    .. autofunction:: pyinform.conditionalentropy.conditional_entropy

.. _entropy-rate:

Entropy Rate
------------
.. automodule:: pyinform.entropyrate

    API Documentation
    -----------------

    .. autofunction:: pyinform.entropyrate.entropy_rate

.. _mutual-information:

Mutual Information
------------------
.. automodule:: pyinform.mutualinfo

    API Documentation
    -----------------

    .. autofunction:: pyinform.mutualinfo.mutual_info

.. _relative-entropy:

Relative Entropy
----------------
.. automodule:: pyinform.relativeentropy

    API Documentation
    -----------------

    .. autofunction:: pyinform.relativeentropy.relative_entropy

.. _transfer-entropy:

Transfer Entropy
----------------
.. automodule:: pyinform.transferentropy

    API Documentation
    -----------------

    .. autofunction:: pyinform.transferentropy.transfer_entropy

References
----------

.. [Cover1991] T.M. Cover amd J.A. Thomas (1991). "Elements of information theory" (1st ed.). New York: Wiley. ISBN 0-471-06259-6.

.. [Kraiser2002] A. Kaiser, T. Schreiber, "`Information transfer in continuous processes`__", Physica D: Nonlinear Phenomena, Volume 166, Issues 1–2, 1 June 2002, Pages 43-62, ISSN 0167-2789
.. __: http://dx.doi.org/10.1016/S0167-2789(02)00432-3

.. [Kullback1951] Kullback, S.; Leibler, R.A. (1951). "`On information and sufficiency`__". Annals of Mathematical Statistics. 22 (1): 79-86. doi:10.1214/aoms/1177729694. MR 39968.
.. __: http://projecteuclid.org/DPubS?service=UI&version=1.0&verb=Display&handle=euclid.aoms/1177729694

.. [Lizier2008] J.T. Lizier M. Prokopenko and A. Zomaya, "`Local information transfer as a spatiotemporal filter for complex systems`__", Phys. Rev. E 77, 026110, 2008.
.. __: http://dx.doi.org/10.1103/PhysRevE.77.026110

.. [Lizier2012] J.T. Lizier, M. Prokopenko and A.Y. Zomaya, "`Local measures of information storage in complex distributed computation`__" Information Sciences, vol. 208, pp. 39-54, 2012.
.. __: http://dx.doi.org/10.1016/j.ins.2012.04.016

.. [Schreiber2000] T. Schreiber, "`Measuring information transfer`__", Phys.Rev.Lett. 85 (2) pp.461-464, 2000.
.. __: http://dx.doi.org/10.1103/PhysRevLett.85.461

.. [Shannon1948] Shannon, Claude E. (July-October 1948). "`A Mathematical Theory of Communication`__". Bell System Technical Journal. 27 (3): 379-423. doi:10.1002/j.1538-7305.1948.tb01448.x.
.. __: https://dx.doi.org/10.1002%2Fj.1538-7305.1948.tb01338.x
