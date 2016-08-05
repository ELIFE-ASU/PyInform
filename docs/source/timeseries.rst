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

    .. autofunction:: pyinform.conditionalentropy.conditional_entropy

.. _entropy-rate:

Entropy Rate
------------
.. automodule:: pyinform.entropyrate

    .. autofunction:: pyinform.entropyrate.entropy_rate

.. _mutual-information:

Mutual Information
------------------
.. automodule:: pyinform.mutualinfo

    .. autofunction:: pyinform.mutualinfo.mutual_info

.. _relative-entropy:

Relative Entropy
----------------
.. automodule:: pyinform.relativeentropy

    .. autofunction:: pyinform.relativeentropy.relative_entropy

.. _transfer-entropy:

Transfer Entropy
----------------
.. automodule:: pyinform.transferentropy

    .. autofunction:: pyinform.transferentropy.transfer_entropy
