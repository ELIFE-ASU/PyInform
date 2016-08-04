Timeseries Measures
===================

The :ref:`dist` and :ref:`shannon` come together to make information measures
on time series almost trivial to implement. Every such measure amounts to
constructing distributions and applying an information measure.

Throughout this section we will denote random variables as :math:`X, Y, \ldots`,
and let :math:`x_i, y_i, \ldots` represent the :math:`i`-th time step of a time
series drawn a random variable. Many of the measures consider
:math:`k`-histories (a.k.a :math:`k`-blocks) of the time series, e.g.
:math:`x^{(k)}_i = \{x_{i-k+1}, x_{i-k+2}, \ldots, x_i\}`.

For the sake of conciseness, when denoting probability distributions, we will
only make the random variable explicit in situations where the notation is
ambigous. Generally, we will write :math:`p(x_i)`, :math:`p(x^{(k)}_i)` and
:math:`p(x^{(k)}_i, x_{i+1})` to denote the empirical probability of obseriving
the :math:`x_i` state, the :math:`x^{(k)}_i` :math:`k`-history, and the joint
probability of observing :math:`(x^{(k)}_i, x_{i+1})`.

**Please report any notational ambiguities as an**
`issue <https://github.com/ELIFE-ASU/PyInform/issues>`_.


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
