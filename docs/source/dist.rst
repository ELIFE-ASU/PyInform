.. _dist:

.. testsetup:: Dist

    from pyinform import Dist


Empirical Distributions
=======================

The :py:class:`pyinform.dist.Dist` class provides an *empirical* distribution,
i.e. a histogram, representing the observed frequencies of some fixed-size set
of events. This class is the basis for all of the fundamental information
measures on discrete probability distributions.

Examples
--------

Example 1: Construction
^^^^^^^^^^^^^^^^^^^^^^^
You can construct a distribution with a specified number of unique observables.
This construction method results in an *invalid* distribution as no
observations have been made thus far.

.. doctest:: Dist

    >>> d = Dist(5)
    >>> d.valid()
    False
    >>> d.counts()
    0
    >>> len(d)
    5

Alternatively you can construct a distribution given a list (or NumPy array)
of observation counts:

.. doctest:: Dist

    >>> d = Dist([0,0,1,2,1,0,0])
    >>> d.valid()
    True
    >>> d.counts()
    4
    >>> len(d)
    7

Example 2: Making Observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a distribution has been constructed, we can begin making observations.
There are two methods for doing so. The first uses the standard indexing
operations, treating the distribution similarly to a list:

.. doctest:: Dist

    >>> d = Dist(5)
    >>> for i in range(len(d)):
    ...     d[i] = i*i
    >>> list(d)
    [0, 1, 4, 9, 16]

The second method is to make *incremental* changes to the distribution. This
is useful when making observations of timeseries:

.. doctest:: Dist

    >>> obs = [1,0,1,2,2,1,2,3,2,2]
    >>> d = Dist(max(obs) + 1)
    >>> for event in obs:
    ...     assert(d[event] == d.tick(event) - 1)
    ...
    >>> list(d)
    [1, 3, 5, 1]

It is important to remember that :py:class:`~.dist.Dist` keeps track of your
events as you provide them. For example:

.. doctest:: Dist

    >>> obs = [1, 1, 3, 5, 1, 3, 7, 9]
    >>> d = Dist(max(obs) + 1)
    >>> for event in obs:
    ...     assert(d[event] == d.tick(event) - 1)
    ...
    >>> list(d)
    [0, 3, 0, 2, 0, 1, 0, 1, 0, 1]
    >>> d[3]
    2
    >>> d[7]
    1

If you know there are "gaps" in your time series, e.g. no even numbers, then you
can use the utility function :py:func:`~.utils.coalesce.coalesce_series` to get
rid of them:

.. doctest:: Dist

    >>> from pyinform import utils
    >>> obs = [1, 1, 3, 5, 1, 3, 7, 9]
    >>> coal, b = utils.coalesce_series(obs)
    >>> d = Dist(b)
    >>> for event in coal:
    ...     assert(d[event] == d.tick(event) - 1)
    ...
    >>> list(d)
    [3, 2, 1, 1, 1]
    >>> d[1]
    2
    >>> d[3]
    1

This can significantly improve memory usage in situations where the range of
possible states is large, but is sparsely sampled in the time series.

Example 3: Probabilities
^^^^^^^^^^^^^^^^^^^^^^^^

Once some observations have been made, we can start asking for probabilities.
As in the previous examples, there are multiple ways of doing this. The first
is to just ask for the probability of a given event.

.. doctest:: Dist

    >>> d = Dist([3,0,1,2])
    >>> d.probability(0)
    0.5
    >>> d.probability(1)
    0.0
    >>> d.probability(2)
    0.16666666666666666
    >>> d.probability(3)
    0.3333333333333333

Sometimes it is nice to just dump the probabilities out to an array:

.. doctest:: Dist

    >>> d = Dist([3,0,1,2])
    >>> d.dump()
    array([0.5       , 0.        , 0.16666667, 0.33333333])

Example 4: Shannon Entropy
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have a distribution you can do lots of fun things with it. In
this example, we will compute the shannon entropy of a timeseries of
observed values.

.. testcode:: Dist

    from math import log2
    from pyinform.dist import Dist

    obs = [1,0,1,2,2,1,2,3,2,2]
    d = Dist(max(obs) + 1)
    for event in obs:
        d.tick(event)

    h = 0.
    for p in d.dump():
        h -= p * log2(p)

    print(h)

.. testoutput:: Dist

    1.6854752972273344

Of course **PyInform** provides a function for this:
:py:func:`pyinform.shannon.entropy`.

.. testcode:: Dist

    from pyinform.dist import Dist
    from pyinform.shannon import entropy

    obs = [1,0,1,2,2,1,2,3,2,2]
    d = Dist(max(obs) + 1)
    for event in obs:
        d.tick(event)

    print(entropy(d))

.. testoutput:: Dist

    1.6854752972273344


API Documentation
-----------------

.. automodule:: pyinform.dist

    .. autoclass:: pyinform.dist.Dist

        .. automethod:: pyinform.dist.Dist.__init__

        .. automethod:: pyinform.dist.Dist.__len__

        .. automethod:: pyinform.dist.Dist.__getitem__

        .. automethod:: pyinform.dist.Dist.__setitem__

        .. automethod:: pyinform.dist.Dist.resize

        .. automethod:: pyinform.dist.Dist.copy

        .. automethod:: pyinform.dist.Dist.counts

        .. automethod:: pyinform.dist.Dist.valid

        .. automethod:: pyinform.dist.Dist.tick

        .. automethod:: pyinform.dist.Dist.probability

        .. automethod:: pyinform.dist.Dist.dump
