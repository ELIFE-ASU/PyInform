Empirical Distributions
=======================

The ``pyinform.dist.Dist`` class provides an *empirical* distribution, i.e. a
histogram, representing the observed frequencies of some fixed-size set of
events. This class is the basis for all of the fundamental information measures
on discrete probability distributions.

Examples
--------

Example 1: Construction
^^^^^^^^^^^^^^^^^^^^^^^
You can construct a distribution with a specified number of unique observables.
Using this construction method results in an *invalid* distribution, as no
observations have been made thus far. ::

    d = Dist(5)
    assert(not d.valid())
    assert(d.counts() == 0)
    assert(len(d) == 5)

Alternatively, you can construct a distribution given an list (or NumPy array)
of observation counts: ::

    d = Dist([0,0,1,2,1,0,0])
    assert(d.valid())
    assert(d.counts == 4)
    assert(len(d) == 7)

Example 2: Making Observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a distribution has been constructed, we can begin making observations.
There are two methods for doing so. The first uses the standard indexing
operations, treating the distribution similarly to a list: ::

    d = Dist(5)
    for i in range(len(d)):
        d[i] = i*i              # set the i-th event count
        assert(d[i] == i*i)     # no really, it should be set now

The second method is to make *incremental* changes to the distribution. This
is useful when making observations of timeseries: ::

    obs = [1,0,1,2,2,1,2,3,2,2]

    d = Dist(max(obs) + 1)
    for event in obs:
        d.tick(event)       # increment the number of observations of `event`

    assert(d[0] == 1)
    assert(d[1] == 3)
    assert(d[2] == 5)
    assert(d[3] == 1)

Example 3: Probabilities
^^^^^^^^^^^^^^^^^^^^^^^^

Once some observations have been made, we can start asking for probabilities.
As in the previous examples, there are multiple ways of doing this. The first
is to just ask for the probability of a given event. ::

    d = Dist([3,0,1,2])

    assert(d.probability(0) == 3./6.)
    assert(d.probability(1) == 0./6.)
    assert(d.probability(2) == 1./6.)
    assert(d.probability(2) == 2./6.)

Sometimes it is nice to just dump the probabilities out to an array: ::

    d = Dist([3,0,1,2])

    assert((d.dump() == [3./6., 0./6., 1./6., 2./6.]).all())

Example 4: Shannon Entropy "The Hard Way"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have a distribution you can do lots of fun things with it. In
this example, we will compute the shannon entropy of a timeseries of
observed values. ::

    from math import log2

    obs = [1,0,1,2,2,1,2,3,2,2]
    d = Dist(max(obs) + 1)
    for event in obs:
        d.tick(event)
    
    h = 0.
    for p in d.dump():
        h -= p * log2(p)

    assert(h == (log2(10) - 3*log2(3)/10 - 5*log2(5)/10))

Of course **PyInform** provides a function for this:
:py:func:`pyinform.shannon.entropy`. ::

    from pyinform.shannon import entropy

    obs = [1,0,1,2,2,1,2,3,2,2]
    d = Dist(max(obs) + 1)
    for event in obs:
        d.tick(event)
    
    h = entropy(dist)
    assert(h == (log2(10) - 3*log2(3)/10 - 5*log2(5)/10))


API Documentation
-----------------

.. autoclass:: pyinform.dist.Dist
    :members:

    .. automethod:: pyinform.dist.Dist.__init__

    .. automethod:: pyinform.dist.Dist.__len__

    .. automethod:: pyinform.dist.Dist.__getitem__

    .. automethod:: pyinform.dist.Dist.__setitem__
