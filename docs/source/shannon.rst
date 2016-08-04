.. _shannon:

Shannon Information Measures
============================

The :py:mod:`pyinform.shannon` module provides a collection of entropy and
information measures on discrete probability distributions
(:py:class:`pyinform.dist.Dist`). This module forms the core of PyInform as
all of the time series analysis functions are built upon this module.

Examples
--------

Example 1: Entropy and Random Numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:func:`pyinform.shannon.entropy` function allows us to calculate the
Shannon entropy of a distributions. Let's try generating a random distribution
and see what the entropy looks like? ::

    import numpy as np

    xs = np.random.randint(0,10,10000)
    d = Dist(10)
    for x in xs:
        d.tick(x)
    print(entropy(d))       # 3.32137023165359
    print(entropy(d, b=10)) # 0.9998320664331565

This is exactly what you should expect; the pseudo-random number generate does
a decent job producing integers in a uniform fashion.

Example 2: Mutual Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How correlated are consecutive integers? Let's find out using
:py:func:`mutual_info`. ::

    from pyinform.dist import Dist
    from pyinform.shannon import mutual_info
    import numpy as np

    obs = np.random.randint(0, 10, 10000)
    
    p_xy = Dist(100)
    p_x  = Dist(10)
    p_y  = Dist(10)

    for x in obs[:-1]:
        for y in obs[1:]:
            p_x.tick(x)
            p_y.tick(y)
            p_xy.tick(10*x + y)

    print(mutual_info(p_xy, p_x, p_y))       # -1.7763568394002505e-15
    print(mutual_info(p_xy, p_x, p_y, b=10)) # -6.661338147750939e-16

Due to the subtlties of floating-point computation we don't get zero. Really,
though the mutual information is zero.

Example 3: Relative Entropy and Biased Random Numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Okay. Now let's generate some binary sequences. The first will be roughly
uniform, but the second will be biased toward 0. ::

    from pyinform.dist import Dist
    from pyinform.shannon import relative_entropy
    import numpy as np
    
    p = Dist(2)
    q = Dist(2)

    ys = np.random.randint(0, 2, 10000)
    for y in ys:
        p.tick(y)

    xs = np.random.randint(0, 6, 10000)
    for i, _ in enumerate(xs):
        xs[i] = (((xs[i] % 5) % 4) % 3) % 2 
        q.tick(xs[i])

    print(relative_entropy(q,p)) # 0.3338542254583825
    print(relative_entropy(p,q)) # 0.40107198925821924

API Documentation
-----------------

.. automodule:: pyinform.shannon

    .. autofunction:: pyinform.shannon.entropy

    .. autofunction:: pyinform.shannon.mutual_info

    .. autofunction:: pyinform.shannon.conditional_entropy

    .. autofunction:: pyinform.shannon.conditional_mutual_info

    .. autofunction:: pyinform.shannon.relative_entropy
