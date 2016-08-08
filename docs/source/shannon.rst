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

References
----------

.. [Cover1991a] T.M. Cover amd J.A. Thomas (1991). "Elements of information theory" (1st ed.). New York: Wiley. ISBN 0-471-06259-6.

.. [Dobrushin1959] Dobrushin, R. L. (1959). "General formulation of Shannon's main theorem in information theory". Ushepi Mat. Nauk. 14: 3-104.

.. [Kullback1951a] Kullback, S.; Leibler, R.A. (1951). "`On information and sufficiency`__". Annals of Mathematical Statistics. 22 (1): 79-86. doi:10.1214/aoms/1177729694. MR 39968.
.. __: http://projecteuclid.org/DPubS?service=UI&version=1.0&verb=Display&handle=euclid.aoms/1177729694

.. [Shannon1948a] Shannon, Claude E. (July-October 1948). "`A Mathematical Theory of Communication`__". Bell System Technical Journal. 27 (3): 379-423. doi:10.1002/j.1538-7305.1948.tb01448.x.
.. __: https://dx.doi.org/10.1002%2Fj.1538-7305.1948.tb01338.x

.. [Wyner1978] Wyner, A. D. (1978). "`A definition of conditional mutual information for arbitrary ensembles`__". Information and Control 38 (1): 51-59. doi:10.1015/s0019-9958(78)90026-8.
.. __: http://www.sciencedirect.com/science/article/pii/S0019995878900268
