Tutorial with PyGT
=========================

To help you get started, here, we outline a step-by-step guide on how to use the
`PyGT` package for the analysis of discrete- or continuous-time Markov
chains. First, we will go over the various submodules in the package and then
walk through various examples of using graph transformation for computing mean
first passage times and estimating reduced Markov chains given a community
structure.

Getting Started
---------------

`PyGT` has to two submodules, `lib` and `ktn`. The `lib` module contains
an implementation of the GT algorithm as well as wrappers for analyzing first
passage time statistics between two endpoint regions of interest. The `ktn`
module provides functions to estimate a reduced Markov chain given a community
structure that groups together microstates of the original Markov chain.

At the start of any python script, include the following import statements:

.. code-block:: python

    >>> #library code
    >>> import lib.ktn_io as kio
    >>> import lib.gt_tools as gt
    >>> import lib.partialGT as pgt
    >>> import lib.fpt_stats as fpt
    >>> import lib.conversion as convert
    >>> from ktn.ktn_analysis import *
    >>> #other modules
    >>> import numpy as np
    >>> import scipy as sp
    >>> import scipy.linalg as spla 
    >>> from scipy.sparse.linalg import eigs
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.linalg import eig
    >>> from scipy.linalg import expm
    >>> from pathlib import Path
    >>> import pandas as pd
    >>> from matplotlib import pyplot as plt
    >>> import seaborn as sns
    >>> sns.set()

Input Files
-----------

The graph transformation algorithm requires a branching probability matrix
:math:`\textbf{B}` as input, as well as a vector of escape rates specifying the
inverse waiting time of each node. We will demonstrate a few differet ways of
loading in a Markov chain for analysis with GT.
   
1. If the Markov chain is constructed from the energy landscape, the
   `lib/ktn_io` package contains a function ``load_mat()`` to compute Arrhenius transition
   rates between potential energy minimuma. For example,

    .. code-block:: python

        >>> temp = 1.0
        >>> #GT setup
        >>> B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=Path('KTN_data/32state'), beta=1/temp)
        >>> BF = beta*u-s
        >>> BF -= BF.min()

    In this example, the ``load_mat()`` function read the connectivity of the
    chain from the files `KTN_data/32state/min.data` and `KTN_data/32state/ts.data`
    and returned the sparse branching probability matrix :math:`\textbf{B}`, the
    sparse transition rate matrix :math:`\textbf{K}`, and the sparse diagonal matrix
    :math:`\textbf{D}` with elements corresponding to the escape rates from each
    node. The arrays `u` and `s` correspond to the energies and entropies of the
    nodes so that the stationary distribution :math:`\pi` is given by the Boltzmann
    distribution :math:`\pi_i = e^{-\beta F_i}/\sum_i e^{-\beta F_i}`.

2. CTMC: If you already have a transition rate matrix :math:`\textbf{Q}` representing
   a continuous-time Markov chain, such that the columns of :math:`\textbf{Q}`
   sum to zero, then we can compute the branching probability
   matrix and vector of escape rates as follows:

    .. code-block:: python

        >>> #subtract the diagonal elements of K
        >>> K = Q - np.diag(np.diag(Q))
        >>> escape_rates = -1*np.diag(Q)
        >>> B = K@np.diag(1./escape_rates)

    Here, `B` is a dense matrix with elements :math:`B_{ij} = k_{ij}\tau_j`.

3. DTMC: To setup a GT calculation for the corresponding discrete-time Markov
   chain estimating at a lag time :math:`\tau`,

    .. code-block:: python

        >>> #choose a lag time
        >>> tau = 0.001
        >>> T = expm(Q*tau) #compute T with matrix exponential
        >>> nnodes = T.shape[0]
        >>> escape_rates = np.tile(1/tau, nnodes)

    In this case, we replace `B` with the discrete-time transition probability
    matrix `T`, which can either be calculated with a matrix exponential, or can
    be parameterized from simulation data.

Computing Mean First Passage Times
----------------------------------

First, we must specify endpoint macrostates (communities) of interest. To do so,
we can either specify a community structure, or explicitly enumerate the nodes
in the :math:`\mathcal{A}` and :math:`\mathcal{B}` sets.


