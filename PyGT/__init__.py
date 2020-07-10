"""
PyGT - Python analysis of metastable Markov models using graph transformation
---------------------------------------------------------------------------------------------
Graph transformation [Wales09]_ is a deterministic dimensionality reduction algorithm
that iteratively removes nodes from a Markov Chain while preserving the mean first
passage time (MFPT) and branching probabilites between the retained nodes.
The original Markov chain does not need to satisfy detailed balance.
This package provides an efficient implementation of the graph transformation algorithm,
accelerated via partial block matrix inversions [Swinburne20]_ for abitrary
discrete-time or continuous-time Markov chains [Kannan20a]_. Code is also provided
for the calculation of phenomenological rate constants between endpoint macrostates [Wales09]_.


We also include code for two different approaches to the dimensionality reduction
of Markov chains using the graph transformation algorithm. In the first
approach, we consider the problem of estimating a reduced Markov chain given
a partitioning of the original Markov chain into communities of microstates
(nodes) [Kannan20a]_. Various implementations of the inter-community rates are provided,
including the simplest expression given by the local equilibrium approximation,
as well as the optimal rates originally derived by Hummer and Szabo. In the second
approach, which we call partial graph transformation [Swinburne20]_ [Kannan20b]_, select nodes that
contribute the least to global dynamics are renormalized away with graph
transformation. The result is a smaller-dimensional Markov chain that is
better-conditioned for further numerical analysis.

All methods are discussed in detail in the following manuscripts, which should
also be cited when using this software:

.. References::

.. [Wales09] D.J. Wales, *Calculating rate constants and committor probabilities for transition networks by graph transformation*, J. Chemical Physics (2009), https://doi.org/10.1063/1.3133782

.. [Swinburne20] T.D. Swinburne and D.J. Wales, *Defining, Calculating, and Converging Observables of a Kinetic Transition Network*, J. Chemical Theory and Computation (2020), https://doi.org/10.1021/acs.jctc.9b01211

.. [Kannan20a] D. Kannan, D.J. Sharpe, T.D. Swinburne, D.J. Wales, *Calculating mean first passage time matricies of discrete- and continuous- time Markov chains using graph transformation*, In Prep. (2020)

.. [Kannan20b] D. Kannan, D.J. Sharpe, T.D. Swinburne, D.J. Wales, *Dimensionality reduction of complex networks with graph transformation* In Prep. (2020)

"""
__version__ = "0.2.0"


from . import io

from .core import *
