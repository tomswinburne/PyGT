"""
A Python Implementation of the Graph Transformation Algorithm
-------------------------------------------------------------
Graph transformation is a deterministic dimensionality reduction algorithm that
iteratively removes nodes from a Markov Chain while preserving the mean first
passage time (MFPT) from a set of initial nodes, :math:`\mathcal{B}`, to a set
of asorbing nodes, :math:`\mathcal{A}`. The original Markov chain does not need
to satisfy detailed balance, nor is there any constraint on the initial
probability distribution in :math:`\mathcal{B}`. This package provides an
implementation of the graph transformation algorithm for abitrary discrete-time
or continuous-time Markov chains. Code is also provided for the calculation of 
phenomenological rate constants between endpoint macrostates using expressions 
from discrete path sampling. 

We also include code for two different approaches to the dimensionality reduction 
of Markov chains using the graph transformation algorithm. In the first
approach, we consider the problem of estimating a reduced Markov chain given a partitioning of
the original Markov chain into communities of microstates (nodes). [1]_ Various
implementations of the inter-community rates are provided, including the
simplest expression given by the local equilibrium approximation, as well as
the optimal rates originally derived by Hummer and Szabo. In the second
approach, which we call partial graph transformation, [2]_ select nodes that
contribute the least to global dynamics are renormalized away with graph
transformation. The result is a smaller-dimensional Markov chain that is
better-conditioned for further numerical analysis.

All methods are discussed in detail in the following manuscripts, which should
also be cited when using this software:

.. [1] D. Kannan, D. J. Sharpe, T. D. Swinburne, D. J. Wales, "Dimensionality reduction of Markov chains from mean first passage times using graph transformation." *J. Chem. Phys.* (2020)

.. [2] D. Kannan, D. J. Sharpe, T. D. Swinburne, D. J. Wales, "Dimensionality reduction of complex networks with graph transformation" *Phys. Rev. E.* (2020)

"""
__version__ = "0.1.0"
