"""
PyGT - Graph transformation and reduction in Python
---------------------------------------------------------------------------------------------



**Graph transformation is designed for highly metastable (ill-conditioned) systems, where linear algebra methods fail.** ``PyGT`` **extracts observables and produces coarse-grained models.**

.. note::


	You can install ``PyGT`` using the ``pip`` package manager (preferably in a virtualenv):

	.. code-block:: bash

		pip install PyGT

	Simplest possible usage in Python:

	.. code-block:: python

		import PyGT

		# B : branching probabilities, tau : waiting times, pi : Boltzmann
		# rm_vec : Boolean vector of nodes to remove

		gt_B, gt_tau = PyGT.GT.blockGT(rm_vec,B,tau,block=10)

		# gt_B,gt_tau : exact branching probabilites and waiting times

.. note::

	Tutorials (see menu) can be run online with binder: |binderlink|

	.. |binderlink| image:: https://mybinder.org/badge_logo.svg
					:target: https://mybinder.org/v2/gh/tomswinburne/PyGT.git/master?filepath=examples%2Fbasic_functions.ipynb



	The notebooks can be cloned from the PyGT github repo:

	.. code-block:: bash

		# clone entire source code and examples
		git clone https://github.com/tomswinburne/PyGT.git
		# go to examples folder
		cd PyGT/examples
		# run notebook
		jupyter-notebook basic-functions.ipynb

Graph transformation [Wales09]_ is a deterministic dimensionality reduction algorithm
that iteratively removes nodes from a Markov Chain while preserving the mean first
passage time (MFPT) and branching probabilites between the retained nodes.
The original Markov chain does not need to satisfy detailed balance.

This package provides an efficient implementation of the graph transformation algorithm,
accelerated via partial block matrix inversions [Swinburne20a]_ for abitrary
discrete-time or continuous-time Markov chains [Kannan20a]_. Code is also provided
for the calculation of first passage time statistics and phenomenological
rate constants between endpoint macrostates [Wales09]_ [Swinburne20b]_.

We also include code for two different approaches to the dimensionality reduction
of Markov chains using the graph transformation algorithm. In the first
approach, we consider the problem of estimating a reduced Markov chain given
a partitioning of the original Markov chain into communities of microstates
(nodes) [Kannan20a]_. Various implementations of the inter-community rates are provided,
including the simplest expression given by the local equilibrium approximation,
as well as the optimal rates originally derived by Hummer and Szabo. In the second
approach, which we call partial graph transformation [Swinburne20a]_ [Kannan20b]_, select nodes that
contribute the least to global dynamics are renormalized away with graph
transformation. The result is a smaller-dimensional Markov chain that is
better-conditioned for further numerical analysis.

All methods are discussed in detail in the following manuscripts, which should
also be cited when using this software:

.. rubric:: References

.. [Wales09] D.J. Wales, *Calculating rate constants and committor probabilities for transition networks by graph transformation*, J. Chemical Physics (2009), https://doi.org/10.1063/1.3133782

.. [Swinburne20a] T.D. Swinburne and D.J. Wales, *Defining, Calculating, and Converging Observables of a Kinetic Transition Network*, J. Chemical Theory and Computation (2020), https://doi.org/10.1021/acs.jctc.9b01211

.. [Swinburne20b] T.D. Swinburne, D. Kannan, D.J. Sharpe and D.J. Wales, *Rare Events and First Passage Time Statistics From the Energy Landscape*, Submitted to J. Chemical Physics (2020)

.. [Kannan20a] D. Kannan, D.J. Sharpe, T.D. Swinburne and D.J. Wales, *Dimensionality reduction of Markov chains using mean first passage times with graph transformation*, In Prep. (2020)

.. [Kannan20b] D. Kannan, D.J. Sharpe, T.D. Swinburne and D.J. Wales, *Dimensionality reduction of complex networks with graph transformation*, In Prep. (2020)

"""
__version__ = "0.2.0"

from . import GT

from . import io

from . import tools

from . import stats

from . import mfpt

from . import spectral
