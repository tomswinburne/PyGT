# -*- coding: utf-8 -*-
r"""
Calculate matricies of mean first passage times with graph transformation
-------------------------------------------------------------------------

.. note::

	Install the `pathos` package to parallelize MFPT computations, with e.g.

	.. code-block:: none

		```
		pip install pathos
		```
"""

import numpy as np
from io import StringIO
import time,os, importlib
np.set_printoptions(linewidth=160)
from . import GT

import scipy as sp
import scipy.linalg as spla

""" test for ipython environment (is this being loaded from a notebook) """
try:
	__IPYTHON__
except NameError:
	in_notebook = False
else:
	in_notebook = True

""" test for tqdm progress bars """

try:
	if in_notebook:
		from tqdm import tqdm_notebook as tqdm
	else:
		from tqdm import tqdm
	has_tqdm=True
except:
	has_tqdm=False

try:
	from pathos.multiprocessing import ProcessingPool as Pool
	have_pathos = True
except:
	have_pathos = False


def full_MFPT_matrix(B, tau, pool_size=1, screen=False, **kwargs):
	r"""Compute full matrix of inter-microstate MFPTs with GT.

	Parameters
	----------
	B : sparse or dense matrix (N,N)
		branching probability matrix.
	tau : array-like (N,)
		vector of escape rates, i.e. inverse waiting times from each node.
	pool_size : int,optional
		Number of cores over which to parallelize computation.
		only attempted if ``pool_size>1`` and ``pathos`` package is installed.
		Default=1.
	screen: bool, optional
		Show progress bar. Default=False
	Returns
	-------
	mfpt : np.ndarray[float64] (N,N)
		matrix of inter-microstate MFPTs between all pairs of nodes

	"""
	N = tau.size

	mfpt = np.zeros((N,N))

	i_a = np.arange(N*N) % N
	j_a = np.arange(N*N) // N
	matrix_elements = np.vstack((i_a,j_a)).T[i_a<j_a,:]

	if screen and has_tqdm:
		pbar = tqdm(total=len(matrix_elements),leave=True,mininterval=0.0,desc='MFPT matrix computation')


	def given_ij(ij):
		i, j = ij
		if screen and has_tqdm:
			if have_pathos and pool_size>1:
				pbar.update(pool_size)
			else:
				pbar.update(1)
		MFPTAB, MFPTBA = compute_MFPT(i, j, B, tau, **kwargs)
		return MFPTAB, MFPTBA

	if have_pathos and pool_size>1:
		with Pool(processes=pool_size) as p:
			results = p.map(given_ij, matrix_elements)
		for k, result in enumerate(results):
			i, j = matrix_elements[k]
			mfpt[i][j], mfpt[j][i] = result
	else:
		for ij in matrix_elements:
			i, j = ij
			mfpt[i][j], mfpt[j][i] = given_ij(ij)
	if screen and has_tqdm:
		pbar.close()

	return mfpt

def community_MFPT_matrix(communities, B, tau, pi, MS_approx=False, pool_size=1, screen=False,**kwargs):
	r"""Compute matrix of effective inter-macrostate MFPTs with GT, defined as
	:math:`\mathcal{T}_{AB} = \sum_{i\in A, j\in B} \pi_i,\pi_j\mathcal{T}_{ij} / (\sum_{i\in A}\pi_i)/(\sum_{j\in B}\pi_j)`.
	This can be shown to satisfy the Kemeny constant condition, and thus is
	suitable for producing a coarse grained rate matrix. [Kannan20a]_

	Parameters
	----------
	communities : dict
		mapping from community ID (0-indexed) to a boolean array
		of shape (N, ) which selects out the states in that community.
		Communities must be disjoint.
		Communities must be disjoint.
	B : sparse or dense matrix (N,N)
		branching probability matrix.
	tau : array-like (N,), float
		vector of waiting times from each node.
	pi : array-like (N,), float
		microscopic stationary probability distribution

	MS_approx : bool, optional
		If True, assume all communities are sufficiently metastable to
		use only one microstate pair per community pair, a much more
		efficient but approximate computation. [Kannan20a]_

	pool_size : int,optional
		Number of cores over which to parallelize computation. Default=1
		Only active if MS_approx = False
	screen: bool, optional
		Print progress. Default = False
	Returns
	-------
	Pi : array-like (N_comm,)
		Macroscopic stationary distribution. Indexed in ascending order
	tau_AB : dense matrix (N_comm,N_comm)
		matrix of effective inter-macrostate MFPTs
	"""

	comms = np.unique(communities)
	comms = comms[comms.argsort()] # force ascending order

	N = pi.size
	ck = communities.keys()
	Nc = len(ck)

	pi /= pi.sum() # ensure normalization
	c_pi = np.r_[[pi[communities[c]].sum() for c in ck]]

	if not MS_approx:
		# (Nc,N) projection tensor of stationary distribution
		A_pi = np.r_[[(pi * communities[c] )/pi[communities[c]].sum() for c in ck]]
		tauM = full_MFPT_matrix(B,tau,pool_size,screen=screen)

		c_tau = A_pi@tauM@A_pi.T

	else:
		if screen and has_tqdm:
			pbar = tqdm(total=(Nc*(Nc+1))//2,leave=True,mininterval=0.0,desc='MFPT matrix computation (MS approx)')

		c_tau = np.zeros((Nc,Nc))
		for i,cA in enumerate(communities.keys()):
			for j,cB in enumerate(communities.keys()):
				if i<=j:
					# select lowest free energy state
					i_s = np.arange(N)[communities[cA]][pi[communities[cA]].argmax()]
					j_s = np.arange(N)[communities[cB]][pi[communities[cB]].argmax()]
					c_tau[i][j], c_tau[j][i] = compute_MFPT(i_s, j_s, B, tau, **kwargs)
					if screen and has_tqdm:
						pbar.update(1)
		if screen and has_tqdm:
			pbar.close()
	return c_pi, c_tau

def compute_MFPT(i, j, B, tau, block=10, **kwargs):
	r"""Compute the inter-microstate :math:`i\leftrightarrow j` MFPT using GT.
	Called by ``full_MFPT_matrix()``. Unlike ``compute_rates()`` function,
	which assumes there is at least 2 microstates in the absorbing macrostate,
	this function does not require knowledge of equilibrium
	occupation probabilities since :math:`\mathcal{T}_{ij}=\tau_j^\prime/B_{ij}^\prime`
	when there are only two nodes remaining after GT

	Parameters
	----------
	i : int
		node-ID (0-indexed) of first microstate.
	j : int
		node-ID (0-indexed) of second microstate.
	B : sparse or dense matrix (N,N)
		branching probability matrix.
	tau : array-like (N,)
		vector of escape rates, i.e. inverse waiting times from each node.
	block : int, optional
		block size for matrix generalization GT procedure.
		Reverts to slower but guaranteed stable one-by-one GT (block=1)
		when ``numpy`` matrix inversion routine raises errors. Default=10

	Returns
	-------
	MFPTij : float
		mean first passage time :math:`i \leftarrow j`
	MFPTji : float
		mean first passage time :math:`j \leftarrow i`

	"""

	N = B.shape[0]
	AS = np.zeros(N, bool)
	AS[i] = True
	BS = np.zeros(N, bool)
	BS[j] = True
	#GT away all I states
	inter_region = ~(AS+BS)
	#left with a 2-state network
	rB, tau_Fs = GT.partialGT(rm_vec=inter_region,B=B,tau=tau,rates=False,block=block,**kwargs)
	rD = 1.0/tau_Fs
	rN = tau_Fs.size
	#remaining network only has 1 in A and 1 in B = 2 states
	r_AS = AS[~inter_region]
	r_BS = BS[~inter_region]
	#tau_a^F / P_Ba^F
	P_BA = rB[r_BS, :][:, r_AS]
	P_AB = rB[r_AS, :][:, r_BS]
	MFPTBA = tau_Fs[r_AS]/P_BA
	MFPTAB = tau_Fs[r_BS]/P_AB
	return MFPTAB[0,0], MFPTBA[0,0]
