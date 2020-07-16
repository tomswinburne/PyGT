# -*- coding: utf-8 -*-
r"""
Iteratively remove nodes from a Markov chain with graph transformation
----------------------------------------------------------------------

This module implements the graph transformation algorithm to eliminate
nodes from a discrete- or continuous-time Markov chain. When the removed nodes
are chosen judiciously, the resulting network is less sparse, of lower dimensionality,
and is generally better-conditioned. See `PyGT.tools` for various tools
which help select nodes to eliminate, namely, ranking nodes based on their mean
waiting times and equilibrium occupation probabilities. [Kannan20b]_

The graph transformation algorithm requires a branching probability matrix
:math:`\textbf{B}` with elements :math:`B_{ij}=k_{ij}{\tau}_j` where :math:`k_{ij}` is
the :math:`i \leftarrow j` inter-microstate transition rate and :math:`\tau_j`
is the mean waiting time of node :math:`j`. In a discrete-time Markov chain,
:math:`\textbf{B}` is replaced with the discrete-time transition probability
matrix :math:`\textbf{T}(\tau)` and the waiting times of all nodes are uniform,
equal to the lag time :math:`\tau`.

In each iteration of GT, a single node :math:`x` is removed, and the branching
probabilities and waiting times of the neighboring nodes are updated according
to

.. math::

	\begin{eqnarray}
		B_{ij}^{\prime} &\leftarrow B_{ij} + \frac{B_{ix}B_{xj}}{1-B_{xx}} \\
		\tau_j^\prime  &\leftarrow \tau_j + \frac{B_{xj}\tau_j}{1-B_{xx}}
	\end{eqnarray}

A matrix version of the above equations permits the removal of blocks of nodes
simulatenously. The code monitors for stability in the inversion
algorithm, to allow for for one-by-one node removal instead if a block is
ill-conditioned. [Swinburne20]_

"""

import os,time
import numpy as np

from scipy.sparse import csgraph, csr_matrix, lil_matrix,eye, save_npz, load_npz, diags,kron, issparse
from scipy.sparse.linalg import spsolve,inv

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



def blockGT(rm_vec,B,tau,block=20,order=None,rates=False,Ndense=50,screen=False):

	r"""
	Main function for GT code, production of a reduced matrix by graph transformation.

	Parameters
	----------
	rm_vec:	(N,) array-like, bool
			Boolean array of which nodes to remove

	B:		(N,N) dense or sparse matrix
			Matrix of branching probabilities (CTMC) or transition probabilities (DTMC)

	tau:	(N,) array-like
			Array of waiting times (CTMC) or lag times (DTMC)

	block:	int, optional
			Number of node to attempt to remove simultaneously. Default = 20

	order: 	(N,) array-like, optional
			Order in which to remove nodes. Default ranks on node connectivity.
			Modify with caution: large effect on performance

	rates: 	bool, optional
			Whether to return the GT-reduced rate matrix in addition to B and tau.
			Only vaid for CTMC case. Default = False

	Ndense: int, optional
			Force switch to dense representation if N<Ndense. Default = 50

	screen: bool, optional
			Whether to print progress of GT. Default = False


	Returns
	-------
	B:		(N',N') dense or sparse matrix
			Matrix of N'<N renormalized branching probabilities.
			Will be returned as same type (sparse/dense) as input

	tau:	(N',) array-like
	     	Array of N'<N renormalized waiting times

	K:		(N',N') dense or sparse matrix (same type as B)
			Matrix of N'<N renormalized transition rates. Only if ``rates=True``


	"""
	dense = not issparse(B)
	force_sparse = issparse(B)

	if dense:
		density = B[B>0.0].sum()/float(B.size)

	retry=0
	N = rm_vec.size
	#total number of states to remove
	NI = rm_vec.sum()
	D = 1.0 / tau


	if screen and has_tqdm:
		pbar = tqdm(total=NI,leave=True,mininterval=0.0,desc='GT')
	tst = 0.0
	tmt = 0.0
	tc = 0

	pass_over = np.empty(0,bool)
	pobar = None

	dense_onset = 0
	if dense:
		dense_onset = B.shape[0]
	if screen:
		t = time.time()
	while NI>0:

		if N<Ndense and not dense:
			#when network is small enough, more efficient to switch to dense format
			dense = True
			#nnz is number of stored values in B, including explicit zeros
			density = float(B.nnz) /  float( B.shape[0]*B.shape[1])
			dense_onset = B.shape[0]
			B = B.todense()

		rm = np.zeros(N,bool)
		if pass_over.sum()>0:
			Bd = np.ravel(B.diagonal())
			Bd[~pass_over] = Bd.min()-1.0
			Bd[~rm_vec] = Bd.min()-1.0
			rm[Bd.argmax()] = True
			if not pobar is None:

				pobar.update(1)
		else:
			#order the nodes to remove
			if order is None:
				if not dense:
					#order contains number of elements in each row
					#equivalent to node in-degree
					order = B.indptr[1:]-B.indptr[:-1]
				else:
					order = np.linspace(0.,1.0,B.shape[0])
			if not pobar is None:
				pobar.close()
			pobar = None
			order[~rm_vec] = order.max()+1
			rm[order.argsort()[:min(block,NI)]] = True

		B, tau, success = singleGT(rm,B,tau)

		if success:
			if screen and has_tqdm:
				pbar.update(rm.sum())
			N -= rm.sum()
			NI -= rm.sum()
			rm_vec = rm_vec[~rm]
			if not (order is None):
				order = order[~rm]
			if pass_over.sum()>0:
				pass_over = pass_over[~rm]
			rmb = 1 + (block-1)*int(pass_over.sum()==0)
		else:
			pass_over = rm
			rmb = 1
			retry += 1
			if screen and has_tqdm:
				pobar = tqdm(total=rm.sum(),leave=False,mininterval=0.0,desc="STATE-BY-STATE SUBLOOP")
	if screen and has_tqdm:
		pbar.close()

	if dense and screen:
		print("GT BECAME DENSE AT N=%d, density=%f" % (dense_onset,density))

	# revert back to sparse for final clean up
	B = csr_matrix(B)

	B.eliminate_zeros()

	Bd = np.ravel(B.diagonal()) # only the diagonal (Bd_x = B_xx)
	Bn = B - diags(Bd) # B with no diagonal (Bn_xx = 0, Bn_xy = B_xy)
	Bn.eliminate_zeros()
	Bnd = np.ravel(Bn.sum(axis=0)) # Bnd_x = sum_x!=y B_yx = 1-B_xx
	nBd = np.zeros(N)
	nBd[Bd>0.99] = Bnd[Bd>0.99]
	nBd[Bd<0.99] = 1.0-Bd[Bd<0.99]
	omB = diags(nBd) - Bn # 1-B
	tau = np.ravel(tau.flatten())
	if dense:
		B = B.todense()
	if rates:
		K = -omB.dot(diags(1.0/tau)) # (B-1).D = K ( :) )
		if dense:
			K = K.todense()

	if screen:
		print("GT done in %2.2g seconds with %d floating point corrections" % (time.time()-t,retry))

	if rates:
		return B,tau,K
	return B,tau


def singleGT(rm_vec,B,tau):
	r"""
	Single iteration of GT algorithm used by main GT function.
	Either removes a single node with float precision correction [Wales09]_
	or attemps node removal via	matrix inversion [Swinburne20]_. In the latter
	case, if an error is raised by ``np.linalg.inv`` this is communicated
	through ``success``

	Parameters
	----------
	rm_vec:	(N,) array-like, bool
			Boolean array of which nodes to remove

	B:		(N,N) dense or sparse matrix
			Matrix of branching probabilities

	tau:	(N,) array-like
	     	Array of waiting times

	Returns
	-------
	B:		(N',N') dense or sparse matrix
			Matrix of N'<N renormalized branching probabilities

	tau:	(N',) array-like
	     	Array of N'<N renormalized waiting times

	success: bool
			 False if ``LinAlgError`` raised by ``np.linalg.inv``

	"""

	dense = not issparse(B)

	Bxx = B[rm_vec,:][:,rm_vec]
	Bxj = B[rm_vec,:][:,~rm_vec]
	Bix = B[~rm_vec,:][:,rm_vec]
	Bij = B[~rm_vec,:][:,~rm_vec]
	iDjj = tau[~rm_vec]
	iDxx = tau[rm_vec]

	if rm_vec.sum()>1: # Try block GT
		# Get Bxx
		Bd = np.ravel(Bxx.diagonal())

		# Get sum Bix for i not equal x
		Bxxnd = Bxx - np.diag(Bd)
		Bs = np.ravel(Bix.sum(axis=0))
		Bs += np.ravel(Bxxnd.sum(axis=0))

		# Float
		Bs[Bd<0.99] = 1.0-Bd[Bd<0.99]
		iGxx = np.diag(Bs) - Bxxnd
		try:
			Gxx = np.linalg.inv(iGxx)
		except np.linalg.LinAlgError as err:
			return B,tau,False
		iDG = Gxx.transpose().dot(iDxx).transpose()
		iDjj += np.ravel(Bxj.transpose().dot(iDG)).flatten()

		# This is computational bottleneck
		if dense:
			Bij += Bix@Gxx@Bxj
		else:
			Bij += Bix@csr_matrix(Gxx)@Bxj

		return Bij,iDjj,True
	else: # Standard GT
		Bxx = Bxx.sum()
		if Bxx>0.99:
			b_xx = Bix.sum()
		else:
			b_xx = 1.0-Bxx
		if dense:
			iDjj = iDjj.flatten()
			Bxj = np.ravel(Bxj.flatten() / b_xx)
			Bix = np.ravel(Bix.flatten())
			Bij += np.outer(Bix,Bxj)
			iDjj += iDxx.sum() * Bxj
		else:
			Bxj.data /= b_xx
			Bij += kron(Bix,Bxj)
			iDjj[Bxj.indices] += Bxj.data*iDxx

		return Bij,iDjj,True
