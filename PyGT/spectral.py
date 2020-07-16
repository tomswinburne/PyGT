# -*- coding: utf-8 -*-
r"""
Spectral analysis for community-based dimensionality reduction
--------------------------------------------------------------

Produce projection matricies in order to reduce a CTMC rate matrix
from a given community structure, using the local equilibrium approximation
(LEA) or spectral clustering method as investigated in [Swinburne20b]_.

The LEA method assumes each community is locally metastable, meaning
the distribution of states in each community will be approximately
proportional to the local Boltzmann distribution :math:`\pi_j`.

The LEA thus takes a single left and right vector pair per community `J`,
namely

.. math::

	\begin{equation}
		[{\bf 1}_J]_j = \delta(j\in J)
		,\quad
		[\hat{\pi}_J]_j=\delta(j\in J)
		\frac{\pi_j}{\sum_{j'\in J}\pi_{j'}},
	\end{equation}

to produce the reduced rate matrix, which corresponds to the local equilibrum
distribution projected onto the community.

The spectral clustering method generalizes this approach, performing a local
eigendecomposition and projection to generate a set of left and right vector
pairs, from which a subset is used to produce the reduced rate matrix.
In particular, the set is projected such that the slowest eigenvector pair
becomes the LEA pair :math:`({\bf 1}_J,\hat{\pi}_J)`, allowing interpolation
between the LEA and the exact solution

The vector pairs are chosen until the first ``nmodes`` moments of the
community escape time is reprodued to a relative error of ``obs_err``.
Further details can be found in [Swinburne20b]_.

``PyGT.spectral.project()`` returns left and right projections matricies
:math:`{\bf Y,X}` such that the reduction operation is given by

.. math::

	\begin{equation}
		{\bf Q}\in\mathbb{R}^{N\times N}
		\to
		{\bf Y}{\bf Q}{\bf X}\in\mathbb{R}^{N'\times N'}
		,\quad N'\leq N
	\end{equation}

With an error tolerance ``obs_err=0`` we recover the exact solution,
i.e. :math:`N'\to N, {\bf Y}\to\mathbb{I}_N, {\bf X}\to\mathbb{I}_N`

``PyGT.spectral.reduce()`` uses these matricies to produce the reduced rate
matrix and reduced left and right stationary vectors and
initial distribution :math:`\rho` given by

.. math::

	\begin{equation}
		\hat{\pi}\in\mathbb{R}^{N}
		\to
		{\bf Y}\hat{\pi}\in\mathbb{R}^{N'}
		,\quad
		{\bf1}\in\mathbb{R}^{N}
		\to
		{\bf1}{\bf X}\in\mathbb{R}^{N'}
		,\quad
		{\rho}\in\mathbb{R}^{N}
		\to
		{\bf Y}{\rho}\in\mathbb{R}^{N'}
	\end{equation}


.. |br| raw:: html

	<br>

|br|
|br|
|br|


"""

import numpy as np
from io import StringIO
import time,os, importlib
np.set_printoptions(linewidth=160)
from . import GT
from . import tools

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

def reduce(communities,  pi, Q, initial_dist=None, style="specBP",nmoments=2,tol=0.01):
	r"""Returns a reduced rate matrix and initial distribution (optional) using
	``PyGT.spectral.project()``

	Parameters
	----------
	communities : dict
		mapping from community ID (0-indexed) to a boolean array
		of shape (N, ) which selects out the states in that community.
		Communities must be disjoint.

	pi : (N,) array-like

		stationary distribution

	Q : (N, N) array-like

		rate matrix

	initial_dist : (N,) array-like, optional

		Initial probability distribution for first passage time problems, for
		example the local Boltzmann distribtion of a community
		Default=None

	style : string, optional

			Reduction method. Must be one of

				- ``LEA`` : Apply local equilibrum approximation.
				- ``specBP`` : Perform spectral reduction with mode basis modified such that slowest mode becomes the LEA mode, with all other modes orthogonal to this mode but not mutually orthonormal. (Default)
				- ``specBPO`` : Perform spectral reduction with mode basis `rotated` such that slowest mode becomes the LEA mode, with all other modes mutually orthonormal. Gives similar results to ``specBP``.

	nmoments : int, optional

			Number of escape time moments to monitor for accuracy. Ignored if
			``style=LEA``. Higher number implies less reduction in matrix rank.
			Default=2.

	tol : float, optional

			Relative error tolerance for moments. Ignored if ``style=LEA``.
			Default=0.01

	communities : dict
		mapping from community ID (0-indexed) to a boolean array
		of shape (N, ) which selects out the states in that community.
		Communities must be disjoint.

	pi : (N,) array-like
		stationary Boltzmann distribution for entire system

	Q : (N, N) array-like
		rate matrix


	Returns
	-------
		Q: (N', N') array-like
			Reduced rate matrix

		pi: (N',2) array-like
			Reduced left and right stationary distribution vectors for
			calculation of first passage distributions. With no projection
			pi[0] = vector of ones, pi[1] = Boltzmann distribution.

		rho: (N') array-like
			Reduced initial distribution (only if ``initial_dist`` is provided)

	"""
	Y,X = project(communities,pi,Q,style,nmoments,tol)


	if not initial_dist is None:
		return Y@Q@X, [X.sum(axis=0),Y@pi], Y@initial_dist
	else:
		return Y@Q@X, [X.sum(axis=0),Y@pi]



def project(communities,  pi, Q, style="specBP",nmoments=2,tol=0.01):
	r"""Produce a projection matricies in order to reduce a CTMC rate matrix
	from a given community structure.

	Parameters
	----------
	communities : dict
		mapping from community ID (0-indexed) to a boolean array
		of shape (N, ) which selects out the states in that community.
		Communities must be disjoint.

	pi : (N,) array-like

		stationary distribution

	Q : (N, N) array-like

		rate matrix

	style : string, optional

			Reduction method. Must be one of

				- ``LEA`` : Apply local equilibrum approximation.
				- ``specBP`` : Perform spectral reduction with mode basis modified such that slowest mode becomes the LEA mode, with all other modes orthogonal to this mode but not mutually orthonormal. (Default)
				- ``specBPO`` : Perform spectral reduction with mode basis `rotated` such that slowest mode becomes the LEA mode, with all other modes mutually orthonormal. Gives similar results to ``specBP``.

	nmoments : int, optional

			Number of escape time moments to monitor for accuracy. Ignored if
			``style=LEA``. Higher number implies less reduction in matrix rank.
			Default=2.

	tol : float, optional

			Relative error tolerance for moments. Ignored if ``style=LEA``.
			Default=0.01

	Returns
	-------
		Y : (N', N) array-like
			left projection matrix

		X : (N, N') array-like
			right projection matrix

	"""


	new_N = len(communities) #groups.size
	old_N = pi.size

	proj_X = []
	proj_Y = []

	if not style in ["LEA","specBP","specBPO"]:
		print("style must be one of ",style)
		return np.identity(old_N),np.identity(old_N)

	projections=[]

	for i,gid in enumerate(communities.keys()):
		gsel = communities[gid]

		if style=="LEA" or gsel.sum()<2:
			proj_X.append( pi * gsel / pi[gsel].sum())
			proj_Y.append( gsel )
			continue

		# cluster block
		bM = Q[gsel,:][:,gsel].todense()
		brho = pi[gsel] / pi[gsel].sum()


		if style=="specBPO":
			Q = np.identity(gsel.sum()) - np.outer(brho,np.ones(gsel.sum()))

			pnu,pv,pw = tools.eig_wrapper(Q@bM@Q)

			pwv = (pw.sum(axis=0)) * (pv@brho)


		nu,v,w = tools.eig_wrapper(bM)
		wv = (w.sum(axis=0)) * (v@brho)

		ltauv = np.zeros((wv.size,nmodes+2))
		ltauv[:,0] = wv * nu / (wv*nu).sum()
		ltauv[:,1] = wv

		for inm in range(nmodes):
			lv = wv / nu**(inm+1)
			ltauv[:,inm+2] = lv/lv.sum()
		reconst = np.zeros(ltauv.shape[1])


		if obs_err==0.0:
			evorder = np.abs(wv).argsort()[::-1]
		else:
			evorder = np.abs(ltauv.prod(axis=1)).argsort()[::-1]
		if style=="specBPO":
			taken=np.zeros(evorder.size,bool)
			overlap = pv@w * (v@pw).T

		for evind in evorder:
			vecs = np.zeros((old_N,2))

			if style=="specBPO":
				ni = (overlap[:,evind] * (~taken)).argmax()
				taken[ni] = True
				vecs[gsel,0] = pw[:,ni]
				vecs[gsel,1] = pv[ni,:]

			elif style=="specBP":
				if evind == wv.argmax():
					vecs[gsel,0] = brho
					vecs[gsel,1] = np.ones(brho.size)
				else:

					w[:,]

					vecs[gsel,0] = w[:,evind] - w[:,evind].sum()*brho
					vecs[gsel,1] = v[evind,:] - v[evind,:]@brho  * np.ones(brho.size)
			else:
				vecs[gsel,0] = w[:,evind]
				vecs[gsel,1] = v[evind,:]



			reconst += ltauv[evind]
			proj_X.append(vecs[:,0])
			proj_Y.append(vecs[:,1])
			proj_c[i]+=1.0
			if np.abs(reconst-1.0).max()<obs_err:
				#print(reconst)
				break



	proj_X = np.r_[proj_X].T
	proj_Y = np.r_[proj_Y]

	return proj_Y, proj_X
