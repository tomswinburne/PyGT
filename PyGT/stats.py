# -*- coding: utf-8 -*-
r"""
Calculate first passage statistics between macrostates
------------------------------------------------------
Tools to calculate the first passage time distribution
and phenomenological rate constants between endpoint macrostates
:math:`\mathcal{A}` and :math:`\mathcal{B}`.

.. note::

	Install the `pathos` package to parallelize MFPT computations.

"""

import numpy as np
from io import StringIO
import time,os, importlib
#from tqdm import tqdm
np.set_printoptions(linewidth=160)
from . import io as kio
from . import GT
from scipy.sparse import save_npz,load_npz, diags, eye, csr_matrix,bmat
from scipy.sparse.linalg import eigs,inv,spsolve
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


def compute_passage_stats(A_sel, B_sel, pi, K, dopdf=True,rt=None):
	r"""Compute the A->B and B->A first passage time distribution,
	first moment, and second moment using eigendecomposition of a CTMC
	rate matrix.

	Parameters
	----------
	A_sel : (N,) array-like
		boolean array that selects out the A nodes
	B_sel : (N,) array-like
		boolean array that selects out the B nodes
	pi : (N,) array-like
		stationary distribution
	K : (N, N) array-like
		CTMC rate matrix

	dopdf : bool, optional
		Do we calculate full fpt distribution or just the moments. Defaults=True.
	rt: array, optional
		Vector of times to evaluate first passage time distribution in multiples
		of :math:`\left<t\right>` for A->B and B->A. If ``None``, defaults to a logscale
		array from :math:`0.001\left<t\right>` to :math:`1000\left<t\right>`
		in 400 steps, i.e. ``np.logspace(-3,3,400)``.
		Only relevant if ``dopdf=True``

	Returns
	-------
	tau : (4,) array-like
		First and second moments of first passage time distribution for A->B and B->A [:math:`\mathcal{T}_{\mathcal{B}\mathcal{A}}`, :math:`\mathcal{V}_{\mathcal{B}\mathcal{A}}`, :math:`\mathcal{T}_{\mathcal{A}\mathcal{B}}`, :math:`\mathcal{V}_{\mathcal{A}\mathcal{B}}`]
	pt : ( len(rt),4) array-like
		time and first passage time distribution p(t) for A->B and B->A

	"""
	#multiply by negative 1 so eigenvalues are positive instead of negative
	Q=-K
	if rt is None:
		rt = np.logspace(-3,3,400)
	#<tauBA>, <tau^2BA>, <tauAB>, <tau^2AB>
	tau = np.zeros(4)
	if dopdf:
		# time*tau_range, p(t) (first 2: A->B, second 2: B->A)
		pt = np.zeros((4,len(rt)))

	#A -> B
	#P(0) is initialized to local boltzman of source community A
	rho = pi * A_sel
	rho /= rho.sum()
	#B is absorbing, so we want Q in space of A U I
	M = Q[~B_sel,:][:,~B_sel]
	x = spsolve(M,rho[~B_sel])
	y = spsolve(M,x)
	# first moment tau(A->B) = 1.Q^{-1}.rho(A) = 1.x
	tau[0] = x.sum()
	# second moment = 2 x 1.Q^{-2}.rho = 2.0* 1.Q^{-1}.x
	tau[1] = 2.0*y.sum()
	if dopdf:
		#time in multiples of the mean first passage time
		pt[0] = rt*tau[0]
		#nu=eigenvalues, v=left eigenvectors, w=right eigenvectors
		nu,v,w = spla.eig(M.todense(),left=True)
		#normalization factor
		dp = np.sqrt(np.diagonal(w.T.dot(v))).real
		#dot product (v.P(0)=rho)
		v = (v.real.dot(np.diag(1.0/dp))).T.dot(rho[~B_sel])
		#dot product (1.T.w)
		w = (w.real.dot(np.diag(1.0/dp))).sum(axis=0)
		nu = nu.real
		#(v*w/nu).sum() is the same as <tau>, the first bit is the pdf p(t)
		pt[1] = (v*w*nu)@np.exp(-np.outer(nu,pt[0]))*(v*w/nu).sum()

	#B -> A
	rho = pi * B_sel
	rho /= rho.sum()
	M = Q[~A_sel,:][:,~A_sel]
	x = spsolve(M,rho[~A_sel])
	y = spsolve(M,x)
	tau[2] = x.sum()
	tau[3] = 2.0*y.sum()
	if dopdf:
		pt[2] = rt*tau[2]
		nu,v,w = spla.eig(M.todense(),left=True)
		dp = np.sqrt(np.diagonal(w.T.dot(v))).real
		v = (v.real.dot(np.diag(1.0/dp))).T.dot(rho[~A_sel])
		w = (w.real.dot(np.diag(1.0/dp))).sum(axis=0)
		nu = nu.real
		pt[3] = (v*w*nu)@np.exp(-np.outer(nu,pt[2]))*(v*w/nu).sum()
		return tau, pt.T
	else:
		return tau

def compute_escape_stats(B_sel, pi, K, tau_escape=None, dopdf=True,rt=None):
	r"""Compute escape time distribution and first and second moment
	from the basin specified by `B_sel` using eigendecomposition.

	Parameters
	----------
	B_sel : (N,) array-like
		boolean array that selects out the nodes in the active basin
	pi : (N,) array-like
		stationary distribution for CTMC
	K : (N, N) array-like
		CTMC rate matrix
	tau_escape : float
		mean time to escape from B. Used to calculate the escape
		time distribution in multiple of tau_escape (p(t/tau_escape).
		If None, uses the first moment in network defined by K.
	dopdf : bool
		whether to calculate full escape time distribution, defaults to True
	rt: array, optional
		Vector of times to evaluate first passage time distribution in multiples
		of :math:`\left<t\right>` for A->B and B->A. If ``None``, defaults to a logscale
		array from :math:`0.001\left<t\right>` to :math:`1000\left<t\right>`
		in 400 steps, i.e. ``np.logspace(-3,3,400)``.
		Only relevant if ``dopdf=True``

	Returns
	-------
	tau : (2,) array-like
		First and second moments of escape time distribution, [:math:`\left<t\right>_{\mathcal{B}}`, :math:`\left<t^2 \right>_{\mathcal{B}}`]
	pt : (2, len(rt)) array-like
		time and escape time distribution :math:`p(t)\left<t\right>`

	"""
	#multiply by negative 1 so eigenvalues are positive instead of negative
	Q=-K
	if rt is None:
		rt = np.logspace(-3,3,400)

	#<tau>, <tau^2>
	tau = np.zeros(2)
	if dopdf:
		# time*tau_range, p(t)
		pt = np.zeros((2, len(rt)))
	rho = pi * B_sel
	rho /= rho.sum()
	M = Q[B_sel,:][:,B_sel]
	x = spsolve(M,rho[B_sel])
	y = spsolve(M,x)
	tau[0] = x.sum()
	tau[1] = 2.0*y.sum()
	if tau_escape is None:
		tau_escape = tau[0]
	if dopdf:
		pt[0] = rt*tau_escape
		nu,v,w = spla.eig(M.todense(),left=True)
		dp = np.sqrt(np.diagonal(w.T.dot(v))).real
		v = (v.real.dot(np.diag(1.0/dp))).T.dot(rho[B_sel])
		w = (w.real.dot(np.diag(1.0/dp))).sum(axis=0)
		nu = nu.real
		pt[1] = (v*w*nu)@np.exp(-np.outer(nu,pt[0]))*tau_escape
		return tau, pt
	else:
		return tau

def compute_rates(A_sel, B_sel, B, tau, pi, initA=None, initB=None, MFPTonly=True, fullGT=False,
	pool_size=None, block=1, screen=False, **kwargs):
	r"""
	Calculate kSS, kNSS, kF, k*, kQSD, and MFPT for the transition path
	ensemble A_sel --> B_sel from rate matrix K. K can be the matrix of an original
	Markov chain, or a partially graph-transformed Markov chain. [Wales09]_

	Differs from ``compute_passage_stats()`` in that this function removes all intervening states
	using GT before computing FPT stats and rates on the fully reduced network
	with state space :math:`(\mathcal{A} \cup \mathcal{B})`. This implementation also does not rely on a full
	eigendecomposition of the non-absorbing matrix; it instead performs a matrix inversion,
	or if `fullGT` is specified, all nodes in the set :math:`(\mathcal{A} \cup b)^\mathsf{c}`
	are removed using GT for each :math:`b \in \mathcal{B}` so that the
	MFPT is given by an average:

	.. math::

		\begin{equation}
		\mathcal{T}_{\mathcal{A}\mathcal{B}} = \frac{1}{\sum_{b \in
		\mathcal{B}} p_b(0)} \sum_{b \in \mathcal{B}} \frac{p_b(0)
		\tau^\prime_b}{1-P^\prime_{bb}}
		\end{equation}

	If the MFPT is less than :math:`10^{20}`, `fullGT` should not be needed
	since the inversion of the non-absorbing matrix should be numerically
	stable. However, a condition number check is performed regardless, which
	forces a full GT if the MFPT problem is considered numerically unstable.
	

	Parameters
	----------
	A_sel : array-like (N,)
		selects the N_A nodes in the :math:`\mathcal{A}` set.
	B_sel : array-like (N,)
		selects the N_B nodes in the :math:`\mathcal{B}` set.
	B : sparse or dense matrix (N,N)
		branching probability matrix for CTMC
	tau : array-like (N,)
		vector of waiting times from each node.
	pi : array-like (N,)
		stationary distribution of CTMC
	initA : array-like (N_A,), optional
		normalized initial occupation probabilities in :math:`\mathcal{A}` set.
		Default= local Boltzmann distribution
	initB : array-like (N_B,), optional
		normalized initial occupation probabilities in :math:`\mathcal{B}` set.
		Default= local Boltzmann distribution
	MFPTonly : bool
		If True, only MFPTs are calculated (rate calculations ignored).
	fullGT : bool
		If True, all source nodes are isolated with GT to obtain the average
		MFPT.
	pool_size : int
		Number of cores over which to parallelize fullGT computation.

	Returns
	-------
	results: dictionary
		dictionary of results, with keys
		'MFPTAB', 'kSSAB', 'kNSSAB', 'kQSDAB', 'k*AB', 'kFAB' for A<-B and
		'MFPTBA', 'kSSBA', 'kNSSBA', 'kQSDBA', 'k*BA', 'kFBA' for B<-A
		such that
		results['MFPTAB'] = mean first passage time for A<-B
	"""

	N = len(A_sel)
	assert(N==len(B_sel))

	if A_sel.sum()==1 and B_sel.sum()==1:
		raise NotImplementedError('There must be at least 2 microstates in A and B.')


	inter_region = ~(A_sel+B_sel)
	r_AS = A_sel[~inter_region]
	r_BS = B_sel[~inter_region]
	rDSS = 1.0/tau[~inter_region]


	if initA is None:
		initA = pi[A_sel]/pi[A_sel].sum()
	if initB is None:
		initB = pi[B_sel]/pi[B_sel].sum()

	#use GT to renormalize away all I states

	rB, rtau, rQ = GT.blockGT(rm_vec=inter_region,B=B,tau=tau,rates=True,block=block,**kwargs)
	rD = 1.0/rtau
	rN = rtau.size
	rQ = -rQ #multiply by -1 so eigenvalues are positive

	#first do A->B direction, then B->A
	#r_s is the non-absorbing region (A first, then B)
	df = {}
	dirs = ['BA', 'AB']
	inits = [initA, initB]



	for i, r_s in enumerate([r_AS, r_BS]) :

		if has_tqdm and screen:
			pbar = tqdm(total=r_s.sum(),leave=True,mininterval=0.0,desc='Rates')

		#local equilibrium distribution in r_s
		rho = inits[i]
		#MFPTs to B from each source microstate a
		T_Ba = np.zeros(r_s.sum())

		if not fullGT:
			fullGT = bool(np.linalg.cond(rQ[r_s,:][:,r_s])>1.0e12)

		if not fullGT:
			#MFPT calculation via matrix inversion
			invQ = spla.inv(rQ[r_s,:][:,r_s])
			mfpt = invQ.dot(rho).sum(axis=0)
			T_Ba = invQ.sum(axis=0)
		#compare to individual T_Ba quantities from further GT compression
		else:
			def disconnect_sources(s):
				#disconnect all source nodes except for `s`
				rm_reg = np.zeros(rN, bool)
				rm_reg[r_s] = True
				aind = r_s.nonzero()[0][s]
				#print(f'Disconnecting source node {aind}')
				rm_reg[r_s.nonzero()[0][s]] = False

				rfB, tau_Fs = GT.blockGT(rm_vec=rm_reg,B=rB,tau=1.0/rD,rates=False,block=block,screen=screen,Ndense=1)
				#escape time tau_F
				rfD = 1./tau_Fs
				rfN = tau_Fs.size
				#remaining network only as 1 in A and 1 in B = 2 states
				rf_s = r_s[~rm_reg]
				#tau_a^F / P_Ba^F
				P_Ba = np.ravel(rfB[~rf_s,:][:,rf_s].sum(axis=0))[0]
				if screen and has_tqdm:
					if have_pathos and pool_size is not None:
						pbar.update(pool_size)
					else:
						pbar.update(1)
				return tau_Fs[rf_s][0]/P_Ba
			if have_pathos and pool_size is not None:
				with Pool(processes=pool_size) as p:
					T_Ba = p.map(disconnect_sources, [s for s in range(r_s.sum())])
			else:
				for s in range(r_s.sum()):
					T_Ba[s] = disconnect_sources(s)

			#MFPT_BA = (T_Ba@rho)
			mfpt = T_Ba@rho
		df[f'MFPT{dirs[i]}'] = mfpt

		"""
			Rates: SS, NSS, QSD, k*, kF
		"""
		if not MFPTonly:
			#eigendecomposition of rate matrix in non-abosrbing region
			#for A, full_RK[r_A, :][:, r_A] is just a 5x5 matrix
			l, v = spla.eig(rQ[r_s,:][:,r_s])
			#order the eigenvalues from smallest to largest -- they are positive since Q = D - K instead of K-D
			qsdo = np.abs(l.real).argsort()
			nu = l.real[qsdo]
			#v[:, qsdo[0]] is the eigenvector corresponding to smallest eigenvalue
			#aka quasi=stationary distribution
			qsd = v[:,qsdo[0]]
			qsd /= qsd.sum()
			#committor C^B_A: probability of reaching B before A: 1_B.B_BA^I (eqn 6 of SwinburneW20)
			C = np.ravel(rB[~r_s,:][:,r_s].sum(axis=0))
			#for SS, we use waiting times from non-reduced network (DSS)
			df[f'kSS{dirs[i]}'] = C.dot(np.diag(rDSS[r_s])).dot(rho)
			#for NSS, we use waiting times D^I_s from reduced network
			df[f'kNSS{dirs[i]}'] = C.dot(np.diag(rD[r_s])).dot(rho)
			#kQSD is same as NSS except using qsd instead of boltzmann
			df[f'kQSD{dirs[i]}'] = C.dot(np.diag(rD[r_s])).dot(qsd)
			#k* is just 1/MFPT
			df[f'k*{dirs[i]}'] = 1./mfpt
			#and kF is <1/T_Ab>
			df[f'kF{dirs[i]}'] = (rho/T_Ba).sum()

	return df
