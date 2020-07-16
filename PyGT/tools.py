r"""
Various functions to analyze transition matricies
"""
from scipy.sparse import diags, issparse, csgraph, csr_matrix
import scipy.linalg as spla
import numpy as np


def choose_nodes_to_remove(rm_region, pi, tau, style='free_energy', percent_retained=50):
	"""
	Given a branching probability matrix, stationary distribution and waiting times
	of a CTMC, return a Boolean array selecting nodes from a given subset to remove
	by graph transformation according to some simple criteria. [Kannan20b]_

	Parameters
	----------
	rm_region : (N,) array
		boolean array that specifies region in which nodes can be removed
		i.e. for A<->B dynamics, all nodes in A,B should be retained.

	pi: (N,) array
		stationary distribution

	tau: (N,) array
		vector of waiting times

	B: sparse matrix, optional
		sparse matrix of branching (CTMC) or transition (DTMC) probabilities,
		used for `style='node_degree'`

	style : str, optional
		ranking used to remove nodes, from high to low. Highest percentile is removed
		'escape_time' : (`=tau`)ascending,

		'free_energy' : (`=-log|pi|`), descending,

		'hybrid' : remove nodes in highest percentile in `escape_time` and `free_energy`

		'combined' : `tau * pi`, ascending,

		'node_degree': descending (requires `B`),

		Default = `free_energy`

	percent_retained : float, optional
		percent of nodes to keep in reduced network. Default = 50.0



	Returns
	-------
	rm_vec : (N,) array
		boolean array that specifies nodes to remove, which will always be a
		subset of the nodes selected by `rm_region`

	"""
	to_remove = np.zeros(rm_region.size,np.bool)

	if style == 'node_degree':
		if issparse(B):
			B = csr_matrix(B)
			order = (B.indptr[1:]-B.indptr[:-1])[rm_region]
			to_remove[rm_region] = order < np.percentile(order,100.0-percent_retained)
		else:
			style='free_energy'

	elif style == 'escape_time':
		order = tau[rm_region]
		to_remove[rm_region] = order < np.percentile(order,100.0-percent_retained)

	elif style == 'free_energy':
		order = -np.log(pi)[rm_region]
		to_remove[rm_region] = order > np.percentile(order,percent_retained)

	elif style == 'hybrid':
		order = tau[rm_region]
		to_remove[rm_region] = order < np.percentile(order,100.0-percent_retained)

		order = -np.log(pi)[rm_region]
		to_remove[rm_region] *= order > np.percentile(order,percent_retained)

	elif style == 'combined':
		order = tau[rm_region]*pi[rm_region]
		to_remove[rm_region] = order < np.percentile(order,100.0-percent_retained)

	else:
		raise ValueError('Invalid Style: ["escape_time","free_energy","hybrid","combined","node_degree"]')

	return to_remove

def check_detailed_balance(pi, K):
	""" Check if Markov chain satisfies detailed balance condition,
	:math:`k_{ij} \pi_j = k_{ji} \pi_i` for all :math:`i,j`.

	Parameters
	----------
	pi : array-like (N,)
		stationary probabilities of full or reduced system
	K : sparse or dense matrix (N, N)
		transition rate matrix of full or reduced system

	Returns
	-------
	success, bool
		Self-explanatory
	"""
	if issparse(K):
		sM = K@diags(pi)
		sM -= sM.transpose()
		sM.eliminate_zeros()
		return np.abs(sM.data).max()<1.E-10
	else:
		sM = K@np.diag(pi)
		sM -= sM.transpose()
		return np.abs(sM).max()<1.E-10

def make_fastest_path(G,i,f,depth=1,limit=None):
	r"""
	Wrapper for `scipy.sparse.csgraph.shortest_path` which returns node indicies
	on as-determined shortest i->f path and those `depth` connections away.
	Used to determine which nodes to remove by graph transformation during
	sensitivity analysis applied to kinetic transition networks [Swinburne20a]_.



	Parameters
	----------
	B:		(N,N) sparse matrix
			Matrix that will be interpreted as weighted graph for path calculation

	i:		int
			Initial node index

	f:		int
			Initial node index

	depth:	int, (default=1)
			Size of near-path region

	Returns
	-------
	path:		array_like
				indices of path nodes
	path_region:array_like
				indices of near-path nodes
	"""

	d, cspath = csgraph.shortest_path(csgraph=G, indices=[f,i],\
									directed=False, method='D', return_predecessors=True)
	path = [i]
	while path[-1] != f:
		path.append(cspath[0][path[-1]])

	N = G.shape[0]
	path_region = np.zeros(N,bool)

	G=G.tocsc()
	for path_ind in path:
		path_region[path_ind] = True
	for scan in range(depth):
		indscan = np.arange(N)[path_region]
		for path_ind in indscan:
			path_region[G[:,path_ind].indices[G[:,path_ind].data.argsort()[:limit]]] = True
	return path, path_region

def check_kemeny(pi,tauM):
	r"""
	Check that Markov chain satisfies the Kemeny constant relation,
	:math:`\sum_i\pi_i, \mathcal{T}_{ij} = \xi` for all :math:`j`,
	where :math:`\xi` is a constant and :math:`\mathcal{T}_{ij}` is
	the j->i mean first passage time.

	Parameters
	----------
	pi : array-like (N,)
		stationary probabilities of full or reduced system
	tauM : sparse or dense matrix (N, N)
		mean first passage time matrix of full or reduced system

	Returns
	-------
	kemeny_constant, float
		Average kemeny constant :math:`mean(xi)` across nodes
	success, bool
		Self-explanatory. False if :math:`std(xi)/mean(xi)>1e-9`
	"""
	xi = tauM.transpose()@pi / pi.sum()
	return xi.mean(), xi.std()<1.0E-9 * xi.mean()


def load_CTMC(K):
	r""" Setup a GT calculation for a transition rate matrix representing
	a continuous-time Markov chain.

	Parameters
	----------
	K : array-like (nnodes, nnodes)
		Rate matrix with elements :math:`K_{ij}` corresponding to the :math:`i \leftarrow j` transition rate
		and diagonal elements :math:`K_{ii} = \sum_\gamma K_{\gamma i}` such that the columns of :math:`\textbf{K}` sum to zero.

	Returns
	-------
	B : np.ndarray[float64] (nnodes, nnodes)
		Branching probability matrix in dense format, used as input to GT
	escape_rates : np.ndarray[float64] (nnodes,)
		Array of inverse waiting times of nodes, used as input to GT

	"""
	#check that columns of K sum to zero
	assert(np.all(K.sum(axis=0) < 1.E-10))
	Q = K - np.diag(np.diag(K))
	escape_rates = -1*np.diag(K)
	B = Q@np.diag(1./escape_rates)
	return B, escape_rates

def load_DTMC(T, tau_lag):
	r""" Setup a GT calculation for a transition probability matrix representing
	a discrete-time Markov chain.

	Parameters
	----------
	T : array-like (nnodes, nnodes)
		Discrete-time, column-stochastic transition probability matrix.
	tau_lag : float
		Lag time at which `T` was estimated.

	Returns
	-------
	B : np.ndarray[float64] (nnodes, nnodes)
		Branching probability matrix in dense format, used as input to GT
	escape_rates : np.ndarray[float64] (nnodes,)
		Array of inverse waiting times of nodes, used as input to GT

	"""

	nnodes = T.shape[0]
	#check that T is column-stochastic
	assert(np.all(np.abs(np.ones(nnodes) - T.sum(axis=0))<1.E-10))
	escape_rates = np.tile(1./tau_lag, nnodes)
	B=T
	return B, escape_rates

def eig_wrapper(M):
	r"""Wrapper of ``scipy.linalg.eig`` that returns real eigenvalues and
	orthonormal left and right eigenvector pairs

	Parameters
	----------
	M : (N,N) dense matrix

	Returns
	-------
	nu : (N,) array-like
		Real component of eigenvalues
	v : (N,N) array-like
		Matrix of left eigenvectors
	w : (N,N) array-like
		Matrix of right eigenvectors

	"""
	nu,v,w = spla.eig(M,left=True)
    dp = np.diag(1.0/np.sqrt(np.diag((w.T@v).real)))
    nu,v,w = nu.real, (v.real@dp).T, w.real@dp
    return nu,v,w
