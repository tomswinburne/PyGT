r"""
Optimal Markovian coarse-graining for a given community structure
-----------------------------------------------------------------

Various functions to analyze Markov chains, including
estimating the optimal coarse-grained CTMC for a given community structure.

"""
from scipy.sparse import diags, issparse, csgraph, csr_matrix
import scipy.linalg as spla
import numpy as np

from . import mfpt as fpt

import numpy as np
from numpy.linalg import inv
import scipy
import scipy.linalg as spla
from scipy.sparse import diags, issparse, csgraph, csr_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eig, expm
from pathlib import Path

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
	tau : np.ndarray[float64] (nnodes,)
		Vector of waiting times of nodes, used as input to GT

	"""
	#check that columns of K sum to zero
	assert(np.all(K.sum(axis=0) < 1.E-10))
	Q = K - np.diag(np.diag(K))
	escape_rates = -1*np.diag(K)
	tau = 1./escape_rates
	B = Q@np.diag(1./escape_rates)
	return B, tau

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
	tau : np.ndarray[float64] (nnodes,)
		Vector of waiting times of nodes, used as input to GT

	"""

	nnodes = T.shape[0]
	#check that T is column-stochastic
	assert(np.all(np.abs(np.ones(nnodes) - T.sum(axis=0))<1.E-10))
	tau = np.tile(tau_lag, nnodes)
	B=T
	return B, tau

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



class Analyze_KTN(object):
	r""" Estimate a coarse-grained continuous-time Markov chain
	given a partiioning :math:`\mathcal{C} = \{I, J, ...\}` of the :math:`V` nodes into :math:`N<V` communities.
	Various formulations for the inter-community rates are implemented, including the local
	equilibrium approximation, Hummer-Szabo relation, and other routes to obtain the optimal
	coarse-grained Markov chain for a given community structure. [Kannan20a]_

	Attributes
	----------
	path : str or Path object
		path to directory with all relevant files
	K : array-like (nnodes, nnodes)
		Rate matrix with elements :math:`K_{ij}` corresponding to the `i \leftarrow j` transition rate,
		and diagonal elements :math:`K_{ii} = -\sum_\gamma K_{\gamma i}` such that the columns sum to zero.
	pi : array-like (nnodes,)
		Stationary distribution of nodes, :math:`\pi_i`, i.e. vector of equilibrium occupation probabilities.
	commpi : array-like (ncomms,)
		Stationary distribution of communities, :math:`\Pi_J = \sum_{j \in J} \pi_j`.
	communities : dict
		dictionary mapping community IDs (1-indexed) to node IDs (1-indexed).
	commdata : str
		Filename, located in directory specified by `path`, of a single-column file where
		each line contains the community ID (0-indexed) of the node specified
		by the line number in the file.

	Note
	----
	Either `communities` or `commdata` must be specified.

	"""

	def __init__(self, path, K=None, pi=None, commpi=None, communities=None,
				 commdata=None):
		self.path = Path(path)
		self.K = K
		self.pi = pi
		self.commpi = commpi
		if communities is not None:
			self.communities = communities
		elif commdata is not None:
			self.communities = self.read_communities(self.path/commdata)
		else:
			raise AttributeError('Either communities or commdata must' \
								'be specified.')
		if K is not None and pi is not None:
			commpi = self.get_comm_stat_probs(pi, log=False)
			self.commpi = commpi

	def construct_coarse_rate_matrix_LEA(self):
		"""Calculate the coarse-grained rate matrix obtained using the local
		equilibrium approximation (LEA)."""

		if self.K is None:
			raise ValueError("The attributes K, pi, and commpi must be specified.")

		N = len(self.communities)
		Rlea = np.zeros((N,N))

		for i in range(N):
			for j in range(N):
				if i < j:
					ci = np.array(self.communities[i+1]) - 1
					cj = np.array(self.communities[j+1]) - 1
					Rlea[i, j] = np.sum(self.K[np.ix_(ci, cj)]@self.pi[cj]) / self.commpi[j]
					Rlea[j, i] = np.sum(self.K[np.ix_(cj, ci)]@self.pi[ci]) / self.commpi[i]

		for i in range(N):
			Rlea[i, i] = -np.sum(Rlea[:, i])
		return Rlea

	def construct_coarse_rate_matrix_Hummer_Szabo(self):
		r""" Calculate the optimal coarse-grained rate matrix using the Hummer-Szabo
		relation, aka Eqn. (12) in Hummer & Szabo *J. Phys. Chem. B.* (2015)."""

		if self.K is None:
			raise ValueError("The attributes K, pi, and commpi must be specified.")

		N = len(self.communities)
		V = len(self.pi)
		D_N = np.diag(self.commpi)
		D_V = np.diag(self.pi)
		#construct clustering matrix M from community assignments
		M = np.zeros((V, N))
		for ci in self.communities:
			col = np.zeros((V,))
			comm_idxs = np.array(self.communities[ci]) - 1
			col[comm_idxs] = 1.0
			M[:, ci-1] = col

		Pi_col = self.commpi.reshape((N, 1))
		pi_col = self.pi.reshape((V, 1))
		mat_to_invert = pi_col@np.ones((1,V)) - self.K
		first_inverse = spla.inv(mat_to_invert)
		#check that Pi = M^T pi
		Pi_calc = M.T@self.pi
		for entry in np.abs(self.commpi - Pi_calc):
			assert(entry < 1.0E-10)

		#H-S relation
		second_inversion = spla.inv(M.T@first_inverse@D_V@M)
		R_HS = Pi_col@np.ones((1,N)) - D_N@second_inversion
		if not check_detailed_balance(self.commpi, R_HS):
			print(f'HS does not satisfy detailed balance')
		return R_HS

	def construct_coarse_rate_matrix_KKRA(self, mfpt=None, GT=False, **kwargs):
		r"""Calculate optimal coarse-grained rate matrix using Eqn. (79)
		of Kells et al. *J. Chem. Phys.* (2020), aka the KKRA expression
		in Eqn. (10) of  [Kannan20a]_.

		Parameters
		----------
		mfpt : (nnodes, nnodes)
			Matrix of inter-microstate MFPTs between all pairs of nodes. Defaults to None.
		GT : bool
			If True, matrix of inter-microstate MFPTs is computed with GT.
			Kwargs can then be specified for GT (such as the pool_size for parallelization).    Defaults to False.

		"""

		if self.K is None:
			raise ValueError("The attributes K, pi, and commpi must be specified.")
		N = len(self.communities)
		n = len(self.pi)
		D_N = np.diag(self.commpi)
		D_n = np.diag(self.pi)

		#construct clustering matrix M from community assignments
		M = np.zeros((n, N))
		for ci in self.communities:
			col = np.zeros((n,))
			comm_idxs = np.array(self.communities[ci]) - 1
			col[comm_idxs] = 1.0
			M[:, ci-1] = col

		Pi_col = self.commpi.reshape((N, 1))
		pi_col = self.pi.reshape((n, 1))
		if GT:
			B, tau = load_CTMC(self.K)
			mfpt = fpt.full_MFPT_matrix(B, tau, **kwargs)
		elif mfpt is None:
			mfpt = self.get_intermicrostate_mfpts_linear_solve()
		R = Pi_col@np.ones((1,N)) - D_N@spla.inv(Pi_col@Pi_col.T +
												 M.T@D_n@mfpt@pi_col@Pi_col.T -
												 M.T@D_n@mfpt@D_n@M)
		if not check_detailed_balance(self.commpi, R):
			print(f'KKRA does not satisfy detailed balance')
		return R

	def get_intermicrostate_mfpts_linear_solve(self):
		r"""Calculate the matrix of inter-microstate MFPTs between all pairs of nodes
		by solving a system of linear equations given by Eq.(8) of
		[Kannan20a]_."""

		K = self.K
		n = K.shape[0]
		mfpt = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				if i==j:
					mfpt[i][j] = 0.
				else:
					try:
						mfpt[i][j] = -spla.solve(K[np.arange(n)!=i, :][:, np.arange(n)!=i],
												(np.arange(n)==j)[np.arange(n)!=i]).sum()
					except scipy.linalg.LinAlgWarning as err:
						raise Exception('LinAlgWarning')
		return mfpt

	def get_intermicrostate_mfpts_fundamental_matrix(self):
		r"""Calculate the matrix of inter-microstate MFPTs between all pairs of nodes
		using Eq. (6) of [Kannan20a]_. """

		K = self.K
		pi = self.pi
		nmin = K.shape[0]
		pioneK = spla.inv(pi.reshape((nmin,1))@np.ones((1,nmin)) + K)
		zvec = np.diag(pioneK)
		mfpt = np.diag(1./pi)@(pioneK - zvec.reshape((nmin,1))@np.ones((1,nmin)))
		return mfpt

	def get_intercommunity_MFPTs_linear_solve(self):
		r"""Calculate the true MFPTs between communities by inverting the non-absorbing
		rate matrix. Equivalent to Eqn. (14) in Swinbourne & Wales *JCTC* (2020)."""

		K = ktn.K
		pi = ktn.pi
		n = K.shape[0]
		N = len(self.communities)
		mfpt = np.zeros((N,N))
		for i in range(N):
			for j in range(N):
				if i==j:
					mfpt[i][j] = 0.
				else:
					ci = np.array(self.communities[i+1]) - 1
					cj = np.array(self.communities[j+1]) - 1
					cond = np.ones((n,), dtype=bool)
					cond[ci] = False
					initial_cond = np.zeros((n,))
					#initialize probability density to local boltzman in cj
					initial_cond[cj] = pi[cj]/pi[cj].sum()
					mfpt[i][j] = -spla.solve(K[cond, :][:, cond],
											initial_cond[cond]).sum()
		return mfpt

	def get_intercommunity_weighted_MFPTs(self, mfpt, diagzero=True):
		r"""Comppute the matrix :math:`\widetilde{\mathcal{T}}_{\rm C}` of appropriately weighted
		inter-community MFPTs, as defined in Eq. (18) in [Kannan20a]_.

		Parameters
		----------
		mfpt : array-like (N,N)
			matrix of intermicrostate MFPTs.
		diagzero : bool
			Whether to define the inter-community weighted MFPTs such as the diagonal elements are zero. Defaults to True.

		"""

		pi = self.pi
		commpi = self.commpi
		N = len(self.communities)
		tJI = np.zeros((N,N))
		for i in range(N):
			for j in range(N):
				ci = np.array(self.communities[i+1]) - 1
				cj = np.array(self.communities[j+1]) - 1
				tJI[j][i] = pi[cj]@ mfpt[cj,:][:,ci] @ pi[ci] / (commpi[i]*commpi[j])
				if diagzero:
					tJI[j][i] -= pi[cj] @ mfpt[cj,:][:,cj] @ pi[cj] / (commpi[j])**2

		return tJI

	def get_timescale_error(self, m, K, R):
		""" Calculate the ith timescale error for i in {1,2,...m} of a
		coarse-grained rate matrix R compared to the full matrix K.

		Parameters
		----------
		m : int
			Number of dominant eigenvalues (m < N)
		K : np.ndarray[float] (V, V)
			Rate matrix for full network
		R : np.ndarray[float] (N, N)
			Coarse-grained rate matrix

		Returns
		-------
		timescale_errors : np.ndarray[float] (m-1,)
			Errors for m-1 slowest timescales

		"""

		ncomms = len(self.communities)
		if m >= ncomms:
			raise ValueError('The number of dominant eigenvectors must be' \
							 'less than the number of communities.')
		Kevals, Kevecs = calc_eigenvectors(K, m, which_eig='SM')
		Revals, Revecs = calc_eigenvectors(R, m, which_eig='SM')
		#the 0th eigenvalue corresponds to infinite time
		Ktimescales = -1./Kevals[1:]
		Rtimescales = -1./Revals[1:]
		timescale_errors = np.abs(Rtimescales - Ktimescales)
		return timescale_errors

	def get_eigenfunction_error(self, m, K, R):
		r""" Calculate the :math:`i^{\rm th}` eigenvector approximation error for :math:`i \in {1, 2,
		... m}` of a coarse-grained rate matrix `R` by comparing its eigenvector
		to the correspcorresponding eigenvector of the full matrix.
		"""

		ncomms = len(self.communities)
		if m >= ncomms:
			raise ValueError('The number of dominant eigenvectors must be' \
							 'less than the number of communities.')
		Kevals, Kevecs = calc_eigenvectors(K, m, which_eig='SM')
		Revals, Revecs = calc_eigenvectors(R, m, which_eig='SM')
		errors = np.zeros((m,))
		for i in range(0, m):
			print(i)
			for ck in self.communities:
				#minima in community ck
				minima = np.array(self.communities[ck]) - 1
				coarse_evec = np.tile(Revecs[i, ck-1], len(minima)) #scalar
				errors[i] += np.linalg.norm(coarse_evec - Kevecs[i, minima])
		return errors

	def get_comm_stat_probs(self, pi, log=False):
		""" Calculate the community stationary probabilities by summing over
		the stationary probabilities of the nodes in each community.

		Parameters
		----------
		pi : list (nnodes,)
			stationary probabilities of node in original Markov chain

		Returns
		-------
		commpi : list (ncomms,)
			stationary probabilities of communities in coarse coarse_network

		"""

		#pi = np.exp(logpi)
		if (np.sum(pi) - 1.0) > 1.E-10:
			pi = pi/np.sum(pi)
		ncomms = len(self.communities)
		commpi = np.zeros((ncomms,))
		for ci in self.communities:
			#zero-indexed list of minima in community ci
			nodelist = np.array(self.communities[ci]) - 1
			#logcommpi[ci-1] = -np.inf
			commpi[ci-1] = 0
			for node in nodelist:
				commpi[ci-1] += pi[node]
				#logcommpi[ci-1] = np.log(np.exp(logcommpi[ci-1]) + np.exp(logpi[node]))
		#commpi = np.exp(logcommpi)
		logcommpi = np.log(commpi)
		assert abs(np.sum(commpi) - 1.0) < 1.E-10
		if log:
			return logcommpi
		else:
			return commpi

	def read_communities(self, commdat):
		"""Read in a single column file called communities.dat where each line
		is the community ID (zero-indexed) of the minima given by the line
		number.

		Parameters
		----------
		commdat : .dat file
			single-column file containing community IDs of each minimum

		Returns
		-------
		communities : dict
			mapping from community ID (1-indexed) to minima ID (1-indexed)
		"""

		communities = {}
		with open(commdat, 'r') as f:
			for minID, line in enumerate(f, 1):
				groupID =  int(line) + 1
				if groupID in communities:
					communities[groupID].append(minID)
				else:
					communities[groupID] = [minID]
		return communities
