r"""
Various functions to analyze transition matricies
"""
from scipy.sparse import diags, issparse, csgraph
import numpy as np


def check_detailed_balance(pi, K):
	""" Check if Markov chain satisfies detailed balance condition,
	:math:`k_{ij} \pi_j = k_{ji} \pi_i` for all :math:`i,j`.

	Parameters
	----------
	pi : array-like (nnodes,) or (ncomms,)
		stationary probabilities
	K : array-like (nnodes, nnodes) or (ncomms, ncomms)
		transition rate matrix. can be sparse or dense

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
	sensitivity analysis applied to kinetic transition networks [Swinburne20]_.



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
