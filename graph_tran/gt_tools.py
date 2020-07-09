# -*- coding: utf-8 -*-
r"""
Iteratively remove nodes from a Markov chain with graph transformation
----------------------------------------------------------------------
Implements the core GT algorithm for eliminating nodes from a Markov chain
in blocks, or one at a time. For use with the `graph_tran.ktn_io` module,
which reads the input files describing the original Markov chain.

The graph transformation algorithm requires a branching probability matrix
:math:`\textbf{B}` with elements :math:`B_{ij}=k_{ij}{\tau}_j` where :math:`k_{ij}` is
the :math:`i \leftarrow j` inter-microstate transition rate and :math:`\tau_j`
is the mean waiting time of node :math:`j`. In a discrete-time Markov chain,
:math:`\textbf{B}` is replaced with the discrete-time transition probability
matrix :math:`\textbf{T}(\tau)` and the waiting times of all nodes are uniform,
equal to :math:`\tau`.

In each iteration of GT, a single node :math:`x` is removed, and the branching
probabilities and waiting times of the neighboring nodes are updated according
to

.. math::

    \begin{eqnarray}
        B_{ij}^{\prime} &\leftarrow B_{ij} + \frac{B_{ix}B_{xj}}{1-B_{xx}} \\
        \tau_j^\prime  &\leftarrow \tau_j + \frac{B_{xj}\tau_j}{1-B_{xx}}
    \end{eqnarray}

A matrix version of the above equations permits the removal of blocks of nodes
simulatenously. [3]_ In practice, the larger the block of nodes, the less
numerically stable the block-GT operation will be.

.. note::

	Install the `tqdm` package for progress bars.

.. [3] T. D. Swinburne, D. J. Wales, *JCTC* (2020)

"""

import os,time
import numpy as np
os.system('mkdir -p cache')
os.system('mkdir -p output')

from scipy.sparse import csgraph, csr_matrix, lil_matrix,eye, save_npz, load_npz, diags,kron

class timer:
	def __init__(self):
		self.t = time.time()
	def __call__(self,str=None):
		t = time.time() - self.t
		self.t = time.time()
		if not str is None:
			print(str,":",t)
		else:
			return t


""" test for custom umfpack_hack
try:
	from linalgcond import spsolvecond
	try:
		b = np.ones(4)
		A = diags(np.r_[[0.1,0.3,0.4,0.5]],format="csr")
		x,cond = spsolvecond(A,b,giveCond=True)
		print("Using custom UMFPACK with condition number calculation")
		has_umfpack_hack=True
	except TypeError or ImportError:
		from scipy.sparse.linalg import spsolve
		print("Using standard UMFPACK, no condition number calculation")
		has_umfpack_hack=False
except ModuleNotFoundError:
	from scipy.sparse.linalg import spsolve,inv
	print("Using standard UMFPACK, no condition number calculation")
	has_umfpack_hack=False
"""
from scipy.sparse.linalg import spsolve,inv

""" test for tqdm progress bars """
try:
	from tqdm import tqdm
	print("Using tqdm package for pretty progress bars!")
	has_tqdm=True
except:
	print("Install tqdm package for pretty progress bars!")
	has_tqdm=False



def direct_solve(B,initial_states,final_states,rho=None):
	basins = initial_states+final_states
	inter_region = ~basins
	if rho is None:
		rho = np.ones(B.shape[0])
	myrho = rho[initial_states] / rho[initial_states].sum()
	BAB = (B[final_states,:][:,initial_states]@myrho).sum()

	if inter_region.sum()>0:
		iGI = eye(inter_region.sum(),format="csr") - B[inter_region,:][:,inter_region].copy()
		x = spsolve(iGI,B[inter_region,:][:,initial_states]@myrho)
		BABI = (B[final_states,:][:,inter_region]@x).sum()
	else:
		BABI = 0.0
	return BABI,BAB



def make_fastest_path(G,i,f,depth=1,limit=None):
	d, cspath = csgraph.shortest_path(csgraph=G, indices=[f,i],\
									directed=False, method='D', return_predecessors=True)
	path = [i]
	s = "\npath: "
	while path[-1] != f:
		s += str(path[-1])+" -> "
		path.append(cspath[0][path[-1]])
	s += str(path[-1])+"\n"
	print(s)

	N = G.shape[0]
	path_region = np.zeros(N,bool)

	# path +
	G=G.tocsc()
	for path_ind in path:
		path_region[path_ind] = True
	for scan in range(depth):
		indscan = np.arange(N)[path_region]
		for path_ind in indscan:
			path_region[G[:,path_ind].indices[G[:,path_ind].data.argsort()[:limit]]] = True
			#path_region[G[:,path_ind].indices[:limit]] = True
			#for sub_path_ind in G[:,path_ind].indices[:limit]:
			#    path_region[sub_path_ind] = True
	return path, path_region


def singleGT(B,tau,sel,timeit=False,dense=False):
	if timeit:
		t=timer()

	""" seems crazy but isn't the bottleneck ... """
	"""
	Bxx = B[sel,:].transpose()[sel,:].transpose()
	Bxj = B[sel,:].transpose()[~sel,:].transpose()
	Bix = B[~sel,:].transpose()[sel,:].transpose()
	Bij = B[~sel,:].transpose()[~sel,:].transpose()
	"""
	Bxx = B[sel,:][:,sel]
	Bxj = B[sel,:][:,~sel]
	Bix = B[~sel,:][:,sel]
	Bij = B[~sel,:][:,~sel]
	iDjj = tau[~sel]
	iDxx = tau[sel]

	if timeit:
		t("slicing")
	if sel.sum()>1:
		# Get Bxx
		Bd = np.ravel(Bxx.diagonal())
		# Get sum Bix for i not equal x
		Bxxnd = Bxx - np.diag(Bd)
		Bs = np.ravel(Bix.sum(axis=0))
		Bs += np.ravel(Bxxnd.sum(axis=0))

		Bs[Bd<0.99] = 1.0-Bd[Bd<0.99]
		iGxx = np.diag(Bs) - Bxxnd
		try:
			Gxx = np.linalg.inv(iGxx)
		except np.linalg.LinAlgError as err:
			return B,tau,False
		iDG = Gxx.transpose().dot(iDxx).transpose()
		iDjj += np.ravel(Bxj.transpose().dot(iDG)).flatten()

		if timeit:
			t("Gxx inv + tau mult.")

		if dense:
			Bij += Bix.dot(Gxx).dot(Bxj) # this is where the work is....
			if timeit:
				t("B mult. (dense)")
		else:
			Bij += Bix*csr_matrix(Gxx)*Bxj # this is where the work is....
			if timeit:
				t("B mult. (sparse)")
		return Bij,iDjj,True
	else:
		Bxx = Bxx.sum()
		#print("HHH",Bxx,Bix.sum(),Bxx+Bix.sum())
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

		return Bij,iDjj,True#,ts,tm

def GT(rm_vec,B,tau=None,block=1,dense=False,order=None,
	Ndense=500,force_sparse=True,screen=False,retK=False):
	r""" Main function for GT code. By default, takes in a branching probability matrix, :math:`\textbf{B}`,
	in sparse format and a vector tau of waiting times. If :math:`\textbf{B}` is supplied in dense format,
	then `dense` should be set to True.

	Parameters
	----------
	rm_vec : array-like[bool] (N,)
		selects out states to eliminate with GT
	B : array-like[float] (N,N)
		Branching probability matrix in dense or sparse format.
	tau : array-like[float] (N,)
		array of waiting times.
	block : int
		Number of states to remove in a given block. Defaults to 1.
	dense : bool
		Whether or not inputted branching probability matrix is dense. Defaults to False.
	force_sparse : bool
		Whether to return matrices in sparse format. Defaults to True.
		TODO: current code requires matrices to be returned in sparse format. Can we have the option to return dense?
	screen : bool
		Whether to print progress of GT.
	retK : bool
		Whether to return the GT-reduced rate matrix in addition to B and D.

	"""
	rmb=trmb
	retry=0
	N = rm_vec.size
	#total number of states to remove
	NI = rm_vec.sum()
	D = 1.0 / tau
    

	if screen:
		print("GT regularization removing %d states:" % NI)
		if has_tqdm:
			pbar = tqdm(total=NI,leave=False,mininterval=0.0)
	tst = 0.0
	tmt = 0.0
	tc = 0

	pass_over = np.empty(0,bool)
	pobar = None
	dense_onset = 0
	if dense:
		dense_onset = B.shape[0]
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
			rm[order.argsort()[:min(rmb,NI)]] = True

		B, tau, success = singleGT(B,tau,rm,timeit=False,dense=dense)

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
			rmb = 1 + (trmb-1)*int(pass_over.sum()==0)
		else:
			pass_over = rm
			rmb = 1
			retry += 1
			if screen and has_tqdm:
				pobar = tqdm(total=rm.sum(),leave=False,mininterval=0.0,desc="STATE-BY-STATE GT")
	if screen and has_tqdm:
		pbar.close()
	if dense and screen:
		print("GT BECAME DENSE AT N=%d, density=%f" % (dense_onset,density))
	if force_sparse:
		if screen:
			print("casting to csr_matrix")
		B = csr_matrix(B)
		B.eliminate_zeros()
	if screen:
		print("GT done, %d rescans due to LinAlgError" % retry)

	Bd = np.ravel(B.diagonal()) # only the diagonal (Bd_x = B_xx)
	Bn = B - diags(Bd) # B with no diagonal (Bn_xx = 0, Bn_xy = B_xy)
	Bn.eliminate_zeros()
	Bnd = np.ravel(Bn.sum(axis=0)) # Bnd_x = sum_x!=y B_yx = 1-B_xx
	nBd = np.zeros(N)
	nBd[Bd>0.99] = Bnd[Bd>0.99]
	nBd[Bd<0.99] = 1.0-Bd[Bd<0.99]
	omB = diags(nBd) - Bn # 1-B

	D = 1.0/tau
	D = np.ravel(D).flatten()
	if retK:
		K = omB.dot(diags(D)) # (1-B).D = K ( :) )
		return B,D,K,N,retry
	return B,D,N,retry
