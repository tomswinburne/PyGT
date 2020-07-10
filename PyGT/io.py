# -*- coding: utf-8 -*-
"""
Read in input files describing the Markov chain to analyze
----------------------------------------------------------
This module reads in the following input files:
Files defining A and B sets
...........................
min.A: single-column[int], (N_A + 1, )
	First line contains the number of nodes in community A.
	Subsequent lines contain the node ID (1-indexed) of the nodes belonging to A.

min.B: single-column[int], (N_B + 1, )
	First line contains the number of nodes in community B.
	Subsequent lines contain the node ID (1-indexed) of the nodes belonging to B.

communities.dat : single-column (nnnodes,)
	used by DISCOTRESS and ktn_analysis module; each line contains community ID (0-indexed)
	of the node specified by the line number in the file

Files describing stationary points of energy landscape
......................................................

The following files are designed to describe a Markov chain in which nodes
correspond to potential or free energy minima, and edges correspond to the
transition states that connect them. These files are also used as input to
the PATHSAMPLE program implemented in the Fortran language:

min.data: multi-column, (nnodes, 6)
	Line numbers indicate node-IDs. Each line contains energy of local minimum [float],
	log product of positive Hessian eigenvalues [float], point group [int], sorted
	eigenvalues of inertia tensor itx [float], ity [float], itz [float]
ts.data: multi-column, (nts, 8)
	Each line contains energy of transition state [float], log product of positive
	Hessian eigenvalues [float], point group [int], ID of first minimum it connects [int],
	ID of second minimum it connects [int], sorted eigenvalues of inertia tensor itx [float],
	ity [float], itz [float]

The ktn.io package then calculates transition rates using unimolecular rate theory.

"""

import os,time,sys
from io import StringIO
import numpy as np
from scipy.sparse import csgraph, csr_matrix, csc_matrix, eye, save_npz, load_npz, diags, issparse
import warnings
from .core import GT
from scipy.special import factorial

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

class output_str:
	def __init__(self):
		self.print_str=""
	def __call__(self,sa):
		_print_str = ""
		for s in sa:
			_print_str += str(s)+" "
		print(_print_str)
		self.print_str += _print_str
	def summary(self):
		print("SUMMARY:\n",self.print_str)


def load_ktn_AB(data_path,index_sel=None):
	""" Read in A_states and B_states from min.A and min.B files, only keeping
	the states that are part of the largest connected set, as specified by
	index_sel.

	Parameters
	----------
	data_path: str
		path to location of min.A, min.B files
	index_sel: array-like[bool] (nnodes, )
		selects out indices of the maximum connected set

	Returns
	-------
	A_states : array-like[bool] (index_sel.size, )
		boolean array that selects out the A states
	B_states : array-like[bool] (index_sel.size, )
		boolean array that selects out the B states

	"""

	Aind = np.zeros(1).astype(int)
	for line in open(os.path.join(data_path,'min.A')):
		Aind = np.append(Aind,np.genfromtxt(StringIO(line.strip())).astype(int)-1)
	Aind = Aind[2:]

	Bind = np.zeros(1).astype(int)
	for line in open(os.path.join(data_path,'min.B')):
		Bind = np.append(Bind,np.genfromtxt(StringIO(line.strip())).astype(int)-1)
	Bind = Bind[2:]

	if index_sel is None:
		return Aind,Bind

	keep = np.zeros(index_sel.size,bool)
	keep[Bind] = True

	B_states = keep[index_sel]

	keep = np.zeros(index_sel.size,bool)
	keep[Aind] = True
	A_states = keep[index_sel]
	return A_states,B_states

def read_communities(commdat, index_sel, screen=False):
	"""Read in a single column file called communities.dat where each line
	is the community ID (zero-indexed) of the minima given by the line
	number. Produces boolean arrays, one per community, selecting out the
	nodes that belong to each community.

	Parameters
	----------
	commdat : .dat file
		single-column file containing community IDs of each minimum
	index_sel : (N,) boolean array
		selects out the largest connected component of the network

	Returns
	-------
	communities : dict
		mapping from community ID (0-indexed) to a boolean array
		of shape (len(index_sel), ) which selects out the states in that community.
	"""

	communities = {}
	with open(commdat, 'r') as f:
		for minID, line in enumerate(f, 0):
			groupID =  int(line) #number from 0 to N-1
			if groupID in communities:
				communities[groupID].append(minID)
			else:
				communities[groupID] = [minID]

	for ci in range(len(communities)):
		#create a new index_selector to select out the minima in community ci
		keep = np.zeros(index_sel.size,bool)
		keep[communities[ci]] = True
		#re-assign communities[ci] to be the index-selector for the maximally connected component of the graph
		communities[ci] = keep[index_sel]
		if screen:
			print(f'Community {ci}: {keep.sum()}')

	return communities

def load_ktn(path,beta=1.0,Nmax=None,Emax=None,screen=False,discon=False):
	r""" Load in min.data and ts.data files, calculate rates, and find connected
	components.

	Parameters
	----------
	path : str
		path to min.data and ts.data files

	beta : float, optional
		value for 1 / (k_B T). Default = 1.0

	Nmax : int, optional
		maximum number of minima to include in KTN. Default = None (i.e. infinity)

	Emax : float, optional
		maximum potential energy of minima/TS to include in KTN. Default = None (i.e. infinity)

	screen : bool, optional
		whether to print progress. Default = False

	discon : bool, optional
			Output data for disconnectivity graph construction (undocumented) Default = False

	Returns
	-------
	B : (N,N) matrix
		sparse matrix of branching probabilities

	K : (N,N) csr matrix
		sparse matrix where off-diagonal elements :math:`K_{ij}` contain :math:`i \leftarrow j` transition rates.
		Diagonal elements are 0.

	tau : (N,) array_like
		vector of waiting times such that total escape rate in state :math:`i` is :math:`1/\tau_i`
		Full rate matrix is then given by :math:`K_{ij}-\delta_{ij}/\tau_i`

	N : int
		number of nodes in the largest connected component of the Markov chain

	u : (N,) array_like
		energies of the N nodes in the Markov chain

	s : (N,) array_like
		entropies of the N nodes in the Markov chain

	Emin : float
		energy of global minimum (energies in `u` are rescaled so that Emin=0)

	retained : np.ndarray[bool] (nnodes,)
		Boolean array selecting out largest connected component (index_sel.sum() = N).

	"""

	GSD = np.loadtxt(os.path.join(path,'min.data'), \
						dtype={'names': ('E','S','DD','RX','RY','RZ'),\
							'formats': (float,float,int,float,float,float)})
	TSD = np.loadtxt(os.path.join(path,'ts.data'),\
		dtype={'names': ('E','S','DD','F','I','RX','RY','RZ'),\
		'formats': (float,float,int,int,int,float,float,float)})

	TSD['I'] = TSD['I']-1
	TSD['F'] = TSD['F']-1

	#number of minima
	N = max(TSD['I'].max()+1,TSD['F'].max()+1)

	if not Nmax is None:
		N = min(Nmax,N)
	#select out minima < Emax and < Nmax
	sels = (TSD['I']<N) * (TSD['F']<N) * (TSD['I']!=TSD['F'])
	if not Emax is None:
		sels *= GSD['E'][TSD['I']]<Emax
		sels *= GSD['E'][TSD['F']]<Emax
		sels *= TSD['E']<Emax
	TSD = TSD[sels]
	GSD = GSD[:N]
	#re-scale energies so Emin = 0, Smin=0
	#print("N,N_TS:",GSD.size,TSD.size)
	Emin = GSD['E'].min().copy()
	Smin = min(GSD['S'].min().copy(),TSD['S'].min().copy())
	GSD['E'] -= Emin
	TSD['E'] -= Emin
	GSD['S'] -= Smin
	TSD['S'] -= Smin

	""" Calculate rates """
	i = np.hstack((TSD['I'],TSD['F']))
	f = np.hstack((TSD['F'],TSD['I']))
	#(emin - ets)
	du = np.hstack((TSD['E']-GSD[TSD['I']]['E'],TSD['E']-GSD[TSD['F']]['E']))
	#(fvibmin - fvibts)/2
	ds = np.hstack((GSD[TSD['I']]['S']-TSD['S'],GSD[TSD['F']]['S']-TSD['S']))/2.0
	#ordermin/(orderts*2pi)
	dc = np.hstack((GSD[TSD['I']]['DD']/TSD['DD'],GSD[TSD['F']]['DD']/TSD['DD']))/2.0/np.pi
	ds += np.log(dc)

	s = GSD['S']/2.0 + np.log(GSD['DD'])

	"""+ds Fill matricies: K_ij = rate(j->i), K_ii==0. iD_jj = 1/(sum_iK_ij) """
	data = np.zeros(du.shape)
	if discon:
		ddu = du.copy()
	#this is the rates, but data is horizontally stacked
	data[:] = np.exp(-beta*du+ds)
	data[i==f] *= 2.0
	fNi = f*N+i
	fNi_u = np.unique(fNi)
	d_u = np.r_[[data[fNi==fi_ind].sum() for fi_ind in fNi_u]]
	if discon:
		d_du = np.r_[[ddu[fNi==fi_ind].sum() for fi_ind in fNi_u]]
	f_u = fNi_u//N
	i_u = fNi_u%N
	K = csr_matrix((d_u,(f_u,i_u)),shape=(N,N))
	if discon:
		DU = csr_matrix((d_du,(f_u,i_u)),shape=(N,N))

	""" connected components """
	K.eliminate_zeros()
	#nc is number of connected components,
	# cc is list of labels of size K
	nc,cc = csgraph.connected_components(K)
	sum = np.zeros(nc,int)
	mc = 0
	for j in range(nc):
		#count number of minima in each connected component
		sum[j] = (cc==j).sum()
	#select largest connected component (value of j for which sum[j] is
	# greatest)
	sel = cc==sum.argmax()

	if screen:
		ss=95
		print("\n\tConnected Clusters: %d, of which %d%% have <= %d states" % (nc,ss,np.percentile(sum,ss)))
		print("\tRetaining largest cluster with %d nodes\n" % sum.max())
	oN=N

	K,N = K[sel,:][:,sel], sel.sum()

	if discon:
		DU = DU[sel,:][:,sel]

	GSD = GSD[sel]
	#entropy of minima
	s = -GSD['S']/2.0 - np.log(GSD['DD'])

	if discon:
		return N,GSD['E'],DU

	kt = np.ravel(K.sum(axis=0))
	#inverse of D
	iD = csr_matrix((1.0/kt,(np.arange(N),np.arange(N))),shape=(N,N))
	#D_i = sum_i K_ij
	D = csr_matrix((kt,(np.arange(N),np.arange(N))),shape=(N,N))
	#branching probability matrix B_ij = k_ij/(sum_i k_ij)
	B = K@iD
	return B, K, 1.0/kt, N, GSD['E'], s, Emin, sel

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

def load_save_mat(path="../../data/LJ38",beta=5.0,Nmax=8000,Emax=None,generate=True,TE=False,screen=False):
	name = path.split("/")[-1]
	if len(name)==0:
		name = path.split("/")[-2]
	if not generate:
		try:
			B = load_npz('__cache__/temp_%s_%2.6g_B.npz' % (name,beta))
			tau = load_npz('__cache__/temp_%s_%2.6g_tau.npz' % (name,beta))
			K = load_npz('__cache__/temp_%s_%2.6g_K.npz' % (name,beta))
			USEB = np.loadtxt('__cache__/temp_%s_%2.6g_USEB.txt' % (name,beta))
			sel = np.loadtxt('__cache__/temp_%s_%2.6g_sel.txt' % (name,beta)).astype(bool)
		except IOError:
			generate = True
			if screen:
				print("no files found, generating...")

	if generate:
		if screen:
			print("Generating....")
		B, K, tau, N, U, S, Emin, sel = load_ktn(path,beta=beta,Nmax=Nmax,Emax=Emax,screen=screen)
		USEB = np.zeros((U.shape[0]+1,2))
		USEB[-1][0] = beta
		USEB[-1][1] = Emin
		USEB[:-1,0] = U
		USEB[:-1,1] = S
		np.savetxt('__cache__/temp_%s_%2.6g_USEB.txt' % (name,beta),USEB)
		np.savetxt('__cache__/temp_%s_%2.6g_sel.txt' % (name,beta),sel)
		save_npz('__cache__/temp_%s_%2.6g_B.npz' % (name,beta),B)
		save_npz('__cache__/temp_%s_%2.6g_K.npz' % (name,beta),K)
		save_npz('__cache__/temp_%s_%2.6g_tau.npz' % (name,beta),tau)

	beta = USEB[-1][0]
	N = USEB.shape[0]-1
	Emin = int(USEB[-1][1])
	U = USEB[:-1,0]
	S = USEB[:-1,1]
	#print("%d states, beta=%f, emin=%f" % (N,beta,Emin))

	kt = np.ravel(K.sum(axis=0)).copy()
	K.data = 1.0/K.data
	kcon = kt * np.ravel(K.sum(axis=0)).copy()
	K.data = 1.0/K.data

	return beta, B, K, D, N, U, S, kt, kcon, Emin, sel

def load_save_mat_gt(keep_ind,beta=10.0,path="../../data/LJ38",Nmax=None,Emax=None,generate=True):
	name = path.split("/")[-1]
	if len(name)==0:
		name = path.split("/")[-2]

	if not generate:
		try:
			B = load_npz('__cache__/temp_%s_B.npz' % name)
			tau = load_npz('__cache__/temp_%s_tau.npz' % name)
			F = np.loadtxt('__cache__/temp_%s_F.txt' % name)
			map = np.loadtxt('__cache__/temp_%s_M.txt' % name,dtype=int)
		except IOError:
			generate = True
			print("no files found, generating...")


	if generate:
		print("Generating....")
		B,tau,F,map = load_mat_gt(keep_ind,path,beta=beta,Nmax=Nmax,Emax=Emax)
		np.savetxt('__cache__/temp_%s_F.txt' % name,F)
		np.savetxt('__cache__/temp_%s_M.txt' % name,map,fmt="%d")
		save_npz('__cache__/temp_%s_B.npz' % name,B)
		save_npz('__cache__/temp_%s_tau.npz' % name,tau)

	return B,D,F,map

def load_mat_gt(keep_ind,path='../data/LJ38/raw/',beta=10.0,Nmax=None,Emax=None):

	""" load data """
	GSD = np.loadtxt(os.path.join(path,'min.data'),\
		dtype={'names': ('E','S','DD','RX','RY','RZ'),\
		'formats': (float,float,int,float,float,float)})

	TSD = np.loadtxt(os.path.join(path,'ts.data'),\
		dtype={'names': ('E','S','DD','F','I','RX','RY','RZ'),\
		'formats': (float,float,int,int,int,float,float,float)})

	TSD = TSD[TSD['I']!=TSD['F']] # remove self transitions??
	TSD['I'] = TSD['I']-1
	TSD['F'] = TSD['F']-1

	N = max(TSD['I'].max()+1,TSD['F'].max()+1)

	Emin = GSD['E'].min().copy()
	GSD['E'] -= Emin
	TSD['E'] -= Emin
	if not Emax is None:
		Emax -= Emin

	""" Build rate matrix """
	i = np.hstack((TSD['I'],TSD['F']))
	f = np.hstack((TSD['F'],TSD['I']))
	du = np.hstack((TSD['E']-GSD[TSD['I']]['E'],TSD['E']-GSD[TSD['F']]['E']))
	ds = np.hstack((TSD['S']-GSD[TSD['I']]['S'],TSD['S']-GSD[TSD['F']]['S']))

	K = csr_matrix((np.exp(-beta*du+ds),(f,i)),shape=(N,N))
	TE = csc_matrix((np.hstack((TSD['E'],TSD['E'])),(f,i)),shape=(N,N))
	D = np.ravel(K.sum(axis=0)) # vector...

	# oN -> N map : could be unit
	oN = N.copy()
	basins = np.zeros(N,bool)
	basins[keep_ind] = True
	print(D.min())

	nc,cc = csgraph.connected_components(K)
	mc = 0
	if nc>1:
		for j in range(nc):
			sc = (cc==j).sum()
			if sc > mc:
				mc = sc
				ccsel = cc==j
		K = K.tocsc()[ccsel,:].tocsr()[:,ccsel]
		N = ccsel.sum()
		TE = TE.tocsc()[ccsel,:].tocsr()[:,ccsel]
		D = D[ccsel]
		Nb = basins[ccsel].sum()
		print("removing unconnected states: N=%d -> %d, Nbasin=%d -> %d" % (oN,N,oNb,Nb))

	map = -np.ones(oN,int)
	map[ccsel] = np.arange(N)

	""" select states to remove - find everything that jumps less than x high from every state in sel??"""
	B = K.dot(diags(1.0/D,format="csr"))
	F = GSD['E']-GSD['S']/beta

	rm_vec = np.ones(N,bool) # remove all

	f_keep = np.empty(0,int)

	n_keep = map[obasins].copy()
	n_keep = n_keep[n_keep>-1]

	for depth in range(20):
		nn_keep = np.empty(0,int)
		for state in n_keep:
			if n_keep in f_keep:
				continue
			ss = TE.indices[TE.indptr[state]:TE.indptr[state+1]]
			ee = TE.data[TE.indptr[state]:TE.indptr[state+1]]
			nn_keep = np.append(nn_keep,ss[ee<Emax])
		f_keep = np.unique(np.append(f_keep,n_keep))
		n_keep = np.unique(nn_keep.copy()) # for the next round....

	f_keep = np.unique(np.append(f_keep,n_keep))

	rm_vec[f_keep] = False # i.e. ~rm_vec survives

	kept = np.zeros(oN,bool)
	kept[ccsel] = ~rm_vec # i.e. selects those which were kept

	map = -np.ones(oN,int)
	map[kept] = np.arange(kept.sum())

	B,tau = GT(rm_vec=rm_vec,B=B,tau=1.0/D,block=1,order=None)
	N = tau.size

	if not dense:
		B = csr_matrix(B)

	return B,tau,F[~rm_vec],map
