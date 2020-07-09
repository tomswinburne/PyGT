import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from .ktn_io import load_save_mat,load_save_mat_gt
from .gt_tools import gt_seq


class aib_system:
	def __init__(self,path="../../data/LJ13",beta=5.0,Nmax=None,Emax=None,generate=True):

		self.beta, self.B, self.K, self.D, self.N, self.u, self.s, self.kt, kcon, Emin, index_sel = \
			load_save_mat(path=path,beta=beta,Nmax=Nmax,Emax=Emax,generate=generate)
		self.f = self.u-self.s/self.beta

	def gt(self,rm_reg,trmb=50):

		self.B, self.kt, self.N, retry = gt_seq(N=self.N,rm_reg=rm_reg,B=self.B.copy(),D=self.kt,trmb=trmb,retK=False)
		self.K = self.B@sp.diags(self.kt)
		self.u = self.u[~rm_reg]
		self.s = self.s[~rm_reg]
		self.f = self.u-self.s/self.beta

	def setup(self):
		""" B,K,D,N,f defined by this point """
		self.pi = np.exp(-self.beta*self.f)
		self.tG = self.K.tocsr() * sp.diags(self.pi,format='csr')
		self.tG.data = 1.0 / self.tG.data
		self.pi /= self.pi.sum()
		self.kt = np.ravel(self.K.sum(axis=0))
		self.udG = sp.csr_matrix(self.K.copy())
		self.udG.data[:] = 1.0
		self.rK = sp.csr_matrix((self.N,self.N))
		self.regions = False



	def define_AB_regions(self,selA,selB):
		# input vector of bools
		self.selA = selA
		self.selB = selB
		self.selI = ~self.selA * ~self.selB
		selI = ~self.selA * ~self.selB

		self.NA = self.selA.sum()
		self.NB = self.selB.sum()
		self.NI = self.selI.sum()

		print("NA, NB, NI:",self.NA,self.NB,self.NI)

		# fill A.B regions
		self.rK = self.rK.tolil()
		self.rK[np.ix_(selA,selA)] = self.K[np.ix_(selA,selA)].copy()
		self.rK[np.ix_(selB,selB)] = self.K[np.ix_(selB,selB)].copy()
		self.rK[np.ix_(selB,selA)] = self.K[np.ix_(selB,selA)].copy()
		self.rK[np.ix_(selA,selB)] = self.K[np.ix_(selA,selB)].copy()

		self.rK = self.rK.tocsr()
		self.regions = True


	def find_path(self,i,f,depth=1,limit=10,strategy="RATE"):
		if strategy == "DNEB":
			G = self.udG
		else:
			G = self.tG
		d, cspath = sp.csgraph.shortest_path(csgraph=G, indices=[f,i],\
										directed=False, method='D', return_predecessors=True)
		path = [i]
		s = "\npath: "
		while path[-1] != f:
			s += str(path[-1])+" -> "
			path.append(cspath[0][path[-1]])
		s += str(path[-1])+"\n"
		print(s)
		N = self.tG.shape[0]
		path_region = np.zeros(N,bool)
		# path +
		G=G.tocsc()
		for path_ind in path:
			path_region[path_ind] = True
			indscan = np.arange(N)[path_region]
		for scan in range(depth):
			for path_ind in indscan:
				for sub_path_ind in G[:,path_ind].indices[:limit]:
					path_region[sub_path_ind] = True
		return path, path_region

	def remaining_pairs(self):
		return (self.K.nnz-self.rK.nnz)//2

	def add_connections(self,i_a,f_a,k_a):
		nn = -self.rK.nnz
		self.rK = self.rK.tolil()
		for ifk in zip(i_a,f_a,k_a):
			self.rK[ifk[1],ifk[0]] = ifk[2][0]
			self.rK[ifk[0],ifk[1]] = ifk[2][1]
		self.rK = self.rK.tocsr()
		nn += self.rK.nnz
		return (nn//2)
	def tolil(self):
		self.rK = self.rK.tolil()
		self.K = self.K.tolil()
	def tocsr(self):
		self.rK = self.rK.tocsr()
		self.K = self.K.tocsr()
	# fake DNEB search between i_s, f_s
	def DNEB(self,i_s,f_s,pM=None):
		i_a = []
		f_a = []
		k_a = []

		havepM = sp.isspmatrix(pM)
		if self.K[i_s,f_s]==0.0 or havepM:
			# if no direct route, find shortest_path for undirected, unweighted graph with Dijkstra
			if havepM:
				G = pM*3000.0 + self.udG
			else:
				G = self.udG
			d,path = sp.csgraph.shortest_path(G,indices=[f_s,i_s],directed=False,method='D',return_predecessors=True)
			f = i_s
			while f != f_s:
				i_a.append(f)
				f_a.append(path[0][f])
				k_a.append([self.K[path[0][f],f],self.K[f,path[0][f]]])
				f = path[0][f]
		else:
			i_a.append(i_s)
			f_a.append(f_s)
			k_a.append([self.K[f_s,i_s],self.K[i_s,f_s]])
			#print(i_s,"->",f_s,k_a[-1])
		return np.r_[i_a].astype(int),np.r_[f_a].astype(int),np.r_[k_a]

	# fake Saddle Search from i_s and f_s
	def SaddleSearch(self,i_s,f_s=None):
		i_a = []
		f_a = []
		k_a = []

		ki = np.ravel(self.K[:,i_s].todense())
		for n_s in np.arange(self.N)[ki>0.0]:
			i_a.append(i_s)
			f_a.append(n_s)
			k_a.append([self.K[n_s,i_s],self.K[i_s,n_s]])
		if not f_s is None:
			ki = np.ravel(self.K[:,f_s].todense())
			for _f_s in np.arange(self.N)[ki>0.0]:
				i_a.append(f_s)
				f_a.append(n_s)
				k_a.append([self.K[n_s,f_s],self.K[f_s,n_s]])
		return i_a,f_a,k_a

	def ConnectingStates(self,i_s):
		f_a = []
		k_a = []

		ki = np.ravel(self.K[:,i_s].todense())
		for n_s in np.arange(self.N)[ki>0.0]:
			f_a.append(n_s)
			k_a.append(self.K[n_s,i_s])

		return f_a,k_a
