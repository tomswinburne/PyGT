import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from lib.ktn_io import load_save_mat


class aib_system:
	def __init__(self,path="../../data/LJ13",beta=5.0,Nmax=5000,Emax=None,generate=True):
		print(path)

		self.beta, self.B, self.K, self.D, self.N, self.f, self.kt, self.kcon = \
			load_save_mat(path=path,beta=beta,Nmax=Nmax,Emax=Emax,generate=generate)

		""" connected components for matrix """
		nc,cc = sp.csgraph.connected_components(self.K)
		mc = 0
		for j in range(nc):
			sc = (cc==j).sum()
			if sc > mc:
				mc = sc
				sel = cc==j


		self.B = self.B.tocsc()[sel,:].tocsr()[:,sel]
		self.K = self.K.tolil()[:,sel][sel,:]
		self.N = sel.sum()
		self.f = self.f[sel]

		#self.map = self.map[sel]
		self.kt = np.ravel(self.K.sum(axis=0))

		self.iD = sp.diags(1.0/self.kt,format='csr')
		self.D = sp.diags(self.kt,format='csr')

		self.pi = np.exp(-beta*self.f)
		self.pi /= self.pi.sum()

		self.udG = sp.csr_matrix(self.K.copy())
		self.udG.data[:] = 1.0

		self.rK = sp.diags(np.zeros(self.N),format='lil')

		self.regions = False

	def define_AB_regions(self,selA,selB):
		# input vector of bools
		self.selA = selA
		self.selB = selB
		self.selI = ~self.selA * ~self.selB

		self.NA = self.selA.sum()
		self.NB = self.selB.sum()
		self.NI = self.selI.sum()

		print("NA, NB, NI:",self.NA,self.NB,self.NI)

		# fill A.B regions
		self.rK[selA,:][:,selA] = self.K[selA,:][:,selA].copy()
		self.rK[selB,:][:,selB] = self.K[selB,:][:,selB].copy()

		self.regions = True

	def remaining_pairs(self):
		return (self.K.nnz-self.rK.nnz)//2

	def add_connections(self,i_a,f_a,k_a):
		c=0
		for ifk in zip(i_a,f_a,k_a):
			if self.rK[ifk[1],ifk[0]]==0.0:
				self.rK[ifk[1],ifk[0]] = ifk[2][0]
				self.rK[ifk[0],ifk[1]] = ifk[2][1]
				c+=1
		return c

	# fake DNEB search between i_s, f_s
	def DNEB(self,i_s,f_s,pM=None):
		i_a = []
		f_a = []
		k_a = []

		havepM = sp.isspmatrix_csr(pM)
		if self.K[i_s,f_s]==0.0 or havepM:
			# if no direct route, find shortest_path for undirected, unweighted graph with Dijkstra
			if havepM:
				G = pM*3000.0 + self.udG
			else:
				G = self.udG
			d,path = sp.csgraph.shortest_path(G,indices=[f_s,i_s],\
											directed=False,method='D',return_predecessors=True)
			f = i_s
			while f != f_s:
				i_a.append(f)
				f_a.append(path[0][f])
				k_a.append([self.K[path[0][f],f],self.K[f,path[0][f]]])
				#print(f,"->",path[0][f])
				f = path[0][f]
		else:
			i_a.append(i_s)
			f_a.append(f_s)
			k_a.append([self.K[f_s,i_s],self.K[i_s,f_s]])
			#print(i_s,"->",f_s,k_a[-1])
		return i_a,f_a,k_a

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
