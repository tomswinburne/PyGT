import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from lib.ktn_io import load_save_mat,load_save_mat_gt
from lib.gt_tools import gt_seq


class aib_system:
	def __init__(self,path="../../data/LJ13",beta=5.0,Nmax=None,Emax=None,generate=True):
		self.beta, self.B, self.K, self.D, self.N, self.u, self.s, self.kt, self.kcon, self.Emin = \
			load_save_mat(path=path,beta=beta,Nmax=Nmax,Emax=Emax,generate=generate)
		self.f = self.u-self.s/self.beta
		self.oN = self.N
		self.sparse=True
		self.setup()
	def setup(self):

		""" connected components for matrix
		KU = self.KU.copy()
		KU.data[KU.data<np.exp(-beta*(Emax-self.Emin))] = 0.0
		KU.eliminate_zeros()
		zs = np.ravel(KU.sum(axis=0)) > 0.0
		print("eliminate_zeros: ",zs.size,zs.sum())
		KU = KU.tocsc()[zs,:].tocsr()[:,zs]

		nc,cc = sp.csgraph.connected_components(KU,directed=False)
		print("done cc",nc)
		sum = np.zeros(nc,int)
		mc = 0
		for j in range(nc):
			sum[j] = (cc==j).sum()
		sel = cc==sum.argmax()
		np.savetxt("cc",cc,fmt='%d')
		np.savetxt("sum",sum,fmt='%d')
		print("saved cc")
		rm_reg = np.ones(self.N,bool)
		rm_reg[np.arange(self.N)[zs][sel]] = False
		"""

		""" B,K,D,N,f defined by this point """
		self.pi = np.exp(-self.beta*self.f)
		self.tG = self.K.tocsr() * sp.diags(self.pi,format='csr')
		self.tG.data = 1.0 / self.tG.data
		self.pi /= self.pi.sum()
		self.kt = np.ravel(self.K.sum(axis=0))
		self.udG = sp.csr_matrix(self.K.copy())
		self.udG.data[:] = 1.0
		self.rK = sp.diags(np.zeros(self.N),format='lil')
		self.regions = False

	def gt(self,sel,trmb=1):
		rm_reg = np.zeros(self.N,bool)
		rm_reg[sel] = True
		self.oN = self.N
		self.D = self.D.data
		self.B,self.D,self.N,retry = gt_seq(self.N,rm_reg,self.B,D=self.D,trmb=trmb,order=None,force_sparse=True)

		""" u,s APPROX """
		self.u = self.u[~rm_reg]
		self.s = self.s[~rm_reg]
		print(self.B.shape,self.B.shape)
		self.K = self.B.copy() * sp.diags(self.D,format="csr")
		self.kt = np.ravel(self.K.sum(axis=0))
		self.f = self.u - self.s / self.beta
		self.setup()

	def find_path(self,i,f,depth=1,limit=10,strategy="RATE"):
		if strategy == "DNEB":
			G = self.udG
		else:
			G = self.tG
		d, cspath = sp.csgraph.shortest_path(csgraph=G, indices=[f,i],\
										method='D', return_predecessors=True)
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

	def define_AB_regions(self,selA,selB):
		# input vector of bools
		self.selA = selA
		self.selB = selB
		self.selI = ~(self.selA+self.selB)

		self.NA = self.selA.sum()
		self.NB = self.selB.sum()
		self.NI = self.selI.sum()

		print("NA, NB, NI:",self.NA,self.NB,self.NI)

		# fill A.B regions
		sel = selA+selB
		data = self.K[sel,:].transpose()[sel,:].transpose().todense().copy()
		dN = data.shape[0]
		f=np.outer(np.arange(self.N)[sel],np.ones(dN,int)).astype(int).flatten()
		i=np.outer(np.arange(self.N)[sel],np.ones(dN,int)).transpose().astype(int).flatten()
		self.rK=sp.csr_matrix((np.ravel(data.flatten()),(f,i)),shape=(self.N,self.N))
		self.rK.eliminate_zeros()
		self.rK -= sp.diags(self.rK.diagonal(),format='csr')
		self.rK += sp.diags(self.K.diagonal().copy(),format='csr')

		self.regions = True

	def remaining_pairs(self):
		if self.sparse:
			return (self.K.nnz-self.rK.nnz)#//2
		else:
			return (self.K>0.0).sum()-(self.rK>0.0).sum()#//2 - (self.rK>0.0).sum()//2

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
		if self.sparse:
			ki = np.ravel(self.K.transpose()[i_s,:].todense())
		else:
			ki = np.ravel(self.K[:,i_s])
		for n_s in np.arange(self.N)[ki>0.0]:
			i_a.append(i_s)
			f_a.append(n_s)
			k_a.append([self.K[n_s,i_s],self.K[i_s,n_s]])
		if not f_s is None:
			if self.sparse:
				ki = np.ravel(self.K.transpose()[f_s,:].todense())
			else:
				ki = np.ravel(self.K[:,f_s])
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

	def save(self,filename):
		""" B,K,D,N,f defined by this point """
		sp.save_npz(filename+"_rB",self.B)
		USD = np.zeros((self.u.size+1,3))
		USD[:-1,0] = self.u
		USD[:-1,1] = self.s
		USD[:-1,2] = np.ravel(self.D.data)
		USD[-1][0] = self.beta
		USD[-1][1] = self.Emin
		USD[-1][2] = self.oN
		np.savetxt(filename+"_USD",USD)

	def load(self,filename):
		self.B = sp.load_npz(filename+"_rB.npz")
		USD = np.loadtxt(filename+"_USD")
		self.u = USD[:-1,0]
		self.s = USD[:-1,1]
		self.D = USD[:-1,2]
		self.K = self.B * sp.diags(self.D)
		self.beta = USD[-1][0]
		self.Emin = USD[-1][1]
		self.oN = int(USD[-1][2])
		self.N = self.u.shape[0]
		self.f = self.u-self.s/self.beta
		""" B,K,D,N,f defined by this point """
		self.setup()

	def make_dense(self):
		self.sparse=False
		M = np.empty(self.rK.shape)
		self.rK.todense(out=M)
		self.rK = M.copy()

		M = np.empty(self.K.shape)
		self.K.todense(out=M)
		self.K = M.copy()


		del M
