import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from PyGT.gt_tools import *
""" test for tqdm progress bars """
try:
	from tqdm import tqdm
	has_tqdm=True
except:
	has_tqdm=False


class sampler:
	def __init__(self,sys,max_d=20):
		self.sparse=True
		self.sys = sys
		self.max_d = max_d
		self.sys.rK.eliminate_zeros()
		probed=self.sys.rK.copy()
		self.probed = (probed.astype(bool).copy() + sp.diags(np.ones(self.sys.N),format="csr").astype(bool).copy()).tolil(copy=True)
		del probed
		self.sparseFactor = [.9,1.0] # assume v dense

	def initial_sample(self):
		# do a DNEB, find some paths to start from
		n_a = np.arange(self.sys.N)[self.sys.selA]
		n_b = np.arange(self.sys.N)[self.sys.selB]

		#self.probed = self.probed.tolil()
		self.probed[self.sys.selA,:][:,self.sys.selA] = True
		self.probed[self.sys.selB,:][:,self.sys.selB] = True

		Np = self.sys.remaining_pairs()
		#n_aib = []
		self.sys.tolil()

		for t_i in n_a:
			#n_aib.append(t_i)
			for t_f in n_b:
				#n_aib.append(t_f)
				if not self.probed[t_i,t_f]:
					ia,fa,ka = self.sys.SaddleSearch(t_i,t_f)
					self.probed[t_i,t_f] = True
					self.probed[t_f,t_i] = True
					#for ii in ia:
					#	n_aib.append(ii)
					#for ff in fa:
					#	n_aib.append(ff)
					self.sys.add_connections(ia,fa,ka)
		#n_aib = list(set(n_aib))
		dNp = Np-self.sys.remaining_pairs()
		print("INITIAL SADDLE SAMPLE DONE : found %d/%d pairs" % (dNp,Np))

		for t_i in n_b:
			for t_f in n_a:
				if not self.probed[t_i,t_f]:
					ia,fa,ka = self.sys.DNEB(t_i,t_f,pM=sp.csr_matrix(self.probed))
					self.probed[t_i,t_f]=True
					self.probed[t_f,t_i]=True
					self.sys.add_connections(ia,fa,ka)
		dNp = Np-self.sys.remaining_pairs()
		print("INITIAL DNEB SAMPLE DONE : found %d/%d pairs" % (dNp,Np))
		self.sys.tocsr()

	def initial_sample_path(self,path):
		# do a DNEB, find some paths to start from
		n_a = np.arange(self.sys.N)[self.sys.selA]
		n_b = np.arange(self.sys.N)[self.sys.selB]
		self.probed[self.sys.selA,:][:,self.sys.selA] = True
		self.probed[self.sys.selB,:][:,self.sys.selB] = True
		Np = self.sys.remaining_pairs()

		if has_tqdm:
			pbar = tqdm(total=len(path)*(len(path)-1)//2,leave=False)
		print(len(path))
		for t_i in path:
			for t_f in path:
				if t_i<t_f:
					ia,fa,ka = self.sys.DNEB(t_i,t_f,pM=self.probed)
					if self.dense:
						self.probed[t_i][t_f]=True
						self.probed[t_f][t_i]=True
					else:
						self.probed[t_i,t_f]=True
						self.probed[t_f,t_i]=True
					self.sys.add_connections(ia,fa,ka)
					if has_tqdm:
						pbar.update(1)
		if has_tqdm:
			pbar.close()
		dNp = Np-self.sys.remaining_pairs()
		print("INITIAL SAMPLE DONE : found %d/%d pairs" % (dNp,Np))

	def initial_sample_path_region(self,path_region,ncs=4):
		# do a DNEB, find some paths to start from
		"""
		n_a = np.arange(self.sys.N)[self.sys.selA]
		n_b = np.arange(self.sys.N)[self.sys.selB]

		sel = self.sys.selA + self.sys.selB
		self.probed[sel,:][:,sel] = True


		sel = np.zeros(self.sys.N,bool)
		sel[path_region] = True

		# copy existing rK? no, just make a new matrix idiot.....
		data = self.sys.K[sel,:].transpose()[sel,:].transpose().todense().copy()
		f=np.outer(np.arange(self.sys.N)[sel],np.ones(data.shape[0],int)).astype(int).flatten()
		i=np.outer(np.arange(self.sys.N)[sel],np.ones(data.shape[0],int)).transpose().astype(int).flatten()
		rK=sp.csr_matrix((np.ravel(data.flatten()),(f,i)),shape=(self.sys.N,self.sys.N))
		rK.eliminate_zeros()
		self.sys.rK += rK

		"""
		if self.sparse:
			self.sys.rK = self.sys.rK.tolil()
			self.probed = self.probed.tolil()
		Np = self.sys.remaining_pairs()
		if has_tqdm:
			pbar = tqdm(total=len(path_region),leave=False,miniters=0)
		for t_i in path_region:
			ia,fa,ka = self.sys.SaddleSearch(t_i)
			for t_f in fa[:ncs]:
				self.probed[t_i,t_f]=True
				self.probed[t_f,t_i]=True

			self.sys.add_connections(ia[:ncs],fa[:ncs],ka[:ncs])
			if has_tqdm:
				pbar.update(1)
		if has_tqdm:
			pbar.close()
		#if self.sparse:
		#	self.sys.rK = self.sys.rK.tocsr()
		#	self.probed = self.probed.tocsr()

		"""
		print(self.sys.rK.shape,self.sys.K.shape,self.probed.shape)
		print(self.probed.sum())
		#print((self.sys.rK*self.probed-self.sys.K*self.probed).min())
		print(sp.issparse(self.sys.rK))
		print((self.sys.rK>0.0).sum(),(self.sys.K * self.probed).shape,(self.sys.K[self.probed]>0.0).sum(),self.probed.max())
		self.sys.rK[~self.probed] += self.sys.K[~self.probed]
		"""
		dNp = Np-self.sys.remaining_pairs()
		self.init = dNp
		print("INITIAL SAMPLE DONE : found %d/%d pairs; %d remaining" % (dNp,Np,self.sys.remaining_pairs()))

	def remaining_pairs(self):
		return self.sys.remaining_pairs()

	def make_dense(self):
		self.sparse = False
		M = np.empty(self.probed.shape,bool)
		self.probed.astype(bool).todense(out=M)
		self.probed=M.copy()
		del M
		self.sys.make_dense()

	def sample(self,ignore_distance=False,npairs=20,nfilter=10000,ss=False):
		pairs,res = self.sensitivity(ignore_distance=ignore_distance,npairs=4*npairs,nfilter=nfilter)
		c=0
		pc=0
		for p in pairs:
			if ss:
				tia,tfa,tka = self.sys.SaddleSearch(p[0],p[1])
				ia,fa,ka=[],[],[]
				for i in range(len(tia)):
					if tia[i]!=p[0]:
						continue
					if not self.probed[tia[i],tfa[i]]:
						ia.append(tia[i])
						fa.append(tfa[i])
						ka.append(tka[i])
						self.probed[tia[i],tfa[i]] = True
						self.probed[tfa[i],tia[i]] = True
					if len(ia)==1:
						break
				for i in range(len(tia)):
					if tfa[i]!=p[1]:
						continue
					if not self.probed[tia[i],tfa[i]]:
						ia.append(tia[i])
						fa.append(tfa[i])
						ka.append(tka[i])
						self.probed[tia[i],tfa[i]] = True
						self.probed[tfa[i],tia[i]] = True
					if len(ia)==1:
						break

			else:
				ia,fa,ka = self.sys.DNEB(p[0],p[1],pM=self.probed)
				self.probed[p[0],p[1]] = True
				self.probed[p[1],p[0]] = True
			pc += int(self.probed[p[0],p[1]])
			cc = self.sys.add_connections(ia,fa,ka)
			c += cc
			if c>=npairs:
				break
		return len(pairs),res,c

	def sensitivity(self,ignore_distance=True,npairs=20,nfilter=10000,rho=None):
		Nr = np.arange(self.sys.N)
		if self.sparse:
			rK = self.sys.rK.copy()# - sp.diags(self.sys.rK.diagonal(),format='csr')
		else:
			rK = self.sys.rK.copy() - np.diagflat(self.sys.rK.diagonal())

		kt = np.ravel(rK.sum(axis=0))
		selA,selI,selB=self.sys.selA*(kt>0.0),self.sys.selI*(kt>0.0),self.sys.selB*(kt>0.0)
		nA,nI,nB = selA.sum(),selI.sum(),selB.sum()
		mapA,mapI,mapB = Nr[selA], Nr[selI], Nr[selB]
		if rho is None:
			rho = np.ones(self.sys.N)
		rhoB = rho[selB].copy() / rho[selB].sum()

		if self.sparse:
			iDI = sp.diags(1.0 / kt[selI], format='csr')
			iDB = sp.diags(1.0 / kt[selB], format='csr')
			rK = rK.tocsr()
			BIB = rK[selI,:][:,selB]@iDB
			BAB = rK[selA,:][:,selB]@iDB
			BAI = rK[selA,:][:,selI]@iDI
			BII = rK[selI,:][:,selI]@iDI
			BBI = rK[selB,:][:,selI]@iDI
		else:
			iDI = np.diagflat(1.0/kt[selI])
			iDB = np.diagflat(1.0/kt[selB])
			BIB = rK[selI,:][:,selB].dot(iDB)
			BAB = rK[selA,:][:,selB].dot(iDB)
			BAI = rK[selA,:][:,selI].dot(iDI)
			BII = rK[selI,:][:,selI].dot(iDI)
			BBI = rK[selB,:][:,selI].dot(iDI)

		pi = np.exp(-self.sys.beta*self.sys.f)
		piA,piI,piB = pi[selA],pi[selI],pi[selB]
		oneA = np.ones(nA)
		Nt = float(nA+nI+nB) * float(nA+nI+nB)
		Na = float(self.probed.sum() // 2)
		if self.sparse:
			ed = float(self.sys.rK.nnz) / Nt
		else:
			ed = 1.0/Nt * (self.sys.rK>0.0).sum()
		"""
		linear solves:
		(1-BII).x = iGI.x = BIB.piB
		y.(1-BII) = y.iGI = 1.BAI
		"""
		# inverse Green function
		if self.sparse:
			iGI = sp.diags(np.ones(nI), format='csr') - BII
			x = spsolve(iGI,BIB@rhoB)
			y = spsolve(iGI.transpose(),oneA@BAI)
		else:
			iGI = np.identity(nI) - BII
			try:
				x = np.linalg.solve(iGI,np.ravel(BIB@rhoB))
			except np.linalg.LinAlgError as err:
				x,resid,rank,s = np.linalg.lstsq(iGI,np.ravel(BIB@rhoB),rcond=None)
			try:
				y = np.linalg.solve(iGI.transpose(),np.ravel(oneA@BAI))
			except np.linalg.LinAlgError as err:
				y = np.linalg.lstsq(iGI.transpose(),np.ravel(oneA@BAI),rcond=None)[0]

		iDx = iDI@x
		bab = (BAI@x).sum()+(BAB@rhoB).sum()
		yBIB = np.ravel(y@BIB)

		# i.e. take largest ij rate to ~ remove state Boltzmann factor
		if self.sparse:
			mM = (rK+rK.transpose())/2.0
			lM = -np.log(mM.data)
		else:
			mM = np.vstack((rK.flatten(),rK.transpose().flatten())).max(axis=0)
			lM = -np.log(mM[mM>0.0])

		mix = min(1.0,np.exp(-lM.std()/lM.mean()))
		ko = 1.0*np.exp(-3.0)
		mink = ko*(1.0-mix) + mix*np.exp(-lM.mean())

		fmapI = mapI
		iDBs = np.ravel(iDB@rhoB)
		yvB = np.ravel(yBIB) * np.ravel(iDBs)

		yvI = np.ravel(y) * np.ravel(iDx)

		cII = np.outer(y,iDx)-np.outer(np.ones(nI),yvI)
		cAI = np.outer(np.ones(nA),iDx-yvI)
		cIB = np.outer(y,iDBs)-np.outer(np.ones(nI),yvB)-np.outer(yvI,np.ones(nB)) # need phi.....


		"""
		b = k/kt -> (k+dk) / (kt+dk) -> 1.0 as dk->infty
		=> db < 1-b....

		dp/dt = -G^{-1}Dp => p = exp(-G^{-1}Dt)p = sum vlexp(-llt)pl
		G^{-1}
		p(t) = int_0^t dp/dt = sum (pl/ll)(1-exp(-llt))vl
		p(t) ~ p0/l0 qsd, [p0/l0] = (1/T)/(1/T) = 1

		iGDp -> sum vl ll pl
		=> D^{-1}G p = sum (1/ll) vl pl ~ p0/l0 qsd
		=> GP ~ p0/l0 D.qsd ->  D/rate . pi
		=> cab . Dpi/rate = 1 doesn't tell me anything..
		=> cab = rate/Dpi
		"""

		kf = np.outer(piA,1.0/piI)
		kf[kf>1.0] = 1.0
		cAI *= kf * mink

		kf = np.outer(piI,1.0/piI)
		kf[kf>1.0] = 1.0
		cII *= kf * mink

		kf = np.outer(piI,1.0/piB)
		kf[kf>1.0] = 1.0
		cIB *= kf * mink

		if self.sparse:
			cAI[self.probed[selA,:][:,selI].A] = 0.0
			cII[self.probed[selI,:][:,selI].A] = 0.0
			cIB[self.probed[selI,:][:,selB].A] = 0.0
		else:
			cAI[self.probed[selA,:][:,selI].A] = 0.0
			cII[self.probed[selI,:][:,selI].A] = 0.0
			cIB[self.probed[selI,:][:,selB].A] = 0.0

		#print("\n----\n",cAI.max(),cAI.min(),cII.max(),cII.min(),cIB.max(),cIB.min(),"\n----")
		c_tot = np.hstack(((np.triu(cII+cII.T)).flatten(),cAI.flatten(),cIB.flatten()))

		#print("c_tot:",c_tot.max(),c_tot.min(),c_tot.mean(),cmp,cmn,cm)

		res = {}

		# sparsity
		res['Sparsity'] = 0.5*np.exp(-0.001*Na) + (1.0 - np.exp(-0.001*Na))*ed
		em = res['Sparsity'] / (1.0-res['Sparsity'])

		res['SingleMaxMin'] = [c_tot.max(),c_tot.min()]
		res['ExpectMaxMin'] = [c_tot[c_tot>0.0].mean() * em ,c_tot[c_tot<0.0].mean() * em]
		res['TotalMaxMin'] = [c_tot[c_tot>0.0].sum(),c_tot[c_tot<0.0].sum()]
		res['TotalSparseMaxMin'] = [c_tot[c_tot>0.0].sum()*res['Sparsity'],c_tot[c_tot<0.0].sum()*res['Sparsity']]
		res['ExpectMaxMaxMin'] = [c_tot[c_tot>0.0].max() * em ,c_tot[c_tot<0.0].min() * em]



		res['ebab'] = bab
		res['mink'] = mink
		res['MaxInRegion'] = np.r_[[np.abs(cAI).max()/bab,np.abs(cII).max()/bab,np.abs(cIB).max()/bab]]
		"""
		pp = np.abs(c_tot).argsort()[-npairs:]#[::-1][:npairs]
		fp_in = np.zeros((npairs,2),int)

		nII = (pp<nI*nI).sum()
		if nII>0:
			fp = pp[pp<nI*nI]
			fp_in[:nII] = np.vstack((fmapI[fp//nI],fmapI[fp%nI])).T

		nAI = (pp<nI*nI+nA*nI).sum()
		if nAI>nII:
			fp = pp[(pp>nI*nI)*(pp<nI*nI+nA*nI)]-nI*nI
			fp_in[nII:nAI] = np.vstack((fmapI[fp//nI],fmapI[fp%nI])).T
		nIB = (pp>=nI*(nI+nA)).sum()
		if nIB>nAI:
			fp = pp[pp>=nI*(nI+nA)]-nI*(nI+nA)
			fp_in[-nIB:] = np.vstack((fmapI[fp//nB],fmapI[fp%nB])).T
		"""
		fp_in = []
		for fp in np.abs(c_tot).argsort()[::-1]:
			if np.abs(c_tot[fp])>1.0e-9*bab:
				fff = np.abs(c_tot[fp])
				if fp<nI*nI:
					p = [fmapI[fp//nI],fmapI[fp%nI]]
				elif fp<nI*nI+nA*nI:
					fp -= nI*nI
					p = [mapA[fp//nI],mapI[fp%nI]]
				else:
					fp -= nI*nI+nA*nI
					p = [mapI[fp//nB],mapB[fp%nB]]
				if not self.probed[p[0],p[1]]:
					fp_in.append(p)
				else:
					print("PROB")

			if len(fp_in) >= npairs:
				break
		return fp_in,res

	def new_true_branching_probability(self,gt_check=True,rho=None):
		BAIB, BAB = direct_solve(self.sys.B,self.sys.selB,self.sys.selA,rho=rho)
		print("DIRECT = %2.4g + %2.4g = %2.4g" % (BAB,BAIB,BAB+BAIB))
		res = (BAIB+BAB)*1.0
		if gt_check:
			rB, rtau = GT(rm_vec=self.sys.selI,B=self.sys.B,tau=1.0/self.sys.kt,block=1)
			rD = 1.0 / rtau
			rN = rtau.size
			r_initial_states = self.sys.selB[~self.sys.selI]
			r_final_states = self.sys.selA[~self.sys.selI]
			BAIB, BAB = direct_solve(rB,r_initial_states,r_final_states,rho=rho)
			print("GT = %2.4g + %2.4g = %2.4g" % (BAB,BAIB,BAB+BAIB))
			gtBAB = BAB+BAIB
			return res,abs(gtBAB-res)/res
		else:
			return res
