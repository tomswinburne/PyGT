import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from lib.gt_tools import *
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
		probed=self.sys.rK.copy()
		self.probed = probed.astype(bool).copy() + sp.diags(np.ones(self.sys.N),format="csr").astype(bool).copy()
		del probed
		self.sparseFactor = [.9,1.0] # assume v dense

	def initial_sample(self):
		# do a DNEB, find some paths to start from
		n_a = np.arange(self.sys.N)[self.sys.selA]
		n_b = np.arange(self.sys.N)[self.sys.selB]
		self.probed[self.sys.selA,:][:,self.sys.selA] = True
		self.probed[self.sys.selB,:][:,self.sys.selB] = True

		Np = self.sys.remaining_pairs()
		#n_aib = []

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
		for t_i in n_b:
			for t_f in n_a:
				if not self.probed[t_i,t_f]:
					ia,fa,ka = self.sys.DNEB(t_i,t_f,pM=sp.csr_matrix(self.probed))
					self.probed[t_i,t_f]=True
					self.probed[t_f,t_i]=True
					self.sys.add_connections(ia,fa,ka)
		dNp = Np-self.sys.remaining_pairs()
		print("INITIAL SAMPLE DONE : found %d/%d pairs" % (dNp,Np))

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
					ia,fa,ka = self.sys.DNEB(t_i,t_f,pM=sp.csr_matrix(self.probed))
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
			pbar = tqdm(total=len(path_region),leave=False)
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
		pairs,sens,ebp = self.sensitivity(ignore_distance=ignore_distance,npairs=4*npairs,nfilter=nfilter)
		c=0
		pc=0
		for p in pairs:
			if ss>0:
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
			if ss>0 and c>=ss:
				break
			elif c>=npairs:
				break
		#print(pc)
		#self.sparseFactor[0] += pc
		#self.sparseFactor[1] += len(pairs)


		return len(pairs),sens,ebp

	def sensitivity(self,ignore_distance=True,npairs=20,nfilter=10000):
		Nr = np.arange(self.sys.N)
		if self.sparse:
			rK = self.sys.rK.copy()# - sp.diags(self.sys.rK.diagonal(),format='csr')
		else:
			rK = self.sys.rK.copy() - np.diagflat(self.sys.rK.diagonal())

		kt = np.ravel(rK.sum(axis=0))
		selA,selI,selB=self.sys.selA*(kt>0.0),self.sys.selI*(kt>0.0),self.sys.selB*(kt>0.0)
		nA,nI,nB = selA.sum(),selI.sum(),selB.sum()
		mapA,mapI,mapB = Nr[selA], Nr[selI], Nr[selB]

		if self.sparse:
			iDI = sp.diags(1.0 / kt[selI], format='csr')
			iDB = sp.diags(1.0 / kt[selB], format='csr')
			BIB = rK[selI,:].transpose()[selB,:].transpose() * iDB
			BAB = rK[selA,:].transpose()[selB,:].transpose() * iDB
			BAI = rK[selA,:].transpose()[selI,:].transpose() * iDI
			BII = rK[selI,:].transpose()[selI,:].transpose() * iDI
			BBI = rK[selB,:].transpose()[selI,:].transpose() * iDI
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

		oneB = np.ones(nB)
		oneA = np.ones(nA)

		#print("OB",BIB.dot(oneB).min())

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
			x = spsolve(iGI,BIB.dot(oneB))
			y = spsolve(iGI.transpose(),BAI.transpose().dot(oneA))
		else:
			iGI = np.identity(nI) - BII
			try:
				x = np.linalg.solve(iGI,BIB.dot(oneB))
			except np.linalg.LinAlgError as err:
				x,resid,rank,s = np.linalg.lstsq(iGI,BIB.dot(oneB),rcond=None)
			try:
				y = np.linalg.solve(iGI.transpose(),BAI.transpose().dot(oneA))
			except np.linalg.LinAlgError as err:
				y = np.linalg.lstsq(iGI.transpose(),BAI.transpose().dot(oneA),rcond=None)[0]

		iDx = iDI.dot(x)
		bab = (BAI.dot(x)).sum()+(BAB.dot(oneB)).sum()

		yBIB = np.ravel(BIB.transpose().dot(y))

		# i.e. take largest ij rate to ~ remove state Boltzmann factor
		if self.sparse:
			mM = rK[selI,:][:,selI]
			mMt = rK[selI,:][:,selI].transpose()
			mM.data = np.vstack((mMt.data,mM.data)).max(axis=0)
			lM = -np.log(mM.data)
			mmm = np.percentile(mM.data,50)
		else:
			mM = np.vstack((rK[selI,:][:,selI].flatten(),rK[selI,:][:,selI].transpose().flatten())).max(axis=0)
			lM = -np.log(mM[mM>0.0])

		mix = min(1.0,np.exp(-lM.std()/lM.mean()))
		ko = 5.0*np.exp(-3.0)
		mink = ko*(1.0-mix) + mix*np.exp(-lM.mean())
		#print(np.exp(-lM.mean()),mM[mM>0.0].mean(),ko/(1.0+np.abs(lM.mean())),mink)
		#mink = np.exp(np.log(ko)*(1.0-mix) -lM.mean() * mix)
		#mink = ko/(1.0+np.abs(lM.mean()))
		nnI = piI.shape[0]
		fmapI = mapI
		iDBs = np.ravel(iDB.dot(oneB))

		yvB = np.zeros(yBIB.shape[0])
		for i in range(yBIB.shape[0]):
			yvB[i] = yBIB[i]*iDBs[i]

		yvI = np.zeros(nnI)
		for i in range(nnI):
			yvI = y[i] * iDx[i]

		cII = np.outer(y,iDx)-np.outer(np.ones(nnI),yvI)
		cAI = np.outer(np.ones(nA),iDx-yvI)
		cIB = np.outer(y,iDBs)-np.outer(np.ones(nnI),yvB)-np.outer(yvI,np.ones(nB)) # need phi.....


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

		mkf=1.0
		for m in range(cAI.shape[1]):
			# k_ml = 0.1 * min(1.0,pi_m/pi_l)
			kf = piA/piI[m]
			kf[kf>1.0] = 1.0
			cAI[:,m] *= kf * mink
			mkf = min(mkf,kf.min())
		mkf=1.0
		for l in range(cIB.shape[1]):
			# k_ml = 0.1 * min(1.0,pi_m/pi_l)
			kf = piI/piB[l]
			kf[kf>1.0] = 1.0
			cIB[:,l] *= kf * mink
			mkf = min(mkf,kf.min())

		mkf=1.0
		mkfib=1.0
		for l in np.arange(nnI):
			# k_ml = 0.1 * min(1.0,pi_m/pi_l)
			kf = piI/piI[l]
			kf[kf>1.0] = 1.0
			cII[:,l] *= kf * mink
			mkf = min(mkf,kf.min())

		##cAI[cAI<-bab] = -bab
		#cIB[cIB<-bab] = -bab
		#cII[cII<-bab] = -bab
		#cAI[cAI>1.0-bab] = 1.0-bab
		#cIB[cIB>1.0-bab] = 1.0-bab
		#cII[cII>1.0-bab] = 1.0-bab
		#print("\nmax:",cII.max(),cIB.max(),cAI.max(),mink,ed,np.log(mM.data).var()/mMm,"\n")

		#cII[self.probed[selI,:][:,selI].todense().astype(bool)] = 0.0
		if self.sparse:
			cAI *= 1.0 - self.probed[selA,:][:,selI].todense()
			cII *= 1.0 - self.probed[selI,:][:,selI].todense()
			cIB *= 1.0 - self.probed[selI,:][:,selB].todense()
		else:
			cAI[self.probed[selA,:][:,selI]] = 0.0
			cII[self.probed[selI,:][:,selI]] = 0.0
			cIB[self.probed[selI,:][:,selB]] = 0.0
			#cII *= 1.0 - self.probed[selI,:][:,selI]
			#cIB *= 1.0 - self.probed[selI,:][:,selB]


		c_tot = np.hstack(((np.triu(cII+cII.T)).flatten(),cAI.flatten(),cIB.flatten()))

		#print("c_tot:",c_tot.max(),c_tot.min(),c_tot.mean(),cmp,cmn,cm)

		sens = np.zeros(11)

		# sparsity
		sens[0] = np.exp(-0.0001*Na) + (1.0 - np.exp(-0.0001*Na))*ed

		# sigma^1_\pm
		sens[1] = c_tot.max()
		sens[2] = c_tot.min()

		# <sigma>
		sens[3] = c_tot[c_tot>0.0].mean() * sens[0] / (1.0-sens[0])
		cmn=0.0
		if (c_tot<0.0).sum()>0:
			sens[4] = c_tot[c_tot<0.0].mean() * sens[0] / (1.0-sens[0])
			sens[7] = c_tot[c_tot<0.0].sum() #* sens[0]
			cmn = -np.exp(np.log(np.abs(c_tot[c_tot<0.0])).mean())#*(cneg).sum()
		cmp = np.exp(np.log(np.abs(c_tot[c_tot>0.0])).mean())#cpos.sum()
		cm = (cmp*(c_tot>0.0).sum() + cmn*(c_tot<0.0).sum()) / c_tot.size


		sens[5] = c_tot.mean() * sens[0] / (1.0-sens[0])

		# \sigma_\pm
		sens[6] = c_tot[c_tot>0.0].sum() #* sens[0]

		# gsigma
		sens[8] = cmp
		sens[9] = cmn
		sens[10] = cm

		#sens[8] = int(len(c_tot)*sens[7])*0.5

		#c_tot[c_tot.argsort()[-int(sens[8]):]].sum()
		#c_tot[c_tot.argsort()[:int(sens[8])]].sum()



		fp_in = []
		for fp in c_tot.argsort()[::-1]:
			if np.abs(c_tot[fp])>1.0e-9*bab:
				fff = np.abs(c_tot[fp])
				if fp<nnI*nnI:
					p = [fmapI[fp//nnI],fmapI[fp%nnI]]
				elif fp<nnI*nnI+nA*nI:
					fp -= nnI*nnI
					p = [mapA[fp//nI],mapI[fp%nI]]
				else:
					fp -= nnI*nnI+nA*nI
					p = [mapI[fp//nB],mapB[fp%nB]]
				#print("HH",fff,p,(fff>1.0e-9*bab),self.probed[p[0],p[1]])
				if not self.probed[p[0],p[1]]:
					fp_in.append(p)

			if len(fp_in) >= npairs:
				break
		return fp_in,sens,bab

	def true_branching_probability(self,gt_check=True):
		kt = np.ravel(self.sys.K.sum(axis=0))
		selA,selI,selB=self.sys.selA*(kt>0.0),self.sys.selI*(kt>0.0),self.sys.selB*(kt>0.0)
		nA,nI,nB = selA.sum(),selI.sum(),selB.sum()
		oneB = np.ones(selB.sum())
		if gt_check:
			ikcon = self.sys.kcon.copy()
			ikcon[~self.sys.selI] = ikcon.max()
			rB, rN, retry = gt_seq(N=self.sys.N,rm_reg=selI,B=self.sys.B,trmb=1,order=ikcon)
			r_initial_states = self.sys.selB[~selI]
			r_final_states = self.sys.selA[~selI]
			gtBAB =( rB[r_final_states,:].tocsr()[:,r_initial_states].dot(oneB)).sum()
			print("\nGT BAB:",gtBAB)

		iDI = sp.diags(1.0 / kt[selI], format='csr')
		iDB = sp.diags(1.0 / kt[selB], format='csr')
		BIB = self.sys.K[selI,:][:,selB].dot(iDB)
		BAI = sp.csr_matrix(self.sys.K[selA,:][:,selI]).dot(iDI)
		BAB = np.ravel(self.sys.K[selA,:][:,selB].dot(iDB).dot(oneB)).sum()

		# inverse Green function
		iGI = sp.diags(np.ones(nI), format='csr')
		iGI -= sp.csr_matrix(self.sys.K[selI,:][:,selI]).dot(iDI)
		x = spsolve(iGI,BIB.dot(oneB))
		BAIB = np.ravel(BAI.dot(x)).sum()
		print("MATRIX:",BAB+BAIB)
		if gt_check:
			return BAB+BAIB, abs(gtBAB-BAIB-BAB)/(BAIB+BAB)
		else:
			return BAB+BAIB

	def new_true_branching_probability(self,gt_check=True):
		BAIB, BAB, cond = direct_solve(self.sys.B,self.sys.selB,self.sys.selA)
		print("DIRECT = %2.4g + %2.4g = %2.4g" % (BAB,BAIB,BAB+BAIB))
		res = (BAIB+BAB)*1.0
		if gt_check:
			rB, rD, rN, retry = gt_seq(N=self.sys.N,rm_reg=self.sys.selI,B=self.sys.B,D=self.sys.D.data,trmb=1)
			#print(rB)
			r_initial_states = self.sys.selB[~self.sys.selI]
			r_final_states = self.sys.selA[~self.sys.selI]
			BAIB, BAB, cond = direct_solve(rB,r_initial_states,r_final_states)
			print("GT = %2.4g + %2.4g = %2.4g" % (BAB,BAIB,BAB+BAIB))
			gtBAB = BAB+BAIB
			return res,abs(gtBAB-res)/res
		else:
			return res

	def new_estimated_branching_probability(self,gt_check=True):
		kt = np.ravel(self.sys.rK.sum(axis=0))
		N,selA,selI,selB=(kt>0.0).sum(),self.sys.selA[kt>0.0],self.sys.selI[kt>0.0],self.sys.selB[kt>0.0]
		B = sp.csr_matrix(self.sys.rK[(kt>0.0),:][:,(kt>0.0)]).dot(sp.diags(1.0/kt[kt>0.0],format='csr'))
		print("Bxx.max=",B.diagonal().min())
		BAIB, BAB, cond = direct_solve(B,selB,selA)
		print("EST DIRECT = %2.4g + %2.4g = %2.4g" % (BAB,BAIB,BAB+BAIB))
		rB, rN, retry = gt_seq(N=N,rm_reg=selI,B=B,trmb=1)#,condThresh=1.0e10,order=ikcon)
		print(rB)
		DD = rB.diagonal()
		r_initial_states = selB[~selI]
		r_final_states = selA[~selI]
		BAIB, BAB, cond = direct_solve(rB,r_initial_states,r_final_states)
		print("GT = %2.4g + %2.4g = %2.4g" % (BAB,BAIB,BAB+BAIB))

	def estimated_branching_probability(self,direct=True):
		kt = np.ravel(self.sys.rK.sum(axis=0))
		selA,selI,selB=self.sys.selA*(kt>0.0),self.sys.selI*(kt>0.0),self.sys.selB*(kt>0.0)
		nA,nI,nB = selA.sum(),selI.sum(),selB.sum()
		oneB = np.ones(nB)

		iDI = sp.diags(1.0 / kt[selI], format='csr')
		iDB = sp.diags(1.0 / kt[selB], format='lil')
		BIB = (self.sys.rK[selI,:][:,selB].dot(iDB))
		BAB = (self.sys.rK[selA,:][:,selB].dot(iDB))
		BAI = sp.csr_matrix(self.sys.rK[selA,:][:,selI]).dot(iDI)

		# inverse Green function
		iGI = sp.diags(np.ones(nI), format='csr')
		iGI -= sp.csr_matrix(self.sys.rK[selI,:][:,selI]).dot(iDI)

		x = spsolve(iGI,BIB.dot(oneB))
		BAB = np.ravel(BAB.dot(oneB)).sum()
		if not direct:
			BAB=0.0
		return np.ravel(BAI.dot(x)).sum() + BAB
