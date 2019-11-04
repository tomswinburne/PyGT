import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

sys.path.insert(0,"./lib")
from gt_tools import gt_seq

class sampler:
	def __init__(self,sys,max_d=20):
		self.sys = sys
		self.max_d = max_d
		self.probed=sp.lil_matrix(self.sys.K.shape,dtype=bool)
		for ii in range(sys.N):
			self.probed[ii,ii] = True
		self.sparseFactor = [.9,1.0] # assume v dense

	def initial_sample(self):
		# do a DNEB, find some paths to start from
		n_a = np.arange(self.sys.N)[self.sys.selA]
		n_b = np.arange(self.sys.N)[self.sys.selB]
		self.probed[self.sys.selA,:][:,self.sys.selA] = True
		self.probed[self.sys.selB,:][:,self.sys.selB] = True
		n_aib = []
		for t_i in n_a:
			n_aib.append(t_i)
			for t_f in n_b:
				n_aib.append(t_f)
				if not self.probed[t_i,t_f]:
					ia,fa,ka = self.sys.SaddleSearch(t_i,t_f)
					self.probed[t_i,t_f]=True
					self.probed[t_f,t_i]=True
					for ii in ia:
						n_aib.append(ii)
					for ff in fa:
						n_aib.append(ff)
					self.sys.add_connections(ia,fa,ka)
		print("AB SAMPLE DONE : %d" % self.sys.remaining_pairs())
		n_aib = list(set(n_aib))

		for t_i in n_aib[:10]:
			for t_f in n_aib[-10:]:
				if not self.probed[t_i,t_f]:
					ia,fa,ka = self.sys.DNEB(t_i,t_f,pM=sp.csr_matrix(self.probed))
					self.probed[t_i,t_f]=True
					self.probed[t_f,t_i]=True
					self.sys.add_connections(ia,fa,ka)

		print("INITIAL SAMPLE DONE : %d remaining pairs" % self.sys.remaining_pairs())



	def remaining_pairs(self):
		return self.sys.remaining_pairs()

	def sample(self,ignore_distance=False,npairs=20,nfilter=10000,ss=False):
		pairs,sens,ebp = self.sensitivity(ignore_distance=ignore_distance,npairs=4*npairs,nfilter=nfilter)
		c=0
		pc=0


		#print((self.sys.K[self.probed]>0.0).sum(),(self.sys.rK[self.probed]>0.0).sum(),self.probed.sum())
		#print(len(pairs))
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

	def sensitivity(self,epsilon = 1.0e-11,ignore_distance=True,npairs=20,nfilter=10000):
		M = self.sys.rK
		Nr = np.arange(self.sys.N)
		kt = np.ravel(M.sum(axis=0))

		selA,selI,selB=self.sys.selA*(kt>0.0),self.sys.selI*(kt>0.0),self.sys.selB*(kt>0.0)

		nA,nI,nB = selA.sum(),selI.sum(),selB.sum()
		mapA,mapI,mapB = Nr[selA], Nr[selI], Nr[selB]

		piA,piI,piB = self.sys.pi[selA],self.sys.pi[selI],self.sys.pi[selB]
		oneB = np.ones(nB)

		Nt = float(nA+nI+nB) * float(nA+nI+nB)
		Na = float(self.probed.sum() // 2)
		ed = float(self.sys.rK.nnz) / Nt
		#print(piI,piB)
		#dbf = min(1.0,np.exp(np.log(piI).mean() - np.log(piB).mean()))

		"""
		linear solves:
		(1-BII).x = iGI.x = BIB.piB
		y.(1-BII) = y.iGI = 1.BAI
		"""

		iDI = sp.diags(1.0 / kt[selI], format='csr')
		iDB = sp.diags(1.0 / kt[selB], format='lil')
		BIB = M[selI,:][:,selB].dot(iDB)
		BAB = M[selA,:][:,selB].dot(iDB)
		BAI = sp.csr_matrix(M[selA,:][:,selI]).dot(iDI)

		# inverse Green function
		iGI = sp.diags(np.ones(nI), format='csr')
		iGI -= sp.csr_matrix(M[selI,:][:,selI]).dot(iDI)
		x = spsolve(iGI,BIB.dot(oneB))
		y = spsolve(iGI.transpose(),BAI.transpose().dot(np.ones(selA.sum())))

		iDx = iDI.dot(x)
		bab = (BAI.dot(x)).sum() + (BAB.dot(oneB)).sum()
		yBIB = BIB.transpose().dot(y)




		mM = sp.csr_matrix(M[selI,:][:,selI])/2.0
		mM += mM.transpose() # i.e. take largest ij rate to ~ remove state Boltzmann factor
		mMm = np.log(mM.data).mean()
		mix = np.exp(-np.log(mM.data).var()/np.abs(mMm)+1.0)
		mink = np.exp(-3.0)*(1.0-mix) + np.exp(mMm) * mix

		#np.exp(np.log(mM.data).mean()) * 5.0 * (1.0-np.exp(-0.001*Na)) + np.exp(-0.001*Na)*np.exp(-3.0)
		# ~ mean of current energy barriers == geometric mean of rates * 5

		# TODO- compact on A?
		cAI = np.outer(np.ones(nA),(1.0-y)*iDx)
		for m in range(cAI.shape[0]):
			# k_ml = 0.1 * min(1.0,pi_m/pi_l)
			kf = piA[m]/piI
			kf[kf>1.0] = 1.0
			cAI[m,:] *= kf * mink


		cIB = np.outer(y,iDB.dot(piB))-np.outer(np.ones(nI),yBIB*(iDB.dot(piB))) - np.outer(y*iDx,1.0)
		for l in range(cIB.shape[1]):
			# k_ml = 0.1 * min(1.0,pi_m/pi_l)
			kf = piI/piB[l]
			kf[kf>1.0] = 1.0
			cIB[:,l] *= kf * mink

		"""
		filterI = (y*piI).argsort()[-100:] # actual indicies....
		y = y[filterI]
		iDx = iDx[filterI]
		piI = piI[filterI]
		fmapI = mapI[filterI]
		"""
		nnI = piI.shape[0]
		fmapI = mapI
		cII = np.outer(y,iDx)-np.outer(np.ones(nnI),y*iDx)

		#print(len(cII))

		"""
		if not ignore_distance:
			for iII in range(len(cII)):
				cII[iII] *= float(np.abs(mapI[iII//nI]-mapI[iII%nI])<=self.max_d) # not too far
		"""

		"""
		k_ml = min(1.0,pi_m/pi_l)*k_Uk_
		bcII_ml = (y_m-y_l)*iDx_l
		cII_ml = k_ml * bcII_ml + transpose
		"""
		for l in np.arange(nnI):
			# k_ml = 0.1 * min(1.0,pi_m/pi_l)
			kf = piI[l]/piI
			kf[kf>1.0] = 1.0
			cII[l,:] *= kf * mink

		#cII[self.probed[selI,:][:,selI].todense().astype(bool)] = 0.0
		cAI *= 1.0 - self.probed[selA,:][:,selI].todense()
		cII *= 1.0 - self.probed[selI,:][:,selI].todense()
		cIB *= 1.0 - self.probed[selI,:][:,selB].todense()

		c_tot = np.hstack(((np.triu(cII+cII.T)).flatten(),cAI.flatten(),cIB.flatten()))

		sens = np.zeros(9)
		sens[0] = c_tot.max()
		sens[1] = c_tot.min()

		sens[7] = np.exp(-0.001*Na) + (1.0 - np.exp(-0.001*Na))*ed

		sens[8] = int(len(c_tot)*sens[7])*0.5
		sens[2] = c_tot[c_tot.argsort()[-int(sens[8]):]].sum()
		sens[3] = c_tot[c_tot.argsort()[:int(sens[8])]].sum()

		sens[4] = c_tot[c_tot>0.0].sum()
		sens[5] = c_tot[c_tot<0.0].sum()
		sens[6] = c_tot.sum()

		#sens[4] *= sens[7]/(1.0-sens[7])
		#sens[5] *= sens[7]/(1.0-sens[7])
		#sens[6] *= sens[7]/(1.0-sens[7])


		# nNI*nNI, NA*NI, NI*NB
		# matrix M, shape (I,J). M[i][j] = M.flatten()[i*J+j] => M.flatten()[k] = M[k//J][k%J]

		fp_in = []
		for fp in np.abs(c_tot).argsort()[::-1]:
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
			rB, rN, retry = gt_seq(N=self.sys.N,rm_reg=self.sys.selI,B=self.sys.B,trmb=1,order=ikcon)
			r_initial_states = self.sys.selB[~self.sys.selI]
			r_final_states = self.sys.selA[~self.sys.selI]
			gtBAB =( rB[r_final_states,:].tocsr()[:,r_initial_states].dot(oneB)).sum()
			print("GT:",gtBAB)

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
		return BAB+BAIB


	def estimated_branching_probability(self):
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

		return np.ravel(BAI.dot(x)).sum() + np.ravel(BAB.dot(oneB)).sum()
