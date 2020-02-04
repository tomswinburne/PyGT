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
	has_tqdm=True
except:
	print("Install tqdm package for pretty progress bars!")
	has_tqdm=False



def direct_solve(B,initial_states,final_states,D=None):
	B=B.tocsc()

	basins = initial_states+final_states
	inter_region = ~basins
	pi = np.ones(initial_states.sum()) #/ initial_states.sum()
	BAB = (B[final_states,:].tocsr()[:,initial_states].dot(pi)).sum()
	cond = 1.0
	NI = inter_region.sum()
	if NI>0:
		BI = B[inter_region,:].tocsr()[:,inter_region]
		BAI = B[final_states,:].tocsr()[:,inter_region].transpose().dot(np.ones(final_states.sum()))
		BIB = B[inter_region,:].tocsr()[:,initial_states].dot(pi)
		iGI = eye(NI,format="csr") - BI
		"""
		if has_umfpack_hack:
			x,cond = spsolvecond(iGI,BIB,giveCond=True)
		else:
			x = spsolve(iGI,BIB)
		"""
		x = spsolve(iGI,BIB)
		BABI = BAI.dot(x)
	else:
		BABI = 0.0
	if D is None:
		return BABI,BAB
	#else:



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

def gt(B,sel,condThresh=1.0e8):

	Bxx = B[sel,:].transpose()[sel,:].transpose()
	Bxj = B[sel,:].transpose()[~sel,:].transpose()
	Bix = B[~sel,:].transpose()[sel,:].transpose()
	Bij = B[~sel,:].transpose()[~sel,:].transpose()

	if sel.sum()>1:
		# Get Bxx
		Bd = Bxx.diagonal()
		# Get sum Bix for i not equal x
		Bxxnd = Bxx - diags(Bd,format="csr")
		Bs = np.ravel(Bix.sum(axis=0))
		if Bs.max()<1.0e-25:
			return B,False
		Bs+=np.ravel(Bxxnd.sum(axis=0))
		Bs[Bd<0.99] = 1.0-Bd[Bd<0.99]
		iGxx = diags(Bs,format="csr") - Bxxnd
		I = eye(sel.sum(),format=iGxx.format)
		"""
		if has_umfpack_hack:
			Gxx,cond = spsolvecond(iGxx,I,giveCond=True)
		else:
			Gxx = spsolve(iGxx,I)
			cond = 1.0
		if cond>condThresh:
			return B,False
		"""
		Gxx = spsolve(iGxx,I)
	else:
		_b_xx = Bxx.data.sum()
		if _b_xx>0.99:
			b_xx = Bix.sum()
		else:
			b_xx = 1.0-_b_xx
		Gxx = diags(np.r_[1.0/b_xx],format="csr")

	return Bij + Bix*Gxx.tocsr()*Bxj,True

def gtD(B,iD,sel,timeit=False,dense=False):
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
	iDjj = iD[~sel]
	iDxx = iD[sel]

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
			return B,iD,False
		iDG = Gxx.transpose().dot(iDxx).transpose()
		iDjj += np.ravel(Bxj.transpose().dot(iDG)).flatten()

		if timeit:
			t("Gxx inv + iD mult.")

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


def gt_seq(N,rm_reg,B,D=None,trmb=1,condThresh=1.0e10,order=None,Ndense=500,force_sparse=True,screen=False,retK=False):
	if D is None:
		retK=False

	rmb=trmb
	retry=0
	NI = rm_reg.sum()
	if not D is None:
		iD = 1.0/np.ravel(D)

	#B.tolil()
	if screen:
		print("GT regularization removing %d states:" % NI)
	if has_tqdm:
		pbar = tqdm(total=NI,leave=False,mininterval=0.0)
	tst = 0.0
	tmt = 0.0
	tc = 0

	pass_over = np.empty(0,bool)
	pobar = None
	dense = False
	dense_onset = 0
	while NI>0:

		if N<Ndense and not dense:
			dense = True
			density = float(B.nnz) /  float( B.shape[0]*B.shape[1])
			dense_onset = B.shape[0]
			B = B.todense()

		rm = np.zeros(N,bool)
		if pass_over.sum()>0:
			Bd = np.ravel(B.diagonal())
			Bd[~pass_over] = Bd.min()-1.0
			Bd[~rm_reg] = Bd.min()-1.0
			rm[Bd.argmax()] = True
			if not pobar is None:
				pobar.update(1)
		else:
			if order is None:
				if not dense:
					order = B.indptr[1:]-B.indptr[:-1]
				else:
					order = np.linspace(0.,1.0,B.shape[0])
			if not pobar is None:
				pobar.close()
			pobar = None
			order[~rm_reg] = order.max()+1
			rm[order.argsort()[:min(rmb,NI)]] = True

		if not D is None:
			B, iD, success = gtD(B,iD,rm,timeit=False,dense=dense)
		else:
			B, success = gt(B,rm,condThresh=condThresh)
		if success:
			if has_tqdm:
				pbar.update(rm.sum())
			N -= rm.sum()
			NI -= rm.sum()
			rm_reg = rm_reg[~rm]
			if not (order is None):
				order = order[~rm]
			if pass_over.sum()>0:
				pass_over = pass_over[~rm]
			rmb = 1 + (trmb-1)*int(pass_over.sum()==0)
		else:
			pass_over = rm
			rmb = 1
			retry += 1
			if has_tqdm:
				pobar = tqdm(total=rm.sum(),leave=False,mininterval=0.0,desc="STATE-BY-STATE GT")
	if has_tqdm:
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
	if not D is None:
		D = 1.0/iD
		D = np.ravel(D).flatten()
		if retK:
			Bd = np.ravel(B.diagonal()) # only the diagonal (Bd_x = B_xx)
			Bn = B - diags(Bd) # B with no diagonal (Bn_xx = 0, Bn_xy = B_xy)
			Bn.eliminate_zeros()
			Bnd = np.ravel(Bn.sum(axis=0)) # Bnd_x = sum_x!=y B_yx = 1-B_xx
			nBd = np.zeros(N)
			nBd[Bd>0.99] = Bnd[Bd>0.99] 
			nBd[Bd<0.99] = 1.0-Bd[Bd<0.99]
			omB = diags(nBd) - Bn # 1-B
			K = omB.dot(diags(D)) # (1-B).D = K ( :) )
			return B,D,K,N,retry
		return B,D,N,retry

	else:
		return B,N,retry
