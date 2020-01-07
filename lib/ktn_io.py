import os,time,sys
import numpy as np
from scipy.sparse import csgraph, csr_matrix, csc_matrix, eye, save_npz, load_npz, diags
os.system('mkdir -p cache')
os.system('mkdir -p output')
import warnings
from lib.gt_tools import gt_seq, make_fastest_path

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

class printer:
	def __init__(self,screen=False,file=None,timestamp=True):
		self.screen = screen
		self.file = file
		self.t = timer()
		self.timestamp = timestamp
		if not file is None:
			f = open(file,'w')
	def __call__(self,str,dt=True):
		if self.timestamp and dt:
			str += ", dt: %4.4gs" % self.t()
		str = "\t" + str + "\n"
		if not self.file is None:
			f.write(str)
		if self.screen:
			print(str)
	def restart(self):
		self.t()

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

def load_mat(path='../data/LJ38/raw/',Nmax=None,Emax=None,beta=1.0,mytype=np.float64,discon=False,histo=False,gt=False,TE=False):

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
	if not Nmax is None:
		N = min(Nmax,N)

	Efilter = GSD['E']<GSD['E'].max()+1.0

	sels = (TSD['I']<N) * (TSD['F']<N) * (TSD['I']!=TSD['F'])
	if not Emax is None:
		sels *= GSD['E'][TSD['I']]<Emax
		sels *= GSD['E'][TSD['F']]<Emax
		sels *= TSD['E']<Emax
	TSD = TSD[sels]
	GSD = GSD[:N]

	Emin = GSD['E'].min().copy()
	GSD['E'] -= Emin
	TSD['E'] -= Emin

	""" Calculate rates """
	BF = beta*GSD['E']-GSD['S']
	i = np.hstack((TSD['I'],TSD['F']))
	f = np.hstack((TSD['F'],TSD['I']))
	du = np.hstack((TSD['E']-GSD[TSD['I']]['E'],TSD['E']-GSD[TSD['F']]['E']))
	ds = np.hstack((TSD['S']-GSD[TSD['I']]['S'],TSD['S']-GSD[TSD['F']]['S']))
	if TE:
		te = np.hstack((TSD['E'],TSD['E']))


	sel = i!=f
	i = i[sel]
	f = f[sel]
	du = du[sel]
	ds = ds[sel]
	if TE:
		te = te[sel]

	if histo:
		return du,ds

	"""+ds Fill matricies: K_ij = rate(j->i), K_ii==0. iD_jj = 1/(sum_iK_ij) """
	data = np.zeros(du.shape,dtype=mytype)
	data[:] = np.exp(-beta*du+ds)
	K = csr_matrix((data,(f,i)),shape=(N,N),dtype=mytype)
	data[:] = np.exp(-beta*du+ds)

	if discon:
		# take min value, i.e. crank it....
		dS = csr_matrix((np.exp(10.0*ds),(f,i)),shape=(N,N),dtype=mytype)
		# take max value, i.e. crank it....
		dU = csr_matrix((np.exp(-10.0*du),(f,i)),shape=(N,N),dtype=mytype)
		dS.data = 0.1*np.log(dS.data)
		dU.data = -0.1*np.log(dU.data)

	#print(dU.max(),dS.max(),-np.log(K.data).min())

	""" connected components """
	nc,cc = csgraph.connected_components(K)
	sum = np.zeros(nc,int)
	mc = 0
	for j in range(nc):
		sum[j] = (cc==j).sum()
	sel = cc==sum.argmax()

	if TE:
		# B is not B
		TEM = csr_matrix((np.exp(-beta*te),(f,i)),shape=(N,N),dtype=mytype)
		TEM = TEM.tocsc()[sel,:].tocsr()[:,sel]

	K,N = K.tocsc()[sel,:].tocsr()[:,sel], sel.sum()

	if discon:
		dS = dS.tocsc()[sel,:].tocsr()[:,sel]
		dU = dU.tocsc()[sel,:].tocsr()[:,sel]

	GSD = GSD[sel]
	#print("E[0]=%1.4g,E[6]=%1.4g" % (GSD['E'][0],GSD['E'][6]))
	kt = np.ravel(K.sum(axis=0))
	iD = csr_matrix((1.0/kt,(np.arange(N),np.arange(N))),shape=(N,N),dtype=mytype)
	D = csr_matrix((kt,(np.arange(N),np.arange(N))),shape=(N,N),dtype=mytype)

	if discon:
		return dU, dS, GSD['E'], GSD['S']
	if not TE:
		B = K.dot(iD)
		return B, K, D, N, GSD['E'],GSD['S'],Emin
	else:
		return TEM, K, D, N, GSD['E'],GSD['S'],Emin



def load_mat_discon(path="../../data/LJ38",Nmax=8000,Emax=None):
	dU, dS, U, S = load_mat(path,beta=1.0,Nmax=Nmax,Emax=Emax,discon=True)
	return dU, dS, U, S, U.shape[0]

def load_save_mat_histo(path="../../data/LJ38",Nmax=8000,Emax=None,generate=True):
	name = path.split("/")[-1]
	if len(name)==0:
		name = path.split("/")[-2]
	if not generate:
		try:
			dU = np.loadtxt('cache/temp_%s_dU.txt' % name)
			dS = np.loadtxt('cache/temp_%s_dS.txt' % name)
		except IOError:
			generate = True
			print("no files found, generating...")
	if generate:
		  print("Generating....")
		  dU, dS = load_mat(path,beta=1.0,Nmax=Nmax,Emax=Emax,histo=True)
		  np.savetxt('cache/temp_%s_dU.txt' % name,dU)
		  np.savetxt('cache/temp_%s_dS.txt' % name,dS)

	return dU,dS

def load_save_mat_discon(path="../../data/LJ38",Nmax=8000,Emax=None,generate=True):
	name = path.split("/")[-1]
	if len(name)==0:
		name = path.split("/")[-2]

	if not generate:
		try:
			dU = load_npz('cache/temp_%s_dU.npz' % name)
			dS = load_npz('cache/temp_%s_dS.npz' % name)
			US = np.loadtxt('cache/temp_%s_US.txt' % name)
		except IOError:
			generate = True
			print("no files found, generating...")


	if generate:
		  print("Generating....")
		  dU, dS, U, S = load_mat(path,beta=1.0,Nmax=Nmax,Emax=Emax,discon=True)
		  US = np.zeros((U.shape[0],2))
		  US[:,0] = U
		  US[:,1] = S
		  np.savetxt('cache/temp_%s_US.txt' % name,US)
		  save_npz('cache/temp_%s_dU.npz' % name,dU)
		  save_npz('cache/temp_%s_dS.npz' % name,dS)

	return dU,dS,US[:,0],US[:,1]

def load_save_mat(path="../../data/LJ38",beta=5.0,Nmax=8000,Emax=None,generate=True,TE=False):
	name = path.split("/")[-1]
	if len(name)==0:
		name = path.split("/")[-2]
	if not generate:
		try:
			B = load_npz('cache/temp_%s_%2.6g_B.npz' % (name,beta))
			D = load_npz('cache/temp_%s_%2.6g_D.npz' % (name,beta))
			K = load_npz('cache/temp_%s_%2.6g_K.npz' % (name,beta))
			USEB = np.loadtxt('cache/temp_%s_%2.6g_USEB.txt' % (name,beta))
		except IOError:
			generate = True
			print("no files found, generating...")

	if generate:
	  print("Generating....")
	  B, K, D, N, U, S, Emin = load_mat(path,beta=beta,Nmax=Nmax,Emax=Emax,TE=TE)
	  USEB = np.zeros((U.shape[0]+1,2))
	  USEB[-1][0] = beta
	  USEB[-1][1] = Emin
	  USEB[:-1,0] = U
	  USEB[:-1,1] = S
	  np.savetxt('cache/temp_%s_%2.6g_USEB.txt' % (name,beta),USEB)
	  save_npz('cache/temp_%s_%2.6g_B.npz' % (name,beta),B)
	  save_npz('cache/temp_%s_%2.6g_K.npz' % (name,beta),K)
	  save_npz('cache/temp_%s_%2.6g_D.npz' % (name,beta),D)

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

	return beta, B, K, D, N, U, S, kt, kcon, Emin


def load_save_mat_gt(selA,selB,beta=10.0,path="../../data/LJ38",Nmax=None,Emax=None,generate=True):
	name = path.split("/")[-1]
	if len(name)==0:
		name = path.split("/")[-2]

	if not generate:
		try:
			B = load_npz('cache/temp_%s_B.npz' % name)
			D = load_npz('cache/temp_%s_D.npz' % name)
			F = np.loadtxt('cache/temp_%s_F.txt' % name)
			map = np.loadtxt('cache/temp_%s_M.txt' % name,dtype=int)
		except IOError:
			generate = True
			print("no files found, generating...")


	if generate:
		print("Generating....")
		B,D,F,map = load_mat_gt(selA,selB,path,beta=beta,Nmax=Nmax,Emax=Emax)
		np.savetxt('cache/temp_%s_F.txt' % name,F)
		np.savetxt('cache/temp_%s_M.txt' % name,map,fmt="%d")
		save_npz('cache/temp_%s_B.npz' % name,B)
		save_npz('cache/temp_%s_D.npz' % name,D)

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

	rm_reg = np.ones(N,bool) # remove all

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

	rm_reg[f_keep] = False # i.e. ~rm_reg survives

	kept = np.zeros(oN,bool)
	kept[ccsel] = ~rm_reg # i.e. selects those which were kept

	map = -np.ones(oN,int)
	map[kept] = np.arange(kept.sum())

	B,D,N,retry = gt_seq(N=N,rm_reg=rm_reg,B=B,D=D,trmb=1,order=None)
	if dense:
		B = csr_matrix(B)

	return B,diags(D,format='csr'),F[~rm_reg],map
