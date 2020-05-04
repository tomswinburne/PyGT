import os,time,sys
from io import StringIO
import numpy as np
from scipy.sparse import csgraph, csr_matrix, csc_matrix, eye, save_npz, load_npz, diags
os.system('mkdir -p cache')
os.system('mkdir -p output')
import warnings
from lib.gt_tools import gt_seq, make_fastest_path
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

def load_AB(data_path,index_sel=None):
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

def load_mat(path='../data/LJ38/raw/',Nmax=None,Emax=None,beta=1.0,screen=False,discon=False):

	""" load data """
	GSD = np.loadtxt(os.path.join(path,'min.data'),\
		dtype={'names': ('E','S','DD','RX','RY','RZ'),\
		'formats': (float,float,int,float,float,float)})

	TSD = np.loadtxt(os.path.join(path,'ts.data'),\
		dtype={'names': ('E','S','DD','F','I','RX','RY','RZ'),\
		'formats': (float,float,int,int,int,float,float,float)})

	#TSD = TSD[TSD['I']!=TSD['F']] # remove self transitions??


	TSD['I'] = TSD['I']-1
	TSD['F'] = TSD['F']-1

	N = max(TSD['I'].max()+1,TSD['F'].max()+1)

	if not Nmax is None:
		N = min(Nmax,N)

	sels = (TSD['I']<N) * (TSD['F']<N) * (TSD['I']!=TSD['F'])
	if not Emax is None:
		sels *= GSD['E'][TSD['I']]<Emax
		sels *= GSD['E'][TSD['F']]<Emax
		sels *= TSD['E']<Emax
	TSD = TSD[sels]
	GSD = GSD[:N]


	print("N,N_TS:",GSD.size,TSD.size)
	Emin = GSD['E'].min().copy()
	Smin = min(GSD['S'].min().copy(),TSD['S'].min().copy())
	GSD['E'] -= Emin
	TSD['E'] -= Emin
	GSD['S'] -= Smin
	TSD['S'] -= Smin


	""" Calculate rates """
	i = np.hstack((TSD['I'],TSD['F']))
	f = np.hstack((TSD['F'],TSD['I']))
	du = np.hstack((TSD['E']-GSD[TSD['I']]['E'],TSD['E']-GSD[TSD['F']]['E']))

	ds = np.hstack((GSD[TSD['I']]['S']-TSD['S'],GSD[TSD['F']]['S']-TSD['S']))/2.0

	dc = np.hstack((GSD[TSD['I']]['DD']/TSD['DD'],GSD[TSD['F']]['DD']/TSD['DD']))/2.0/np.pi
	ds += np.log(dc)

	s = GSD['S']/2.0 + np.log(GSD['DD'])

	"""+ds Fill matricies: K_ij = rate(j->i), K_ii==0. iD_jj = 1/(sum_iK_ij) """

	data = np.zeros(du.shape)
	if discon:
		ddu = du.copy()
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
	nc,cc = csgraph.connected_components(K)
	sum = np.zeros(nc,int)
	mc = 0
	for j in range(nc):
		sum[j] = (cc==j).sum()
	sel = cc==sum.argmax()

	if screen:
		print("Connected Clusters: %d, 1st 400 states in largest cluster: %d" % (nc,sel[:400].min()))
	oN=N

	K,N = K[sel,:][:,sel], sel.sum()

	if discon:
		DU = DU[sel,:][:,sel]

	if screen:
		print("cc: N: %d->%d" % (oN,N),GSD.shape,sel.shape)


	GSD = GSD[sel]
	s = -GSD['S']/2.0 - np.log(GSD['DD'])

	if discon:
		return N,GSD['E'],DU

	kt = np.ravel(K.sum(axis=0))
	iD = csr_matrix((1.0/kt,(np.arange(N),np.arange(N))),shape=(N,N))
	D = csr_matrix((kt,(np.arange(N),np.arange(N))),shape=(N,N))

	B = K.dot(iD)
	return B, K, D, N, GSD['E'], s, Emin, sel


def load_save_mat(path="../../data/LJ38",beta=5.0,Nmax=8000,Emax=None,generate=True,TE=False,screen=False):
	name = path.split("/")[-1]
	if len(name)==0:
		name = path.split("/")[-2]
	if not generate:
		try:
			B = load_npz('cache/temp_%s_%2.6g_B.npz' % (name,beta))
			D = load_npz('cache/temp_%s_%2.6g_D.npz' % (name,beta))
			K = load_npz('cache/temp_%s_%2.6g_K.npz' % (name,beta))
			USEB = np.loadtxt('cache/temp_%s_%2.6g_USEB.txt' % (name,beta))
			sel = np.loadtxt('cache/temp_%s_%2.6g_sel.txt' % (name,beta)).astype(bool)
		except IOError:
			generate = True
			if screen:
				print("no files found, generating...")

	if generate:
		if screen:
			print("Generating....")
		B, K, D, N, U, S, Emin, sel = load_mat(path,beta=beta,Nmax=Nmax,Emax=Emax,screen=screen)
		USEB = np.zeros((U.shape[0]+1,2))
		USEB[-1][0] = beta
		USEB[-1][1] = Emin
		USEB[:-1,0] = U
		USEB[:-1,1] = S
		np.savetxt('cache/temp_%s_%2.6g_USEB.txt' % (name,beta),USEB)
		np.savetxt('cache/temp_%s_%2.6g_sel.txt' % (name,beta),sel)
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

	return beta, B, K, D, N, U, S, kt, kcon, Emin, sel



def load_save_mat_gt(keep_ind,beta=10.0,path="../../data/LJ38",Nmax=None,Emax=None,generate=True):
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
		B,D,F,map = load_mat_gt(keep_ind,path,beta=beta,Nmax=Nmax,Emax=Emax)
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
