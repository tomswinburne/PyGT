import os,time,sys
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

def load_mat(path='../data/LJ38/raw/',Nmax=None,Emax=None,beta=1.0):

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

	if Nmax is None:
		if not Nmax is None:
			N = min(Nmax,N)


		sels = (TSD['I']<N) * (TSD['F']<N) * (TSD['I']!=TSD['F'])
		if not Emax is None:
			sels *= GSD['E'][TSD['I']]<Emax
			sels *= GSD['E'][TSD['F']]<Emax
			sels *= TSD['E']<Emax
		TSD = TSD[sels]
		GSD = GSD[:N]

	#print("N,N_TS:",GSD.size,TSD.size)
	Emin = GSD['E'].min().copy()
	GSD['E'] -= Emin
	TSD['E'] -= Emin

	""" Calculate rates """
	BF = beta*GSD['E']-GSD['S']
	i = np.hstack((TSD['I'],TSD['F']))
	f = np.hstack((TSD['F'],TSD['I']))
	du = np.hstack((TSD['E']-GSD[TSD['I']]['E'],TSD['E']-GSD[TSD['F']]['E']))
	ds = np.hstack((TSD['S']-GSD[TSD['I']]['S'],TSD['S']-GSD[TSD['F']]['S']))
	dc = np.hstack((
	factorial(GSD[TSD['I']]['DD'])/factorial(TSD['DD']),
	factorial(GSD[TSD['F']]['DD'])/factorial(TSD['DD'])
	))

	"""
	sel = i!=f
	i = i[sel]
	f = f[sel]
	du = du[sel]
	ds = ds[sel]
	dc = dc[sel]
	"""

	"""+ds Fill matricies: K_ij = rate(j->i), K_ii==0. iD_jj = 1/(sum_iK_ij) """
	data = np.zeros(du.shape)
	data[:] = np.exp(-beta*du+ds)

	fNi = f*N+i
	fNi_u = np.unique(fNi)
	d_u = np.r_[[data[fNi==i].sum() for i in fNi_u]]
	f_u = fNi_u//N
	i_u = fNi_u%N
	K = csr_matrix((d_u,(f_u,i_u)),shape=(N,N))


	""" connected components """
	nc,cc = csgraph.connected_components(K)
	sum = np.zeros(nc,int)
	mc = 0
	for j in range(nc):
		sum[j] = (cc==j).sum()
	sel = cc==sum.argmax()

	oN=N.copy()
	K,N = K.tocsc()[sel,:].tocsr()[:,sel], sel.sum()
	#print("cc: N: %d->%d" % (oN,N))


	GSD = GSD[sel]
	kt = np.ravel(K.sum(axis=0))
	iD = csr_matrix((1.0/kt,(np.arange(N),np.arange(N))),shape=(N,N))
	D = csr_matrix((kt,(np.arange(N),np.arange(N))),shape=(N,N))

	B = K.dot(iD)
	return B, K, D, N, GSD['E'],GSD['S'],Emin


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
		except IOError:
			generate = True
			if screen:
				print("no files found, generating...")

	if generate:
		if screen:
			print("Generating....")
		B, K, D, N, U, S, Emin = load_mat(path,beta=beta,Nmax=Nmax,Emax=Emax)
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
