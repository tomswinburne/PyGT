import numpy as np
np.set_printoptions(linewidth=160)

from lib.ktn_io import * # tqdm / hacked scipy test
from lib.gt_tools import *

from lib.sampler import sampler
from lib.aib_system_kgt import aib_system


print("\n\nGT REGULARIZATION + SAMPLING TESTS\n")

"""
First, load KTN data
generate=False looks for cache.
gt_check=True verifies linear algebra is stable
Nmax = maximum size of KTN
"""
data_dir = "KTN_data/LJ38/"

gt_check = False
generate = False
generatep = generate#False
beta = 10.0 # overwritten if generate = False
Emax = None#-169.5

sys = aib_system(path=data_dir,beta=beta,generate=generate,Emax=Emax)

"""
Boolean vectors selecting A and/or B regions
initial_state = 0 #sys.f.argmin()
final_state = 6 #sys.f.argmax()
raw_path, raw_path_region = sys.find_path(initial_state,final_state,depth=4,limit=10,strategy="RATE")
initial_path, initial_path_region = sys.find_path(initial_state,final_state,depth=1,limit=10,strategy="RATE")
"""

initial_states, final_states = np.zeros(sys.N,bool), np.zeros(sys.N,bool)
#initial_states[np.loadtxt('min_oct').astype(int)-1] = True
#final_states[np.loadtxt('min_ico').astype(int)-1] = True
initial_states[0] = True
final_states[5] = True
#final_states[6] = True

basins = initial_states + final_states
inter_region = ~basins

print("\n%d INITIAL STATES -> %d FINAL STATES\n" % (initial_states.sum(),final_states.sum()))

""" build path region """
if generatep:
	piM = sys.D.copy()
	piM.data = np.exp(sys.f)
	TE = sys.K.copy().tocsr() * piM
	TE.data = 1.0/TE.data
	raw_path_region = np.zeros(sys.N,bool)
	raw_path_region[initial_states] = True
	raw_path_region[final_states] = True
	raw_initial_path_region = raw_path_region.copy()
	for start_state in np.arange(sys.N)[initial_states]:
		if final_states.sum() > 5:
			end_states = np.random.choice(np.arange(sys.N)[final_states],size=10,replace=False)
		else:
			end_states = np.arange(sys.N)[final_states].copy()
		for end_state in end_states:
			raw_path_region += make_fastest_path(TE,start_state,end_state,depth=2,limit=10)[1]
			raw_initial_path_region += make_fastest_path(TE,start_state,end_state,depth=1,limit=5)[1]
	np.savetxt("test_rip",raw_initial_path_region,fmt="%d")
	np.savetxt("test_rp",raw_path_region,fmt="%d")
	sel = np.zeros(sys.N,bool)
	map = -np.ones(sys.N,int)
	map[raw_path_region] = np.arange(raw_path_region.sum())
	print(raw_path_region[initial_states].min(),raw_path_region[final_states].min(),map[initial_states].min(),map[final_states].min())
else:
	raw_path_region = np.loadtxt("test_rp").astype(bool)
	raw_initial_path_region = np.loadtxt("test_rip").astype(bool)



print("NPathRegion:",raw_path_region.sum())
print("NInitialPathRegion:",raw_initial_path_region.sum())


if generate:
	sys.gt(~raw_path_region,trmb=100)
	sys.save("test")
else:
	sys.load("test")


#from scipy.sparse.linalg import lgmres

#y = np.linalg.lstsq(iGI.transpose(),BAI.transpose().dot(oneA),rcond=None)[0]
#y = np.linalg.lstsq(iGI.transpose(),BAI.transpose().dot(oneA),rcond=None)[0]

"""
M.x = 0, 1.x = 1
M.x = 0, 1.x = 1
x.M^T M.x/2 + L(1.x-1) = 0
M^T M .x +
"""
import scipy.sparse as sp

iG = sp.diags(np.ones(sys.N),format='csr')-sys.B
K = iG * sp.diags(sys.D,format='csr')

print(K.shape,iG.shape,sys.D.shape)

evals ,evecs = sp.linalg.eigs(iG,k=3,which='SR')
print(evals)
np.savetxt("evecs",evecs)

exit()

iy = np.zeros(sys.N)



for i in range(sys.N):
	iy[i] = sys.D[i] * sys.pi[i]

y = sp.linalg.bicgstab(iG,np.zeros(sys.pi.shape),x0=iy,atol=1e-23)[0]
print(np.abs(iy/y-1.0).max())
x = np.zeros(sys.N)
for i in range(sys.N):
	x[i] = y[i]/sys.D[i]
x /= x.sum()
print(np.abs(sys.pi/x-1.0).max())
K = sp.diags(sys.D,format='csr')-sys.K
print((K.dot(sys.pi)).sum(),(K.dot(x)).sum())
import matplotlib.pyplot as plt
new_phi = np.outer(x,1.0/x).flatten()
old_phi = np.outer(sys.pi,1.0/sys.pi).flatten()
select = (old_phi>0.0) * (old_phi<1.0)
dd = new_phi[select] / old_phi[select] - 1.0

print(dd.min(),dd.max(),dd.mean())

#plt.hist(np.log(dd[dd>0.0]),bins=140,histtype='step')
#K = sp.diags(sys.D,format='csr')-sys.K
#x = sp.linalg.lgmres(K,np.zeros(sys.pi.shape),x0=sys.pi,atol=1e-18)[0]
#x /= x.sum()
#print(np.abs(x-sys.pi).max())


exit()


print(sys.N)
path_region = map[raw_path_region]
path_region = path_region[path_region>-1]

initial_path_region = map[raw_initial_path_region]
initial_path_region = initial_path_region[initial_path_region>-1]

selB = np.zeros(sys.N,bool)
selA = np.zeros(sys.N,bool)

selB[map[initial_states]] = True
selA[map[final_states]] = True


sys.define_AB_regions(selA,selB)
sampler = sampler(sys)
sampler.make_dense()


sampler.initial_sample_path_region(np.arange(sys.N)[initial_path_region],ncs=5000)

print("NPathRegion=",path_region.size)

if gt_check:
	bab, gterr = sampler.new_true_branching_probability(gt_check=True)
	if gterr>0.1: # 10%
		print("MATRIX TOO ILL CONDITIONED! ERROR=%2.2g" % gterr)
		exit()
else:
	bab = sampler.new_true_branching_probability(gt_check=False)


""" open output file """
name = data_dir.split("/")[-1-int(data_dir[-1]=="/")]
ff = open('output/pab_converge_%s' % name,'w')

"""
	parameters for sampling. We try DNEBS,
	if these return nothing we try single ended searches
	ncycles = number of cycles
	npairs = number of DNEBS per cycle
	ssnpairs = number of single ended searches per cycle, if performed
"""

ncycles = 3000
npairs = 50

ssnpairs = 5
ss = 0 # >0 if we do single ended search
print("bab:",bab,"\n\n")

rK = sampler.sys.rK.copy()
sampler.sys.rK = sampler.sys.K
pp,sens,ebab = sampler.sensitivity(npairs=npairs) # sampling process. Returns
print("(e)bab:",ebab,"\n\n")

sampler.sys.rK = rK.copy()
pp,sens,ebab = sampler.sensitivity(npairs=npairs) # sampling process. Returns
print("ebab:",ebab,"\n\n")
del rK

print("{: <4} {: <4} {: <6} {: <6} {: <6} {: <6} {: <6} {: <6}".format(
	"iter","rp","estimated bp","expected bp","expected +","expected -","SingleEnded?","true bp"))

for ii in range(ncycles):

	orp = sampler.remaining_pairs() # DIVINE INFORMATION- number of remaining pairs

	""" sampling process. returns
		pp = number of pairs
		sens = vector of various net sensitivities and found sparsity
		ebab = predicted branching probability (to be compared to bab)
	"""

	pp,sens,ebab = sampler.sample(npairs=npairs,ss=ss) # sampling process. Returns pairs for sampling
	nrp = sampler.remaining_pairs() # DIVINE INFORMATION- number of remaining pairs

	"""
		0-4: ii,nrp,pc, ebab,bab
		sens[0] : <sparsity>
		sens[1-2] : sigma^1
		sens[3-5] : <c> phi/(1-phi)
		sens[6-7] : sigma
		sens[8-10] : gsigma
	"""

	probe_compl = float(sampler.probed.sum()) / float(sampler.sys.K.shape[0]*sampler.sys.K.shape[1])
	data_floats = [probe_compl,ebab,bab,sens[0],sens[1],sens[2],sens[3],sens[4],sens[5],sens[6],sens[7],sens[8],sens[9],sens[10]]

	data_string = "%d %d" % (ii, nrp)
	for df in data_floats:
		data_string+=" %10.10g" % df
	ff.write(data_string+"\n")

	sens[1:] /= bab

	print("{: <4} {: <4} {: <4} {: <6} {: <6} {: <6} {: <6}  {: <6} {: <6} {: <6}  {: <6} {: <6} {: <6}  {: <6} {: <1} {: <6}".format(
	"%d" % ii,"%d" % nrp,"%f" % probe_compl,\
	"<C>: %1.3g" % (ebab/bab),\
	"<st>: %1.3g" % (sens[6]+sens[7]),"st+: %1.3g" % sens[6],"st-: %1.3g" % sens[7],\
	"<s1>: %1.3g" % (sens[1]+sens[2]),"s1+: %1.3g" % sens[1],"s1-: %1.3g" % sens[2],\
	"<dC>: %1.3g" % sens[3],"<dC+>: %1.3g" % sens[4],"<dC->: %1.3g" % sens[5],\
	"spar: %1.3g" % sens[0],"%d" % ss,"%1.3g" % bab))

	"""
	if orp==nrp:
		ss = ssnpairs
	else:
		ss = 0
	"""
ff.close()
exit()
