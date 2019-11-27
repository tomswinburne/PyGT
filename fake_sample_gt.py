import numpy as np
np.set_printoptions(linewidth=160)

from lib.ktn_io import * # tqdm / hacked scipy test
from lib.gt_tools import *

from lib.sampler import sampler
from lib.aib_system import aib_system


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
beta = 10.0 # overwritten if generate = False
selA = np.loadtxt('mina').astype(int)-1
selB = np.loadtxt('minb').astype(int)-1
Emax = None
sys = aib_system(path=data_dir,beta=beta,generate=generate,selA=selA,selB=selB,Emax=Emax)

print("beta: ",sys.beta,"N: ",sys.N)

"""
Boolean vectors selecting A and/or B regions
"""



#path, path_region = sys.find_path(sys.f.argmin(),sys.f.argmax(),depth=1,limit=4,strategy="RATE")
path, path_region = sys.find_path(sys.f[sys.selB].argmin(),sys.f[sys.selA].argmin(),depth=3,limit=4,strategy="DNEB")

sampler = sampler(sys)

sampler.initial_sample_path_region(np.arange(sys.N)[path_region],ncs=100)

print("NPathRegion=",path_region.sum())
#exit()

if gt_check:
	bab, gterr = sampler.new_true_branching_probability()

	if gterr>0.1: # 10%
		print("MATRIX TOO ILL CONDITIONED! ERROR=%2.2g" % gterr)
		exit()
else:
	bab = sampler.new_true_branching_probability(gt_check=False)

#sampler.new_estimated_branching_probability()


#bab = sampler.true_branching_probability(gt_check=True)

#exit()

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
npairs = 2
ssnpairs = 5
ss = 0 # >0 if we do single eneded search

rK = sampler.sys.rK.copy()
sampler.sys.rK = sampler.sys.K
pp,sens,ebab = sampler.sample(npairs=npairs,ss=ss) # sampling process. Returns
print("SENS: ",ebab)
sampler.sys.rK = rK


print("{: <4} {: <10} {: <10} {: <10} {: <10} {: <10} {: <10}".format(
"iter","estimated bp","expected bp","expected +","expected -","SingleEnded?","true bp"))


for ii in range(ncycles):

	orp = sampler.remaining_pairs() # DIVINE INFORMATION- number of remaining pairs

	""" sampling process. returns
	pp = number of pairs
	sens = vector of various net sensitivities and found sparsity
	ebab = predicted branching probability (to be compared to bab)
	"""
	pp,sens,ebab = sampler.sample(npairs=npairs,ss=ss) # sampling process. Returns

	nrp = sampler.remaining_pairs() # DIVINE INFORMATION- number of remaining pairs

	probe_compl = float(sampler.probed.sum()) / float(sampler.sys.K.shape[0]*sampler.sys.K.shape[1])
	ff.write("%d %d %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %d %10.10g\n" %\
	 	(ii,nrp,probe_compl,ebab,sens[0],sens[1],sens[2],sens[3],sens[4],sens[5],sens[6],sens[7],int(sens[8]),bab))

	sens[:-2] += ebab
	sens[:-2] /= bab
	print("{: <4} {: <10} {: <10} {: <10} {: <10} {: <10} {: <10}".format(
	"%d" % ii,"%1.4g" % (ebab/bab),"%1.4g" % sens[6],"%1.4g" % sens[4],"%1.4g" % sens[5],"%d" % ss,"%1.4g" % bab))

	"""
	if orp==nrp:
		ss = ssnpairs
	else:
		ss = 0
	"""

ff.close()

exit()
