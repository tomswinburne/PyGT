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
data_dir = "KTN_data/LJ13/"

gt_check = False
generate = True
direct=True
beta = 10.0 # overwritten if generate = False
Emax = None#-169.5#+173.9

"""
Boolean vectors selecting A and/or B regions
"""
sys = aib_system(path=data_dir,beta=beta,Emax=Emax,generate=generate)



print("beta: ",sys.beta,"N: ",sys.N)
indB = np.loadtxt('mina').astype(int)-1
indA = np.loadtxt('minb').astype(int)-1

selB = np.zeros(sys.N,bool)
selA = np.zeros(sys.N,bool)

selB[indB] = True
selA[indA] = True

minA = np.arange(sys.N)[selA][sys.pi[selA].argsort()[-4:]]
minB = np.arange(sys.N)[selB][sys.pi[selB].argsort()[-4:]]

path_region = np.zeros(sys.N,np.bool)
for _minA in minA:
	for _minB in minB:
		path, _path_region = make_fastest_path(sys.udG,_minB,_minA,depth=3,limit=4)
		path_region += _path_region

print(np.arange(sys.N)[path_region].shape)

print("NPathRegion=",path_region.sum())
#exit()

sampler = sampler(sys)
sys.define_AB_regions(selA,selB)
sampler.initial_sample_path_region(np.arange(sys.N)[path_region],ncs=100)

exit()

if gt_check:
	bab, gterr = sampler.new_true_branching_probability(direct=direct)

	if gterr>0.1: # 10%
		print("MATRIX TOO ILL CONDITIONED! ERROR=%2.2g" % gterr)
		exit()
else:
	bab = sampler.new_true_branching_probability(gt_check=False,direct=direct)

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
npairs = 10
ssnpairs = 5
ss = 0 # >0 if we do single eneded search

rK = sampler.sys.rK.copy()
sampler.sys.rK = sampler.sys.K
pp,sens,ebab = sampler.sample(npairs=npairs,ss=ss,ignore_distance=True,direct=direct) # sampling process. Returns
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
	pp,sens,ebab = sampler.sample(npairs=npairs,ss=ss,ignore_distance=True,direct=direct) # sampling process. Returns

	nrp = sampler.remaining_pairs() # DIVINE INFORMATION- number of remaining pairs

	probe_compl = float(sampler.probed.sum()) / float(sampler.sys.K.shape[0]*sampler.sys.K.shape[1])

	ff.write("%d %d %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %d %10.10g\n" %\
	 	(ii,nrp,probe_compl,ebab,sens[0],sens[1],sens[2],sens[3],sens[4],sens[5],sens[6],sens[7],int(sens[8]),bab))

	sens[:-2] += ebab
	sens[:-2] /= bab
	#if orp==nrp:
	#	ss = ssnpairs
	#else:
	#	ss = 0
	#print('{:04d :04d :04d :04d :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f}'.format(ii,orp,orp-nrp,ss,ebab/bab,sens[0],sens[1],sens[2],sens[3],sens[4],sens[5],sens[6],sens[7],true_dense))
	print("{: <4} {: <10} {: <10} {: <10} {: <10} {: <10} {: <10}".format(
	"%d" % ii,"%1.4g" % (ebab/bab),"%1.4g" % sens[6],"%1.4g" % sens[4],"%1.4g" % sens[5],"%d" % ss,"%1.4g" % bab))
	#print("{: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4}".format(*["%d" % ii,"%d" % orp,"%d" % orp-nrp,"%d" % ss,"%3.3g" % ebab/bab,"%3.3g" % sens[0],"%3.3g" % sens[1],"%3.3g" % sens[2],"%3.3g" % sens[3],"%3.3g" % sens[4],"%3.3g" % sens[5],"%3.3g" % sens[6],"%3.3g" % sens[7],"%3.3g" % true_dense]))
ff.close()

exit()
