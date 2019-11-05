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
beta = 20.0 # overwritten if generate = False

sys = aib_system(path=data_dir,beta=beta,Nmax=15000,generate=generate)

print("beta: ",sys.beta,"N: ",sys.N)

"""
Find fastest path from state_state to end_state, then returns all states on path and 'depth' connections away
depth=1 => path_region =  all direct connections to the path
"""
path, path_region = make_fastest_path(sys.K,sys.f.argmin(),sys.f.argmax(),depth=1,limit=2)

"""
Boolean vectors selecting A and/or B regions
"""
selB = np.zeros(sys.N,bool)
selA = np.zeros(sys.N,bool)
selB[path[0]] = True
selA[path[-1]] = True
sys.define_AB_regions(selA,selB)

sampler = sampler(sys)
sampler.initial_sample_path_region(np.arange(sys.N)[path_region])


""" get "exact" answer by linear algebra and GT. They need to match to trust the linear algebra! """
if gt_check:
	bab,gtdiff = sampler.true_branching_probability(gt_check=True)
	print("GT/MATRIX DIFFERENCE: %2.4g%%" % (100.0*gtdiff))

	if gtdiff>0.001: # i.e. 0.1%
		print("MATRIX METHOD UNSTABLE! NEED TO PRE-GT BEFORE SENSITIVITY!")
		""" Need to implement preGT in this repo """
		exit()
else:
	bab = sampler.true_branching_probability(gt_check=False)


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
ncycles = 300
npairs = 1
ssnpairs = 1
ss = 0 # >0 if we do single eneded search

for ii in range(ncycles):

	orp = sampler.remaining_pairs() # DIVINE INFORMATION- number of remaining pairs

	""" sampling process. returns
	pp = number of pairs
	sens = vector of various net sensitivities and found sparsity
	ebab = predicted branching probability (to be compared to bab)
	"""
	pp,sens,ebab = sampler.sample(npairs=npairs,ss=ss,ignore_distance=True) # sampling process. Returns

	nrp = sampler.remaining_pairs() # DIVINE INFORMATION- number of remaining pairs

	probe_compl = float(sampler.probed.sum()) / float(sampler.sys.K.shape[0]*sampler.sys.K.shape[1])

	ff.write("%d %d %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %d %10.10g\n" %\
	 	(ii,nrp,probe_compl,ebab,sens[0],sens[1],sens[2],sens[3],sens[4],sens[5],sens[6],sens[7],int(sens[8]),bab))

	sens[:-2] += ebab
	sens[:-2] /= bab
	"""
    if orp==nrp:
		ss = ssnpairs
	else:
		ss = 0
    """
	#print('{:04d :04d :04d :04d :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f}'.format(ii,orp,orp-nrp,ss,ebab/bab,sens[0],sens[1],sens[2],sens[3],sens[4],sens[5],sens[6],sens[7],true_dense))
	print("{: <4} {: <10} {: <10} {: <10} {: <10} {: <10}".format(
	"%d" % ii,"%1.4g" % (ebab/bab),"%1.4g" % sens[6],"%1.4g" % sens[4],"%1.4g" % sens[5],"%d" % ss))
	#print("{: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4}".format(*["%d" % ii,"%d" % orp,"%d" % orp-nrp,"%d" % ss,"%3.3g" % ebab/bab,"%3.3g" % sens[0],"%3.3g" % sens[1],"%3.3g" % sens[2],"%3.3g" % sens[3],"%3.3g" % sens[4],"%3.3g" % sens[5],"%3.3g" % sens[6],"%3.3g" % sens[7],"%3.3g" % true_dense]))
ff.close()

exit()
