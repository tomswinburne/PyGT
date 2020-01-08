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
generate = True
beta = 10.0 # overwritten if generate = False
Emax = -169.6
sys = aib_system(path=data_dir,beta=beta,generate=generate,Emax=Emax)

print("beta: ",sys.beta,"N: ",sys.N)

"""
Boolean vectors selecting A and/or B regions
"""

initial_state = 0#sys.f.argmin()
final_state = 5#sys.f.argmax()

#path, path_region = sys.find_path(initial_state,final_state,depth=2,limit=4,strategy="RATE")
path, path_region = sys.find_path(initial_state,final_state,depth=4,limit=20,strategy="DNEB")

selB = np.zeros(sys.N,bool)
selA = np.zeros(sys.N,bool)

nbasin=1


""" B = lowest free energy state and nbasin connecting states with highest rate """
selB[path[0]] = True
cB = sys.ConnectingStates(path[0]) # index, rate
for ii in np.r_[cB[1]].argsort()[::-1][:nbasin-1]: # <=nbasin connecting states with highest rate
	selB[cB[0][ii]]=True

""" A = highest free energy state and nbasin connecting states with highest rate """
selA[path[-1]] = True

cA = sys.ConnectingStates(path[-1]) # index, rate
for ii in np.r_[cA[1]].argsort()[::-1][:nbasin-1]: # <=nbasin connecting states with highest rate
	selA[cA[0][ii]]=True

""" ensure A,B are disjoint """
for ii in np.arange(sys.N)[selB]:
	selA[ii] = False

print("DISJOINT? : ",((selA*selB).sum()==0))
sys.define_AB_regions(selA,selB)


sampler = sampler(sys)
sys.define_AB_regions(selA,selB)

sampler.initial_sample_path_region(np.arange(sys.N)[path_region],ncs=300)

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
npairs = 50
ssnpairs = 50
ss = 0 # >0 if we do single eneded search

rK = sampler.sys.rK.copy()
sampler.sys.rK = sampler.sys.K
pp,sens,ebab = sampler.sample(npairs=npairs,ss=ss) # sampling process. Returns

print("SENS: ",ebab)
sampler.sys.rK = rK


print("{: <4} {: <6} {: <6} {: <6} {: <6} {: <6} {: <6}".format(
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

	"""
	0-4: ii,nrp,pc, ebab,bab
	sens[0] : <sparsity>
	sens[1-2] : sigma^1
	sens[3-5] : <c> phi/(1-phi)
	sens[6-7] : sigma
	sens[8-10] : geosigma
	"""

	probe_compl = float(sampler.probed.sum()) / float(sampler.sys.K.shape[0]*sampler.sys.K.shape[1])
	data_floats = [probe_compl,ebab,bab,sens[0],sens[1],sens[2],sens[3],sens[4],sens[5],sens[6],sens[7],sens[8],sens[9],sens[10]]
	data_string = "%d %d" % (ii, nrp)
	for df in data_floats:
		data_string+=" %10.10g" % df
	ff.write(data_string+"\n")

	#sens[:-2] += ebab
	sens[1:] /= bab
	print("{: <4} {: <6} {: <6} {: <6} {: <6}  {: <6} {: <6} {: <6}  {: <6} {: <6} {: <6}  {: <6} {: <1} {: <6}".format(
	"%d" % ii,"<C>: %1.3g" % (ebab/bab),\
	"<st>: %1.3g" % (sens[6]+sens[7]),"st+: %1.3g" % sens[6],"st-: %1.3g" % sens[7],\
	"<s1>: %1.3g" % (sens[1]+sens[2]),"s1+: %1.3g" % sens[1],"s1-: %1.3g" % sens[2],\
	"<gs>: %1.3g" % sens[10],"gs+: %1.3g" % sens[8],"gs-: %1.3g" % sens[9],\
	"spar: %1.3g" % sens[0],"%d" % ss,"%1.3g" % bab))


	if orp==nrp:
		ss = ssnpairs
	else:
		ss = 0

ff.close()

exit()
