import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"./lib")
from sampler import sampler
from aib_system import AIB_system


sys = AIB_system(path="KTN_data/LJ13/",beta=8.0,Nmax=5000,generate=True)

print(sys.K.shape)

Ecut=8.5
# B -> A
selA = np.zeros(sys.N,bool)
selB = np.zeros(sys.N,bool)

selB[sys.pi.argmax()] = True
cB = sys.ConnectingStates(sys.pi.argmax()) # index, rate
for ii in np.r_[cB[1]].argsort()[::-1][:5]: # <=10 connecting states with highest rate
	selB[cB[0][ii]]=True

selA[sys.pi.argmin()] = True
cA = sys.ConnectingStates(sys.pi.argmin()) # index, rate
for ii in np.r_[cA[1]].argsort()[::-1][:5]: # <=10 connecting states with highest rate
	selA[cA[0][ii]]=True
for ii in np.arange(sys.N)[selB]:
	selA[ii] = False

print("DISJOINT? : ",((selA*selB).sum()==0))


true_dense = float(sys.K.nnz) / float(sys.K.shape[0]*sys.K.shape[1])

sys.define_AB_regions(selA,selB)


sampler = sampler(sys)

bab = sampler.true_branching_probability(gt_check=False)

# find some paths...
sampler.initial_sample()


ss = False




print("HERE")

ff = open('output/pab_converge','w')
npairs = 1
ssnpairs = 2
ss = 0

for ii in range(2000):
	orp = sampler.remaining_pairs()

	pp,sens,ebab = sampler.sample(npairs=npairs,ss=ss,ignore_distance=True)
	nrp = sampler.remaining_pairs()
	probe_compl = float(sampler.probed.sum()) / float(sampler.sys.K.shape[0]*sampler.sys.K.shape[1])
	ff.write("%d %d %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %10.10g %d %10.10g\n" %\
	 	(ii,nrp,probe_compl,ebab,sens[0],sens[1],sens[2],sens[3],sens[4],sens[5],sens[6],sens[7],int(sens[8]),bab))
	sens[:-2] += ebab
	sens[:-2] /= bab
	if orp==nrp:
		ss = ssnpairs
	else:
		ss = 0
	#print('{:04d :04d :04d :04d :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f :02.4f}'.format(ii,orp,orp-nrp,ss,ebab/bab,sens[0],sens[1],sens[2],sens[3],sens[4],sens[5],sens[6],sens[7],true_dense))
	print("{: <4} {: <10} {: <10} {: <10} {: <10}".format("%d" % ii,"%1.4g" % (ebab/bab),"%1.4g" % sens[6],"%1.4g" % sens[4],"%1.4g" % sens[5]))
	#print("{: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4} {: <4}".format(*["%d" % ii,"%d" % orp,"%d" % orp-nrp,"%d" % ss,"%3.3g" % ebab/bab,"%3.3g" % sens[0],"%3.3g" % sens[1],"%3.3g" % sens[2],"%3.3g" % sens[3],"%3.3g" % sens[4],"%3.3g" % sens[5],"%3.3g" % sens[6],"%3.3g" % sens[7],"%3.3g" % true_dense]))
ff.close()
