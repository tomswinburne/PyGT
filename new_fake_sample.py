import numpy as np
from io import StringIO
import time,os, importlib
from tqdm import tqdm
np.set_printoptions(linewidth=160)
import lib.ktn_io as kio
import lib.gt_tools as gt

from lib.sampler import sampler as KTNsampler
from lib.aib_system import aib_system

from scipy.sparse import save_npz,load_npz, diags, eye, csr_matrix,bmat,find
from scipy.sparse.linalg import eigs,inv,spsolve
from scipy.sparse.csgraph import connected_components
import scipy as sp
import scipy.linalg as spla
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

data_dir = "KTN_data/LJ38/4k_proc/"

gt_check = False
generate = False
printout = True
beta = 10.0 # overwritten if generate = False
Emax = None
sys = aib_system(path=data_dir,beta=beta,generate=generate,Emax=Emax)
selA,selB = kio.load_AB(data_dir,np.ones(sys.N,bool))


sys.define_AB_regions(selA,selB)

""" true density """
true_dense = float(sys.K.nnz) / float(sys.K.shape[0]*sys.K.shape[1])

""" Initialize sampler """
sampler = KTNsampler(sys)

depth=10
limit=100
path, path_region = sys.find_path(sys.f[sys.selA].argmin(),sys.f[sys.selB].argmin(),depth=depth,limit=limit,strategy="RATE")
path_r, path_region_r = sys.find_path(sys.f[sys.selB].argmin(),sys.f[sys.selA].argmin(),depth=depth,limit=limit,strategy="RATE")


sampler.initial_sample_path_region(np.arange(sys.N)[path_region+path_region_r],ncs=100)


bab,gterr = sampler.new_true_branching_probability()
keylist = ['TotalMaxMin','TotalSparseMaxMin','ExpectMaxMin','ExpectMaxMaxMin','SingleMaxMin']

if printout:
    """ open output file """
    name = data_dir.split("/")[-1-int(data_dir[-1]=="/")]
    ff = open('output/pab_converge_%s' % name,'w')
    header = "#iteration\tnrp\tebab\t"
    for key in keylist:
        header += key+"\t"
    header += "Sparsity\tbab\n"
    ff.write(header)

nscycles = 30
ncycles = 100
npairs = 2
ssnpairs = 50
ss = 0 # >0 if we do single ended search

rK = sampler.sys.rK.copy()
sampler.sys.rK = sampler.sys.K.copy()
pp,res,ncp = sampler.sample(npairs=npairs,ss=ss) # sampling process. Returns
bab = res['ebab'].copy()
sampler.sys.rK = rK.copy()
ebab=0.0
tmm = [0.0,0.0]

for jj in range(nscycles):
    npc=0
    pbar=tqdm(total=ncycles,miniters=0,leave=True)
    ss=False
    ssc=0
    irp = sampler.remaining_pairs()
    pp,gres,npc = sampler.sample(npairs=npairs + 10*int(ss),ss=ss) # sampling process. Returns
    for ii in range(ncycles):

        orp = sampler.remaining_pairs() # DIVINE INFORMATION- number of remaining pairs

        # sampling process returns pp = number of pairs, sens = vector of various net sensitivities and found sparsity, ebab = predicted branching probability (to be compared to bab)
        nnz = sampler.sys.rK.nnz
        pp,res,_npc = sampler.sample(npairs=npairs + 10*int(ss),ss=ss) # sampling process. Returns

        npc += _npc
        if _npc!=(sampler.sys.rK.nnz-nnz)//2:
            print("npc",_npc,sampler.sys.rK.nnz-nnz)
        nrp = sampler.remaining_pairs() # DIVINE INFORMATION- number of remaining pairs
        #ss = orp==nrp
        ssc += int(ss)
        probe_compl = float(sampler.probed.sum()) / float(sampler.sys.K.shape[0]*sampler.sys.K.shape[1])

        if printout:
            ff.write("%d %d %10.10g %10.10g " % (ii,nrp,probe_compl,res['ebab']))
            for key in keylist:
                ff.write("%10.10g %10.10g " % (res[key][0],res[key][1]))
            ff.write("%10.10g %10.10g\n" % (res['Sparsity'],bab))

        pbar.update(1)
        for key in ['TotalSparseMaxMin','SingleMaxMin','ExpectMaxMaxMin']:
            gres[key][0] = min(gres[key][0],res[key][0])
            gres[key][1] = max(gres[key][1],res[key][1])
    pbar.close()
    ebab = res['ebab']

    print("\n{: <4} {: <5} ".format("%d" % ii,"%1.4g" % (ebab/bab)),end="| ")
    for key in ['TotalSparseMaxMin','SingleMaxMin','ExpectMaxMaxMin']:
        tmm[0] = (1.0-ebab)*(1.0-np.exp(-gres[key][0]/(1.0-ebab)))/bab
        tmm[1] = ebab*(np.exp(gres[key][1]/ebab)-1.0)/bab
        print("{: <5} {: <5} {: <5}".format("%1.4g" % tmm[0],"%1.4g" % tmm[1],"%1.4g" % (tmm[0]+tmm[1])),end="| ")
    print("{: <5} {: <5} | {: <5} {: <5} {: <5} | {: <5} {: <5}".format(\
    "%1.4g" % res['Sparsity'],\
    "%1.4g" % probe_compl,\
    "%1.4g" % res['MaxInRegion'][0],"%1.4g" % res['MaxInRegion'][1],"%1.4g" % res['MaxInRegion'][2],\
    "%d" % (irp-nrp),"%d" % (npc)),"\n")

if printout:
    ff.close()
