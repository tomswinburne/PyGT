import numpy as np
import time
np.set_printoptions(linewidth=160)
from lib.ktn_io import *
from lib.gt_tools import *
from scipy.sparse import save_npz,load_npz



print("\n\nGT REGULARIZATION TESTS\n")

# Do we generate the KTN from data, or read in the "cache"
generate = False

# Do we try a brute solve?
brute = False



beta = 10.0 # overwritten if generate = False
Emax = None#-167.5
beta, B, K, D, N, u, s, kt, kcon, Emin = load_save_mat(path="KTN_data/LJ38/",beta=beta,Emax=Emax,Nmax=150000,generate=generate)
f = u - s/beta
print("beta: ",beta,"N: ",N)

print(B.sum(axis=0).min())



piM = D.copy()
piM.data = np.exp(f)
TE = K.copy().tocsr() * piM
TE.data = 1.0/TE.data

"""
Boolean vectors selecting A and/or B regions
"""
initial_states, final_states = np.zeros(N,bool), np.zeros(N,bool)
initial_states[np.loadtxt('min_oct').astype(int)-1] = True
final_states[np.loadtxt('min_ico').astype(int)-1] = True
basins = initial_states + final_states
inter_region = ~basins

print("\n%d INITIAL STATES -> %d FINAL STATES\n" % (initial_states.sum(),final_states.sum()))

""" build path region """

"""
path_region = np.zeros(N,bool)
for start_state in np.arange(N)[initial_states]:
    for end_state in np.random.choice(np.arange(N)[final_states],size=5,replace=False):
    _path, _path_region = make_fastest_path(TE,start_state,end_state,depth=1,limit=5)
    path_region += _path_region
inter_region[path_region] = False
"""

#path, path_region = make_fastest_path(TE,0,6,depth=1,limit=5)
#inter_region[path_region] = False

out = output_str()
out(["\n%d PATH+ENV STATES\n" % ((~inter_region).sum())])


if brute:
    """ First, try a brute solve. cond variable !=1 iff using hacked scipy """
    BABI, BAB,cond = direct_solve(B,initial_states,final_states)
    out(["\nBRUTE SOLVE:","B(A<-B):",BABI+BAB,"B(AB):",BAB,"B(AIB):",BABI,"COND:",cond,"\n"])

"""
Test of numerical stability of various extended GT removal protocols
remove all states with inter_region == True
Try exact same protocol with varying block size
blocks of 1 is exactly the normal GT process
"""


final_print = ""
for trmb in [100,40]:
    t = time.time()
    # first, large GT


    rB, rD, rN, retry = gt_seq(N=N,rm_reg=inter_region,B=B,D=D.data,trmb=trmb)

    r_initial_states = initial_states[~inter_region]
    r_final_states = final_states[~inter_region]

    print("is it dense?: N=",rN,"N^2=",rN*rN,"rB.data.size=",rB.data.size,"sparsity:",float(rB.data.size)/float(rN)/float(rN))


    # direct solve on retained states
    BABI, BAB, cond = direct_solve(rB,r_initial_states,r_final_states)
    out(["\nGT[%d] justpath:" % trmb,"B(A<-B):",BABI+BAB,"B(AB):",BAB,"B(AIB):",BABI, "RESCANS: ",retry,"TIME:",time.time()-t,"max(diag(BII))):",rB.diagonal().max(),"\n"])

    basins = r_initial_states + r_final_states
    rm_reg= ~basins
    if rm_reg.sum()>0:
        rB, rD, rN, retry = gt_seq(N=rN,rm_reg=(~basins),B=rB,D=rD,trmb=1)
        r_initial_states = r_initial_states[basins]
        r_final_states = r_final_states[basins]

    oneB = np.ones(r_initial_states.sum())
    BABM = rB[r_final_states,:].transpose()[r_initial_states,:].transpose()
    BAB = BABM.dot(oneB).sum()
    BABI = 0.0

    out(["\nGT[%d] complete:" % trmb,"B(A<-B):",BAB,"RESCANS: ",retry,"TIME:",time.time()-t,"\n"])


out.summary()

print("\n\n")
exit()
