import numpy as np
import time
np.set_printoptions(linewidth=160)
from lib.ktn_io import *
from lib.gt_tools import *
from scipy.sparse import save_npz,load_npz
from scipy.sparse.linalg import eigs,inv



print("\n\nGT REGULARIZATION TESTS\n")

# Do we generate the KTN from data, or read in the "cache"
generate = 0

# Do we try a brute solve?
brute = True

beta = 10.0 # overwritten if generate = False
betar = [5.0] # range(1,16)
for _beta in betar:

    Emax = None#-167.5
    beta, B, K, D, N, u, s, kt, kcon, Emin = load_save_mat(path="KTN_data/LJ38/",beta=_beta,Emax=Emax,Nmax=5000,generate=generate)
    f = u - s/beta

print("beta: ",beta,"N: ",N)

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

out = output_str()
out(["\n%d PATH+ENV STATES\n" % ((~inter_region).sum())])


if brute:
    """ First, try a brute solve. cond variable !=1 iff using hacked scipy """
    aK = (D-K)[~final_states,:][:,~final_states]
    #print(aK.sum(axis=1).min())
    evals_small, evecs_small = eigs(aK, 1, sigma=0, which='LM')
    print(evals_small.real)
    exit()

    BABI, BAB = direct_solve(B,initial_states,final_states)
    out(["\nBRUTE SOLVE:","B(A<-B):",BABI+BAB,"B(AB):",BAB,"B(AIB):",BABI,"\n"]) #,"COND:",cond,

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
    BABI, BAB = direct_solve(rB,r_initial_states,r_final_states)
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
