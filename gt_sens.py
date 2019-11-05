import os,time,sys,timeit
from mat_tools import *#load_save_mat, timer, gt, direct_solve, gt_seq, make_fastest_path,

import numpy as np
from scipy.sparse import csr_matrix, eye, save_npz, load_npz, diags
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

generate = False
beta = 10.0

print("\n\nGT REGULARIZATION + SAMPLING TESTS\n")

beta, B, K, D, N, f, kt, kcon = load_save_mat(path="../../data/LJ13",beta=beta,Nmax=5000,generate=generate)



print("beta: ",beta,"N: ",N)

path, path_region = make_fastest_path(K,f.argmin(),f.argmax(),depth=1) # K, i, f

initial_states, final_states = np.zeros(N,bool), np.zeros(N,bool)
initial_states[path[0]] = True
final_states[path[-1]] = True

pi = np.ones(initial_states.sum())/initial_states.sum()
basins = initial_states + final_states
inter_region = ~basins

print("\n%d INITIAL STATES -> %d FINAL STATES\n" % (initial_states.sum(),final_states.sum()))


BABI, BAB,cond = direct_solve(B,initial_states,final_states)

out = output_str()
out(["\nBRUTE SOLVE:","B(A<-B):",BABI+BAB,"B(AB):",BAB,"B(AIB):",BABI,"COND:",cond,"\n"])
out(["\n%d PATH+ENV STATES\n" % (path_region.sum())])

inter_region[path_region] = False
ikcon = kcon.copy()
ikcon[path_region] = ikcon.max()

np.set_printoptions(linewidth=160)

final_print = ""
for trmb in [40,10,1]:
    # remove trmb states at a time by GT

    rB, rN, retry = gt_seq(N=N,rm_reg=inter_region,B=B,trmb=trmb,condThresh=1.0e10,order=ikcon)
    DD = rB.diagonal()

    r_initial_states = initial_states[~inter_region]
    r_final_states = final_states[~inter_region]
    BABI, BAB, cond = direct_solve(rB,r_initial_states,r_final_states)



    out(["\nGT[%d] justpath:" % trmb,"B(A<-B):",BABI+BAB,"B(AB):",BAB,"B(AIB):",BABI, "RESCANS: ",retry,"COND:",cond,"max(diag(BII))):",rB.diagonal().max(),"\n"])

    basins = r_initial_states + r_final_states
    rB, rN, retry = gt_seq(N=rN,rm_reg=(~basins),B=rB,trmb=1,condThresh=1.0e10,order=None)
    r_initial_states = r_initial_states[basins]
    r_final_states = r_final_states[basins]
    BAB = (rB[r_final_states,:].tocsr()[:,r_initial_states]).sum()
    BABI = 0.0

    out(["\nGT[%d] complete:" % trmb,"B(A<-B):",BAB,"RESCANS: ",retry,"\n"])


out.summary()

print("\n\n")
exit()
