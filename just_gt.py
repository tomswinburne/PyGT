import os,time,sys,timeit
os.system('mkdir -p cache')
os.path.insert(0,'./lib')

from ktn_io import * # tqdm / hacked scipy test

# load_save_mat, timer, gt, direct_solve, gt_seq, make_fastest_path,
import numpy as np
from scipy.sparse import csr_matrix, eye, save_npz, load_npz, diags
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
np.set_printoptions(linewidth=160)


print("\n\nGT REGULARIZATION TESTS\n")


generate = True # Do we generate the KTN from data, or read in the "cache"

beta = 8.0 # overwritten if generate = False

beta, B, K, D, N, f, kt, kcon = load_save_mat(path="../../data/Au/au55_10000/",beta=beta,Nmax=15000,generate=generate)

print("beta: ",beta,"N: ",N)

"""
Find fastest path from state_state to end_state, then returns all states on path and 'depth' connections away
depth=1 => path_region =  all direct connections to the path
"""
start_state = f.argmin() # free energy minimum
end_state = f.argmax() # free energy maximum
path, path_region = make_fastest_path(K,start_state,end_state,depth=1)


"""
Boolean vectors selecting A and/or B regions
"""
initial_states, final_states = np.zeros(N,bool), np.zeros(N,bool)
initial_states[path[0]] = True
final_states[path[-1]] = True
basins = initial_states + final_states
inter_region = ~basins


print("\n%d INITIAL STATES -> %d FINAL STATES\n" % (initial_states.sum(),final_states.sum()))


""" First, try a brute solve """
BABI, BAB,cond = direct_solve(B,initial_states,final_states)

out = output_str()
out(["\nBRUTE SOLVE:","B(A<-B):",BABI+BAB,"B(AB):",BAB,"B(AIB):",BABI,"COND:",cond,"\n"])
out(["\n%d PATH+ENV STATES\n" % (path_region.sum())])



""" ikcon estimates the local condition number and orders the GT removal process accordingly """
ikcon = kcon.copy()
ikcon[path_region] = ikcon.max()


"""
Test of numerical stability of various extended GT removal protocols

remove all states with inter_region == True

Try exact same protocol with in blocks of 40,10 or 1

blocks of 1 is exactly the normal GT process
"""

inter_region[path_region] = False
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
