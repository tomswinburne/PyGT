import numpy as np
np.set_printoptions(linewidth=160)

from lib.ktn_io import * # tqdm / hacked scipy test
from lib.gt_tools import *

from scipy.sparse import save_npz,load_npz

print("\n\nGT REGULARIZATION TESTS\n")


# Do we generate the KTN from data, or read in the "cache"
generate = False#True

# Do we try a brute solve?
brute = True

beta = 18.0 # overwritten if generate = False
Emax = None#-167.5
beta, B, K, D, N, u, s, kt, kcon, Emin, index_sel = load_save_mat(path="KTN_data/LJ38/10k/",beta=beta,Emax=Emax,Nmax=150000,generate=generate)
f = u - s/beta
print("beta: ",beta,"N: ",N)

"""
Find fastest path from state_state to end_state, then returns all states on path and 'depth' connections away
depth=1 => path_region =  all direct connections to the path
"""
start_state = 0#f.argmin() # free energy minimum
end_state = 5#f.argmax() # free energy maximum


piM = D.copy()
piM.data = np.exp(f)
TE = K.copy().tocsr() * piM
TE.data = 1.0/TE.data

path, path_region = make_fastest_path(TE,start_state,end_state,depth=4,limit=10)
print(path_region.sum())

"""
Boolean vectors selecting A and/or B regions
"""
initial_states, final_states = np.zeros(N,bool), np.zeros(N,bool)
initial_states[path[0]] = True
final_states[path[-1]] = True
basins = initial_states + final_states
inter_region = ~basins


print("\n%d INITIAL STATES -> %d FINAL STATES\n" % (initial_states.sum(),final_states.sum()))


out = output_str()
if brute:
    """ First, try a brute solve. cond variable !=1 iff using hacked scipy """
    BABI, BAB = direct_solve(B,initial_states,final_states)
    out(["\nBRUTE SOLVE:","B(A<-B):",BABI+BAB,"B(AB):",BAB,"B(AIB):",BABI,"\n"])
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
import time

inter_region[path_region] = False
final_print = ""
for trmb in [500,40]:
    t = time.time()
    # remove trmb states at a time by GT
    rB, rD, rN, retry = gt_seq(N=N,rm_reg=inter_region,B=B,D=D.data,trmb=trmb)
    #rB, rN, retry = gt_seq(N=N,rm_reg=inter_region,B=B,D=None,trmb=trmb,condThresh=1.0e10,order=None)
    #rD=None
    #save_npz('output/rB.npz',rB)
    #np.savetxt('output/rD.txt',rD)
    #retry = 0
    #rB = load_npz('output/rB.npz')
    #rD = np.loadtxt('output/rD.txt')
    #rN = rD.size
    print("is it dense?: N=",rN,"N^2=",rN*rN,"rB.data.size=",rB.data.size,"sparsity:",float(rB.data.size)/float(rN)/float(rN))
    DD = rB.diagonal()
    r_initial_states = initial_states[~inter_region]
    r_final_states = final_states[~inter_region]
    BABI, BAB = direct_solve(rB,r_initial_states,r_final_states)
    out(["\nGT[%d] justpath:" % trmb,"B(A<-B):",BABI+BAB,"B(AB):",BAB,"B(AIB):",BABI, "RESCANS: ",retry,"max(diag(BII))):",rB.diagonal().max(),"\n"])
    basins = r_initial_states + r_final_states
    rB, rD, rN, retry = gt_seq(N=rN,rm_reg=(~basins),B=rB,D=rD,trmb=40)
    #rB, rN, retry = gt_seq(N=rN,rm_reg=(~basins),B=rB,D=rD,trmb=1,condThresh=1.0e10,order=None)
    r_initial_states = r_initial_states[basins]
    r_final_states = r_final_states[basins]
    BAB = (rB[r_final_states,:].tocsr()[:,r_initial_states]).sum()
    BABI = 0.0

    out(["\nGT[%d] complete:" % trmb,"B(A<-B):",BAB,"RESCANS: ",retry,"TIME:",time.time()-t,"\n"])


out.summary()

print("\n\n")
exit()
