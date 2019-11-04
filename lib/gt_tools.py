import os,time
import numpy as np

from scipy.sparse import csgraph, csr_matrix, eye, save_npz, load_npz, diags

""" test for tqdm progress bars """

try:
    from tqdm import tqdm
    has_tqdm=True
except:
    has_tqdm=False



def direct_solve(B,initial_states,final_states):
    B=B.tocsc()
    basins = initial_states+final_states
    inter_region = ~basins
    pi = np.ones(initial_states.sum()) / initial_states.sum()
    NI, BI = inter_region.sum(), B[inter_region,:].tocsr()[:,inter_region]
    BAI = B[final_states,:].tocsr()[:,inter_region].transpose().dot(np.ones(final_states.sum()))
    BIB = B[inter_region,:].tocsr()[:,initial_states].dot(pi)
    BAB = (B[final_states,:].tocsr()[:,initial_states].dot(pi)).sum()
    iGI = eye(NI,format="csr") - BI
    if has_umfpack_hack:
        x,cond = spsolvecond(iGI,BIB,giveCond=True)
    else:
        x = spsolve(iGI,BIB)
        cond = 1.0

    BABI = BAI.dot(x)
    return BABI,BAB,cond

def make_fastest_path(K,i,f,depth=1):
    d,cspath = csgraph.shortest_path(csgraph=K, indices=[f,i],\
                                    directed=True, method='D', return_predecessors=True)
    path = [i]
    s = "\npath: "
    while path[-1] != f:
        s += str(path[-1])+" -> "
        path.append(cspath[0][path[-1]])
    s += str(path[-1])+"\n"
    print(s)

    N = K.shape[0]
    path_region = np.zeros(N,bool)

    # path +
    K=K.tocsc()
    for path_ind in path:
        path_region[path_ind] = True
        indscan = np.arange(N)[path_region]
    for scan in range(depth):
        for path_ind in indscan:
            for sub_path_ind in K[:,path_ind].indices:
                path_region[sub_path_ind] = True
    return path, path_region

def gt(B,sel,condThresh=1.0e8):

    Bxx = B[sel,:].transpose()[sel,:].transpose()
    Bxj = B[sel,:].transpose()[~sel,:].transpose()
    Bix = B[~sel,:].transpose()[sel,:].transpose()
    Bij = B[~sel,:].transpose()[~sel,:].transpose()


    if sel.sum()>1:
        # Get Bxx
        Bd = Bxx.diagonal()
        # Get sum Bix for i not equal x
        Bxxnd = Bxx - diags(Bd,format="csc")
        Bs = np.ravel(Bix.sum(axis=0))
        if Bs.max()<1.0e-25:
            return B,False
        Bs+=np.ravel(Bxxnd.sum(axis=0))
        Bs[Bd<0.99] = 1.0-Bd[Bd<0.99]
        iGxx = diags(Bs,format="csc") - Bxxnd
        I = eye(sel.sum(),format=iGxx.format)

        if has_umfpack_hack:
            Gxx,cond = spsolvecond(iGxx,I,giveCond=True)
        else:
            Gxx = spsolve(iGxx,I)
            cond = 1.0

        if cond>condThresh:
            return B,False
    else:
        _b_xx = Bxx.data.sum()
        if _b_xx>0.99:
            b_xx = Bix.sum()
        else:
            b_xx = 1.0-_b_xx
        Gxx = diags(np.r_[1.0/b_xx],format="csr")



    return Bij + Bix*Gxx.tocsr()*Bxj,True

def gt_seq(N,rm_reg,B,trmb=4,condThresh=1.0e10,order=None):
    rmb=trmb
    retry=0
    NI = rm_reg.sum()

    if has_tqdm:
        print("GT regularization removing %d states:" % NI)
        pbar = tqdm(total=NI-1)

    while NI>0:
        rm = np.zeros(N,bool)
        if order is None:
            rm[rm_reg.nonzero()[0][:min(rmb,NI)]] = True
        else:
            rm[order.argsort()[:min(rmb,NI)]] = True


        B, success = gt(B,rm,condThresh=condThresh)
        if success:
            N -= rm.sum()
            NI -= rm.sum()

            rm_reg = rm_reg[~rm]
            if not (order is None):
                order = order[~rm]
            if has_tqdm:
                pbar.update(rm.sum())
            rmb = trmb
        else:
            rmb = 1
            retry += 1
    #pbar.update(NI)
    return B,N,retry
