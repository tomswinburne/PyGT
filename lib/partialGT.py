""" Script to perform partial GT analysis on networks.

Deepti Kannan, 2020"""

import numpy as np
from io import StringIO
import time,os, importlib
from tqdm import tqdm
np.set_printoptions(linewidth=160)
import lib.ktn_io as kio
import lib.gt_tools as gt
from scipy.sparse import save_npz,load_npz, diags, eye, csr_matrix,bmat
from scipy.sparse.linalg import eigs,inv,spsolve
from scipy.sparse.csgraph import connected_components
import scipy as sp
import scipy.linalg as spla

def read_communities(commdat, index_sel, screen=False):
    """Read in a single column file called communities.dat where each line
    is the community ID (zero-indexed) of the minima given by the line
    number. Produces selectors akin to AS and BS in Tom's code, but
    for all N communities.

    Parameters
    ----------
    commdat : .dat file
        single-column file containing community IDs of each minimum
    index_sel : (N,) boolean array
        selects out the largest connected component of the network

    Returns
    -------
    communities : dict
        mapping from community ID (0-indexed) to a boolean array
        of shape (len(index_sel), ) which selects out the states in that community.
    """

    communities = {}
    with open(commdat, 'r') as f:
        for minID, line in enumerate(f, 0):
            groupID =  int(line) #number from 0 to N-1
            if groupID in communities:
                communities[groupID].append(minID)
            else:
                communities[groupID] = [minID]
    
    for ci in range(len(communities)):
        #create a new index_selector to select out the minima in community ci
        keep = np.zeros(index_sel.size,bool)
        keep[communities[ci]] = True
        #re-assign communities[ci] to be the index-selector for the maximally connected component of the graph
        communities[ci] = keep[index_sel]
        if screen:
            print(f'Community {ci}: {keep.sum()}')
        
    return communities


def compute_passage_stats(AS, BS, BF, Q, dopdf=True):
    """Compute the A->B and B->A first passage time distribution, 
    first moment, and second moment.

    Parameters
    ----------
    AS : (N,) array-like
        boolean array that selects out the A nodes
    BS : (N,) array-like
        boolean array that selects out the B nodes
    BF : (N,) array-like
        array of free energies of all nodes
    Q : (N, N) array-like
        rate matrix
    dopdf : bool
        whether to calculate full fpt distribution, defaults to True

    Returns
    -------
    tau : (4,) array-like
        <tau>_BA, <tau^2>_BA, <tau>_AB, <tau^2>_AB
    pt : (4, 400) array-like
        time in multiples of <tau> and p(t)*<tau> for A->B and B->A
    """

    #<tauBA>, <tau^2BA>, <tauAB>, <tau^2AB>
    tau = np.zeros(4)
    if dopdf:
        # time*tau_range, p(t) (first 2: A->B, second 2: B->A)
        pt = np.zeros((4,400))
        
    #A -> B
    #P(0) is initialized to local boltzman of source community A
    rho = np.exp(-BF) * AS
    rho /= rho.sum()
    #B is absorbing, so we want Q in space of A U I
    M = Q[~BS,:][:,~BS]
    x = spsolve(M,rho[~BS])
    y = spsolve(M,x)
    # first moment tau(A->B) = 1.Q^{-1}.rho(A) = 1.x
    tau[0] = x.sum() 
    # second moment = 2 x 1.Q^{-2}.rho = 2.0* 1.Q^{-1}.x
    tau[1] = 2.0*y.sum() 
    if dopdf:
        #time in multiples of the mean first passage time
        pt[0] = np.logspace(-6,3,pt.shape[1])*tau[0]
        #nu=eigenvalues, v=left eigenvectors, w=right eigenvectors
        nu,v,w = spla.eig(M.todense(),left=True)
        #normalization factor
        dp = np.sqrt(np.diagonal(w.T.dot(v))).real
        #dot product (v.P(0)=rho)
        v = (v.real.dot(np.diag(1.0/dp))).T.dot(rho[~BS])
        #dot product (1.T.w)
        w = (w.real.dot(np.diag(1.0/dp))).sum(axis=0)
        nu = nu.real
        #(v*w/nu).sum() is the same as <tau>, the first bit is the pdf p(t)
        pt[1] = (v*w*nu)@np.exp(-np.outer(nu,pt[0]))*(v*w/nu).sum()
    
    #B -> A
    rho = np.exp(-BF) * BS
    rho /= rho.sum()
    M = Q[~AS,:][:,~AS]
    x = spsolve(M,rho[~AS])
    y = spsolve(M,x)
    tau[2] = x.sum()
    tau[3] = 2.0*y.sum()
    if dopdf:
        pt[2] = np.logspace(-6,3,pt.shape[1])*tau[2]
        nu,v,w = spla.eig(M.todense(),left=True)
        dp = np.sqrt(np.diagonal(w.T.dot(v))).real
        v = (v.real.dot(np.diag(1.0/dp))).T.dot(rho[~AS])
        w = (w.real.dot(np.diag(1.0/dp))).sum(axis=0)
        nu = nu.real
        pt[3] = (v*w*nu)@np.exp(-np.outer(nu,pt[2]))*(v*w/nu).sum()
        return tau, pt
    else:
        return tau

def compute_escape_stats(BS, BF, Q, tau_escape=None, dopdf=True):
    """Compute escape time distribution and first and second moment
    from the basin specified by BS.

    Parameters
    ----------
    BS : (N,) array-like
        boolean array that selects out the nodes in the active basin
    BF : (N,) array-like
        array of free energies of all nodes
    Q : (N, N) array-like
        rate matrix
    tau_escape : float
        mean time to escape from B. Used to calculate the escape
        time distribution in multiple of tau_escape (p(t/tau_escape).
        If None, uses the first moment in network defined by Q.
    dopdf : bool
        whether to calculate full escapte time distribution, defaults to True

    Returns
    -------
    tau : (2,) array-like
        <tau>_B, <tau^2>_B
    pt : (2, 400) array-like
        time in multiples of <tau_escape> and p(t)*<tau_escape> 

    """
    #<tau>, <tau^2>
    tau = np.zeros(2)
    if dopdf:
        # time*tau_range, p(t)
        pt = np.zeros((2, 400))
    rho = np.exp(-BF) * BS
    rho /= rho.sum()
    M = Q[BS,:][:,BS]
    x = spsolve(M,rho[BS])
    y = spsolve(M,x)
    tau[0] = x.sum()
    tau[1] = 2.0*y.sum()
    if tau_escape is None:
        tau_escape = tau[0]
    if dopdf:
        pt[0] = np.logspace(-6,3,pt.shape[1])*tau_escape
        nu,v,w = spla.eig(M.todense(),left=True)
        dp = np.sqrt(np.diagonal(w.T.dot(v))).real
        v = (v.real.dot(np.diag(1.0/dp))).T.dot(rho[BS])
        w = (w.real.dot(np.diag(1.0/dp))).sum(axis=0)
        nu = nu.real
        pt[1] = (v*w*nu)@np.exp(-np.outer(nu,pt[0]))*tau_escape
        return tau, pt
    else:
        return tau

def prune_intermediate_nodes(beta, data_path, rm_type='hybrid', 
                             percent_retained=10, dopdf=True, screen=True):
    """ Prune nodes only in the intermediate region using the heuristic 
    specified in `rm_type`. 

    Parameters
    ----------
    beta : float
        inverse temperature
    data_path : str or Path
        location of min.data, ts.data, min.A, min.B files
    rm_type : str
        heuristic used to remove nodes, 'escape_time', 'free_energy', or 'node_degree'.
        Defaults to 'hybrid'.
    percent_retained : float
        percent of intermediate nodes to keep in reduced network, defaults to 10
    dopdf : bool
        whether to calculate full fpt distribution, defaults to True
    screen : bool
        whether to print nodes that are being eliminated, defaults to True

    Returns
    -------
    tau : (4,) array-like
        <tau>_BA, <tau^2>_BA, <tau>_AB, <tau^2>_AB
    pt : (4, 400) array-like
        time in multiples of <tau> and p(t)*<tau> for A->B and B->A
    gttau : (4,) array-like
        same quantities as tau but in reduced network
    gtpt : (4, 400) array-like
        same quantities as pt but in reduced network 

    """
    B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=data_path,beta=beta,Emax=None,Nmax=None,screen=False)
    node_degree = B.indptr[1:] - B.indptr[:-1]
    D = np.ravel(K.sum(axis=0))
    Q = diags(D)-K
    escape_time = 1./D
    BF = beta*u-s
    BF -= BF.min()
    AS,BS = kio.load_AB(data_path,index_sel)    
    IS = np.zeros(N, bool)
    IS[~(AS+BS)] = True
    if screen:
        print(f'A: {AS.sum()}, B: {BS.sum()}, I: {IS.sum()}')
    
    #First calculate p(t), <tau>, <tau^2> without any GT      
    tau, pt = compute_passage_stats(AS, BS, BF, Q)
        
    """Now calculate <tau>, <tau^2>, p(t) after graph transformation"""
    rm_reg = np.zeros(N, bool)
    
    if rm_type == 'node_degree':
        rm_reg[node_degree < 2] = True
        rm_reg[(AS+BS)] = False #only remove the intermediate nodes
    elif rm_type == 'escape_time':
        #remove nodes with the smallest escape times
        #retain nodes in the top percent_retained percentile of escape time
        rm_reg[IS] = escape_time[IS] < np.percentile(escape_time[IS], 100.0 - percent_retained)
    elif rm_type == 'free_energy':
        #retain nodes in the bottom percent_retained percentile of free energy
        rm_reg[IS] = BF[IS] > np.percentile(BF[IS], percent_retained)   
    elif rm_type == 'hybrid':
        #remove nodes in the top percent_retained percentile of escape time
        time_sel = (escape_time[IS] < np.percentile(escape_time[IS], 100.0 - percent_retained))
        bf_sel = (BF[IS]>np.percentile(BF[IS],percent_retained))
        sel = np.bitwise_and(time_sel, bf_sel)
        #that are also in the lowest percent_retained percentile of free energy
        rm_reg[IS] = sel
    elif rm_type == 'combined':
        #multiple escape_time by occupation probability e^-BF
        #high free energy nodes have a small occupation probability 
        #we want to remove nodes with low occupation probability AND small escape time
        #if we multiply together, should make sense to remove the
        rho = np.exp(-BF) #stationary probabilities
        rho /= rho.sum()
        combo_metric = escape_time * rho
        rm_reg[IS] = combo_metric[IS] < np.percentile(combo_metric[IS], 100.0 - percent_retained)
    else:
        print('Choose a valid GT removal strategy.')
        return   
    #free energies of retained states
    r_BF = BF[~rm_reg]
    if screen:
        print(f'Nodes to eliminate: {rm_reg.sum()}/{N}')
        print(f'in A: {rm_reg[AS].sum()}, in B: {rm_reg[BS].sum()}, in I: {rm_reg[IS].sum()}')
    #perform the graph transformation
    GT_B, GT_D, GT_Q, r_N, retry  = gt.gt_seq(N=N,rm_reg=rm_reg,B=B,D=D,trmb=10,retK=True,Ndense=50,screen=False)
    #compute stats on reduced network
    gttau, gtpt = compute_passage_stats(AS[~rm_reg], BS[~rm_reg], r_BF, GT_Q)
    
    if dopdf:
        return beta, tau, gttau, pt, gtpt
    else:
        return beta, tau, gttau

def prune_source(beta, data_path, rm_type='hybrid', percent_retained_in_B=90.,
                 dopdf=True, BS=None):
    """ Prune nodes in the source community and compute escape time distribution
    using the heuristic specified by `rm_type`.

    Parameters
    ----------
    beta : float
        inverse temperature
    data_path : str or Path
        location of min.data, ts.data, min.A, min.B files
    rm_type : str
        heuristic used to remove nodes, 'escape_time', 'free_energy', or 'node_degree'
    percent_retained_in_B : float
        percent of nodes to keep in the source community B
    dopdf : boolean
        whether to calculate the full escape time distribution, defaults to True.
    BS : (N,) boolean array
        array that selects out the states in the source community. If none, reads in the
        minima in min.B as the source region.


    Returns
    -------
    tau : (2,) array-like
        <tau>_B, <tau^2>_B
    pt : (2, 400) array-like
        time in multiples of <tau> and p(t)*<tau>
    gttau : (2,) array-like
        same quantities as tau but in reduced network
    gtpt : (2, 400) array-like
        same quantities as pt but in reduced network 

    """
    Nmax = None
    B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=data_path,beta=beta,Emax=None,Nmax=Nmax,screen=False)
    D = np.ravel(K.sum(axis=0))
    escape_time = 1./D
    Q = diags(D)-K
    BF = beta*u-s
    BF -= BF.min()
    if BS is None:
        AS, BS = kio.load_AB(data_path,index_sel)
        IS = np.zeros(N, bool)
        IS[~(AS+BS)] = True
        print(f'A: {AS.sum()}, B: {BS.sum()}, I: {IS.sum()}')

    """ First calculate p(t), <tau>, <tau^2> without any GT"""
    tau, pt = compute_escape_stats(BS, BF, Q)
        
    """Now calculate <tau>, <tau^2>, p(t) after graph transforming away the top 10% of B nodes"""
    rm_reg = np.zeros(N,bool)
    #rm_reg = ~(AS+BS)
    if rm_type == 'free_energy':
        rm_reg[BS] = BF[BS]>np.percentile(BF[BS],percent_retained_in_B)
    elif rm_type == 'escape_time':
        #remove nodes with the smallest escape times
        #retain nodes in the top percent_retained percentile of escape time
        rm_reg[BS] = escape_time[BS] < np.percentile(escape_time[BS], 100.0 - percent_retained_in_B)
    elif rm_type == 'hybrid':
        #remove nodes in the top percent_retained percentile of escape time
        time_sel = (escape_time[BS] < np.percentile(escape_time[BS], 100.0 - percent_retained_in_B))
        bf_sel = (BF[BS]>np.percentile(BF[BS],percent_retained_in_B))
        sel = np.bitwise_and(time_sel, bf_sel)
        #that are also in the lowest percent_retained percentile of free energy
        rm_reg[BS] = sel  
    elif rm_type == 'combined':
        #multiple escape_time by occupation probability e^-BF
        #high free energy nodes have a small occupation probability 
        #we want to remove nodes with low occupation probability AND small escape time
        #if we multiply together, should make sense to remove the
        rho = np.exp(-BF)[BS] #local equilibrium in source basin
        rho /= rho.sum()
        combo_metric = escape_time[BS] * rho
        rm_reg[BS] = combo_metric < np.percentile(combo_metric, 100.0 - percent_retained_in_B)
    else:
        print('Choose a valid GT removal strategy.')
        return
    #free energies of retained states
    r_BF = BF[~rm_reg]
    print(f'Nodes to eliminate: {rm_reg.sum()}')
    GT_B, GT_D, GT_Q, r_N, retry = gt.gt_seq(N=N,rm_reg=rm_reg,B=B,D=D,trmb=10,retK=True,Ndense=50,screen=False)
    gttau, gtpt = compute_escape_stats(BS[~rm_reg], r_BF, GT_Q,
                                       tau_escape=tau[0], dopdf=True)

    if dopdf:
        return beta, tau, gttau, pt, gtpt
    else:
        return beta, tau, gttau

def prune_all_basins(beta, data_path, rm_type='hybrid', percent_retained=50., screen=True):
    """ Prune each basin and compare fully reduced network to original network.

    TODO: allow percent_retained to be a vector so you can remove a different
    percent of nodes in each basin.

    Parameters
    ----------
    beta : float
        inverse temperature
    data_path : str or Path
        location of min.data, ts.data, min.A, min.B files
    rm_type : str
        heuristic used to remove nodes, 'escape_time', 'free_energy', or 'node_degree'
    percent_retained : float
        percent of nodes to keep in the source community B
    screen : boolean
        whether to print number of eliminated nodes in each basin


    Returns
    -------
    r_B : (r_N, r_N) np.ndarray[float64]
        branching probability matrix in reduced network
    r_D : (r_N, r_N) np.ndarray[float64]
        diagonal matrix with inverse escape times in reduced network
    r_Q : (r_N, r_N) np.ndarray[float64]
        rate matrix in reduced network
    r_N : int
        number of nodes in reduced network
    r_BF : (r_N,) np.ndarray[float64]
        free energies of nodes in reduced network (not shifted from original)
    r_communities : dict
        mapping from community IDs (0-indexed) to index selectors in reduced network
    """

    B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=data_path,beta=beta,Emax=None,Nmax=None,screen=False)
    D = np.ravel(K.sum(axis=0))
    Q = diags(D)-K
    escape_time = 1./D
    BF = beta*u-s
    BF -= BF.min()
    communities = read_communities(data_path/'communities.dat', index_sel)
    #nodes to remove (top 100-percent_retained percent of nodes in each basin)
    rm_reg = np.zeros(N,bool)
    
    #loop through the basins and update rm_reg[BS] with nodes to remove from each
    for source_commID in communities:
        #BS is the selector for all nodes in the source community
        BS = communities[source_commID]
        if screen:
            print(f'Source comm: {source_commID}, Source nodes: {BS.sum()}')
        if rm_type == 'free_energy':
            rm_reg[BS] = BF[BS] > np.percentile(BF[BS], percent_retained)
        elif rm_type == 'escape_time':
            #remove nodes with the smallest escape times
            #retain nodes in the top percent_retained percentile of escape time
            rm_reg[BS] = escape_time[BS] < np.percentile(escape_time[BS], 100.0 - percent_retained)
        elif rm_type == 'hybrid':
            #remove nodes in the top percent_retained percentile of escape time
            time_sel = (escape_time[BS] < np.percentile(escape_time[BS], 100.0 - percent_retained))
            bf_sel = (BF[BS]>np.percentile(BF[BS],percent_retained))
            sel = np.bitwise_and(time_sel, bf_sel)
            #that are also in the lowest percent_retained percentile of free energy
            rm_reg[BS] = sel
        elif rm_type == 'combined':
            #multiple escape_time by occupation probability e^-BF
            #high free energy nodes have a small occupation probability 
            #we want to remove nodes with low occupation probability AND small escape time
            #if we multiply together, should make sense to remove the
            rho = np.exp(-BF)[BS] #local equilibrium in source basin
            rho /= rho.sum()
            combo_metric = escape_time[BS] * rho
            rm_reg[BS] = combo_metric < np.percentile(combo_metric, 100.0 - percent_retained)
        else:
            print('Choose a valid GT removal strategy.')
            return
        if screen:
            print(f'Percent eliminated from basin: {100*rm_reg[BS].sum()/BS.sum()}')        
    #now do the GT all in one go
    r_B, r_D, r_Q, r_N, retry = gt.gt_seq(N=N,rm_reg=rm_reg,B=B,D=D,trmb=10,retK=True,Ndense=50,screen=False)
    #free energies and escape times of retained states
    r_BF = BF[~rm_reg]
    print(f'Removed {rm_reg.sum()} of {N} nodes, retained {100*r_N/N} percent')
    #community selectors in reduced network -- update shapes
    r_communities = {}
    for com in communities:
        r_communities[com] = communities[com][~rm_reg]
            
    return r_B, r_D, r_Q, r_N, r_BF, r_communities

def compute_rates(AS, BS, BF, B, D, K, **kwargs):
    """ Calculate kSS, kNSS, kF, k*, kQSD, MFPT, and committors for the transition path
    ensemble AS --> BS from rate matrix K. K can be the matrix of an original network,
    or a partially graph-transformed matrix. 
    
    Differs from compute_passage_stats in that this function removes all intervening states
    using GT before computing fpt stats and rates on the fully reduced network
    with state space (A U B)."""

    N = len(AS)
    assert(N==len(BS))
    assert(N==len(BF))

    D = np.ravel(K.sum(axis=0))
    inter_region = ~(AS+BS)
    r_AS = AS[~inter_region]
    r_BS = BS[~inter_region]
    r_BF = BF[~inter_region]
    rDSS = D[~inter_region]

    #use GT to renormalize away all I states
    rB, rD, rQ, rN, retry = gt_seq(N=N,rm_reg=inter_region,B=B,D=D,retK=True,trmb=1,**kwargs)

    #first do A->B direction, then B->A
    #r_s is the non-absorbing region (A first, then B)
    df = pd.DataFrame(columns=['MFPTAB', 'kSSAB', 'kNSSAB', 'kQSDAB', 'k*AB', 'kFAB',
                               'MFPTBA', 'kSSBA', 'kNSSBA', 'kQSDBA', 'k*BA', 'kFBA'])
    dirs = ['AB'. 'BA']
    for i, r_s in enumerate([r_AS, r_BS]) :
        #eigendecomposition of rate matrix in non-abosrbing region
        #for A, full_RK[r_A, :][:, r_A] is just a 5x5 matrix
        l, v = spla.eig(rQ[r_s,:][:,r_s].todense())
        #order the eigenvalues from smallest to largest -- they are positive since Q = D - K instead of K-D
        qsdo = np.abs(l.real).argsort()
        nu = l.real[qsdo]
        #local equilibrium distribution in r_s
        rho = np.exp(-r_BF[r_s])
        rho /= rho.sum()
        #v[:, qsdo[0]] is the eigenvector corresponding to smallest eigenvalue
        #aka quasi=stationary distribution
        qsd = v[:,qsdo[0]]
        qsd /= qsd.sum()
    
        #committor C^B_A: probability of reaching B before A: 1_B.B_BA^I (eqn 6 of SwinburneW20)
        C = np.ravel(rB[~r_s,:][:,r_s].sum(axis=0))
        #MFPT
        invQ = spla.inv(rQ[r_s,:][:,r_s].todense())
        tau = invQ.dot(rho).sum(axis=0)
        #vector of T_Ba 's : in theory, could do another 5 GT's isolating each a in A
        #so that T_Ba = tau_a / P_Ba
        T_Ba = invQ.sum(axis=0)
        #MFPT_BA = (T_Ba@rho)
        df[f'MFPT{dirs[i]}'] = tau
        """
            Rates: SS, NSS, QSD, k*, kF
        """
        #for SS, we use waiting times from non-reduced network (DSS)
        df[f'kSS{dirs[i]}'] = C.dot(np.diag(rDSS[r_s])).dot(rho)
        #for NSS, we use waiting times D^I_s from reduced network
        df[f'kNSS{dirs[i]}'] = C.dot(np.diag(rD[r_s])).dot(rho)
        #kQSD is same as NSS except using qsd instead of boltzmann
        df[f'kQSD{dirs[i]}'] = C.dot(np.diag(rD[r_s])).dot(qsd)
        #finally k* is just 1/MFPT
        df[f'k*{dirs[i]}'] = 1./tau
        df[f'kF{dirs[i]}'] = (rho/T_Ba).sum()  
    return df     
    
    

