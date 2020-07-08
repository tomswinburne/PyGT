r"""
Partial graph transformation: A dimensionality reduction strategy
-----------------------------------------------------------------

This module exploits the graph transformation algorithm to eliminate
nodes that are the least consequential for global dynamics from the 
Markov chain. The resulting network is less sparse, of lower dimensionality,
and is generally better-conditioned. Various strategies are implemented for
choosing nodes to eliminate, namely, ranking nodes based on their mean
waiting times and equilibrium occupation probabilities.

Please cite the following manuscript when using the `dimred.partialGT`
module. [2]_

.. [2] D. Kannan, D. J. Sharpe, T. D. Swinburne, D. J. Wales, "Dimensionality reduction of complex networks with graph transformation" *Phys. Rev. E.* (2020)


"""

import numpy as np
from io import StringIO
import time,os, importlib
#from tqdm import tqdm
np.set_printoptions(linewidth=160)
from .. import ktn_io as kio
from .. import fpt_stats as fpt
from .. import gt_tools as gt
from .. import conversion as convert
from scipy.sparse import save_npz,load_npz, diags, eye, csr_matrix,bmat
from scipy.sparse.linalg import eigs,inv,spsolve
from scipy.sparse.csgraph import connected_components
import scipy as sp
import scipy.linalg as spla
import pandas as pd


def choose_nodes_to_remove(rm_type, percent_retained, region, 
                           BF, escape_time, node_degree, B, trmb=None, pi=None, rm_reg=None):
    """Return an array rm_reg selecting out nodes to remove from
    the `region`.

    TODO: make an elaborate switcher class that allows for
    fine-tuned control over partial GT strategy without a clusterfuck
    of if-else statements.

    Parameters
    ----------
    rm_type : str
        heuristic used to remove nodes, 'escape_time', 'free_energy',
        'node_degree', 'hybrid', or 'combined'
    percent_retained : float
        percent of nodes to keep in reduced network
    region : (N,) array
        boolean array that specifies region in which to remove nodes.
        Ex: region should be IS for computing A->B stats,
        region should be BS for basin escape from B, etc.
    rm_reg : (N,) array
        boolean array selecting nodes to remove. Defaults to None.
        if not None, overwrites rm_reg[region] using rm_type
        and percent_retained.

    Returns
    -------
    rm_reg : (N,) array
        boolean array that specifies nodes to remove within region.
        Note that rm_reg[~region] = False (doesn't remove nodes outside
        of specified region)

    """
    N = len(region)
    if rm_reg is None:
        rm_reg = np.zeros(N, bool)
    Binds = np.nonzero(region)[0]
    V = region.sum()
    if trmb is not None:
        #when the number of remaining nodes is small, change the iteration step size to 1 node at a time
        if V>=1 and V<(4*trmb):
            trmb=1
        if V==0:
            return rm_reg

    if rm_type == 'node_degree':
        rm_reg[node_degree < 2] = True
        #keep nodes that are not in the removal region
        rm_reg[~region] = False 
    elif rm_type == 'escape_time':
        #remove nodes with the smallest escape times
        if trmb is not None:
            to_remove = np.argsort(escape_time[region])[:trmb]
            rm_reg[Binds[to_remove]] = True
        #retain nodes in the top percent_retained percentile of escape time
        else:
            rm_reg[region] = escape_time[region] < np.percentile(escape_time[region], 100.0 - percent_retained)
    elif rm_type == 'free_energy':
        #remove nodes with the highest free energies
        if trmb is not None:
            to_remove = np.argsort(BF[region])[-trmb:]
            rm_reg[Binds[to_remove]] = True
        #retain nodes in the bottom percent_retained percentile of free energy
        else:
            rm_reg[region] = BF[region] > np.percentile(BF[region], percent_retained)   
    elif rm_type == 'hybrid':
        #remove nodes in the top percent_retained percentile of escape time
        time_sel = (escape_time[region] < np.percentile(escape_time[region], 100.0 - percent_retained))
        bf_sel = (BF[region]>np.percentile(BF[region],percent_retained))
        if pi is not None:
            rho = pi[region]/pi[region].sum()
            #use given occupation probabilities instead of free energies
            bf_sel = (rho < np.percentile(rho, 100.0 - percent_retained))
        sel = np.bitwise_and(time_sel, bf_sel)
        #that are also in the lowest percent_retained percentile of free energy
        rm_reg[region] = sel
    elif rm_type == 'combined':
        #multiple escape_time by occupation probability e^-BF
        #high free energy nodes have a small occupation probability 
        #we want to remove nodes with low occupation probability AND small escape time
        #if we multiply together, should make sense to remove the smallest values 
        rho = np.exp(-BF)[region] #stationary probabilities
        rho /= rho.sum()
        if pi is not None:
            rho = pi[region]/pi[region].sum()
        combo_metric = escape_time[region] * rho
        if trmb is not None:
            to_remove = np.argsort(combo_metric)[:trmb]
            rm_reg[Binds[to_remove]] = True
        else:
            rm_reg[region] = combo_metric < np.percentile(combo_metric, 100.0 - percent_retained)
    elif rm_type == 'fund_matrix':
        rho = np.exp(-BF)[region] #stationary probabilities
        rho /= rho.sum()
        if pi is not None:
            rho = pi[region]/pi[region].sum()
        V = region.sum()
        try:
            N = spla.inv(np.eye(V,V) - B[region,:][:,region])
        except Exception as e:
            print(f"Fundamental matrix inversion errored out : {e}")
            print(f"V = {V}")
            print(B[region,:][:,region])
            raise 

        #node_visitations = N@rho #weighted average
        node_visitations = N.sum(axis=1) #unweighted average
        #remove nodes with the least visitations on the path to the absorbing boundary
        if trmb is not None:
            to_remove = np.argsort(node_visitations)[:trmb]
            rm_reg[Binds[to_remove]] = True
        else:
            rm_reg[region] = node_visitations < np.percentile(node_visitations, 100.0 - percent_retained)

    else:
        raise ValueError('Choose a valid GT removal strategy for `rm_type`.')

    return rm_reg  

def prune_intermediate_nodes(beta, data_path, rm_type='hybrid', 
                             percent_retained=10, dopdf=True, screen=True):
    r""" Prune nodes only in the intermediate region between two endpoint
    macrostates of interest using the heuristic specified in `rm_type`. 

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
        First and second moments of first passage time distribution for A->B and B->A [:math:`\mathcal{T}_{\mathcal{B}\mathcal{A}}`, :math:`\mathcal{V}_{\mathcal{B}\mathcal{A}}`, :math:`\mathcal{T}_{\mathcal{A}\mathcal{B}}`, :math:`\mathcal{V}_{\mathcal{A}\mathcal{B}}`]
    pt : (4, 400) array-like
        time in multiples of :math:`\left<t\right>` and first passage time distribution :math:`p(t)\left<t\right>` for A->B and B->A
    gttau : (4,) array-like
        same quantities as `tau` but in reduced network
    gtpt : (4, 400) array-like
        same quantities as `pt` but in reduced network 

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
    tau, pt = fpt.compute_passage_stats(AS, BS, BF, Q)
        
    """Now calculate <tau>, <tau^2>, p(t) after graph transformation"""
    rm_reg = choose_nodes_to_remove(rm_type, percent_retained, IS, BF, escape_time, node_degree)   
    #free energies of retained states
    r_BF = BF[~rm_reg]
    if screen:
        print(f'Nodes to eliminate: {rm_reg.sum()}/{N}, percent retained: {100*(IS.sum()-rm_reg.sum())/IS.sum()}')
        print(f'in A: {rm_reg[AS].sum()}, in B: {rm_reg[BS].sum()}, in I: {rm_reg[IS].sum()}')
    #perform the graph transformation
    GT_B, GT_D, GT_Q, r_N, retry  = gt.gt_seq(N=N,rm_reg=rm_reg,B=B,escape_rates=D,trmb=10,retK=True,Ndense=50,screen=False)
    #compute stats on reduced network
    gttau, gtpt = fpt.compute_passage_stats(AS[~rm_reg], BS[~rm_reg], r_BF, GT_Q)
    
    if dopdf:
        return beta, tau, gttau, pt, gtpt
    else:
        return beta, tau, gttau

def prune_source(beta, data_path, rm_type='hybrid', percent_retained_in_B=90.,
                 dopdf=True, BS=None, screen=True):
    r""" Prune nodes in the source community using the heuristic specified by `rm_type`
    and compute the escape time distribution from the region specified by `BS`
    in the original and graph-transformed networks.

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
        array that selects out the states in the source community. If None, reads in the
        minima in min.B as the source region.


    Returns
    -------
    tau : (2,) array-like
        First and second moments of escape time distribution, [:math:`\left<t\right>_{\mathcal{B}}`, :math:`\left<t^2 \right>_{\mathcal{B}}`]
    pt : (2, 400) array-like
        time in multiples of :math:`\left<t\right>` and escape time distribution :math:`p(t)\left<t\right>`
    gttau : (2,) array-like
        same quantities as `tau` but in reduced network
    gtpt : (2, 400) array-like
        same quantities as `pt` but in reduced network 

    """
    Nmax = None
    B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=data_path,beta=beta,Emax=None,Nmax=Nmax,screen=False)
    node_degree = B.indptr[1:] - B.indptr[:-1]
    D = np.ravel(K.sum(axis=0))
    escape_time = 1./D
    Q = diags(D)-K
    BF = beta*u-s
    BF -= BF.min()
    if BS is None:
        AS, BS = kio.load_AB(data_path,index_sel)
        IS = np.zeros(N, bool)
        IS[~(AS+BS)] = True
        if screen:
            print(f'A: {AS.sum()}, B: {BS.sum()}, I: {IS.sum()}')

    """ First calculate p(t), <tau>, <tau^2> without any GT"""
    tau, pt = fpt.compute_escape_stats(BS, BF, Q)
        
    """Now calculate <tau>, <tau^2>, p(t) after graph transforming away the top 10% of B nodes"""
    rm_reg = choose_nodes_to_remove(rm_type, percent_retained_in_B, BS, BF, escape_time, node_degree)
    #free energies of retained states
    r_BF = BF[~rm_reg]
    if screen:
        print(f'Nodes to eliminate: {rm_reg.sum()/BS.sum()}, percent retained: {100*(BS.sum()-rm_reg.sum())/BS.sum()}')
    GT_B, GT_D, GT_Q, r_N, retry = gt.gt_seq(N=N,rm_reg=rm_reg,B=B,escape_rates=D,trmb=10,retK=True,Ndense=50,screen=False)
    gttau, gtpt = fpt.compute_escape_stats(BS[~rm_reg], r_BF, GT_Q,
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
    node_degree = B.indptr[1:] - B.indptr[:-1]
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
        rm_reg = choose_nodes_to_remove(rm_type, percent_retained, BS, 
            BF, escape_time, node_degree, rm_reg=rm_reg)
        if screen:
            print(f'Percent eliminated from basin: {100*rm_reg[BS].sum()/BS.sum()}')        
    #now do the GT all in one go
    r_B, r_D, r_Q, r_N, retry = gt.gt_seq(N=N,rm_reg=rm_reg,B=B,escape_rates=D,trmb=10,retK=True,Ndense=50,screen=False)
    #free energies and escape times of retained states
    r_BF = BF[~rm_reg]
    print(f'Removed {rm_reg.sum()} of {N} nodes, retained {100*r_N/N} percent')
    #community selectors in reduced network -- update shapes
    r_communities = {}
    for com in communities:
        r_communities[com] = communities[com][~rm_reg]
            
    return r_B, r_D, r_Q, r_N, r_BF, r_communities, rm_reg

def prune_basins_sequentially(beta, data_path, rm_type='hybrid', percent_retained=50., screen=True):
    """ Prune each basin one at a time and feed the reduced network into the
    next round of GT (so that escape times and equilibrium probabilities get
    updated each time a basin is pruned)

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
    node_degree = B.indptr[1:] - B.indptr[:-1]
    D = np.ravel(K.sum(axis=0))
    Q = diags(D)-K
    escape_time = 1./D
    BF = beta*u-s
    BF -= BF.min()
    #pi = np.exp(-BF)
    communities = read_communities(data_path/'communities.dat', index_sel)
    N_original = N
    
    #loop through the basins and graph transform one at a time
    for source_commID in communities:
        #BS is the selector for all nodes in the source community
        BS = communities[source_commID]
        if screen:
            print(f'Source comm: {source_commID}, Source nodes: {BS.sum()}')
        rm_reg = choose_nodes_to_remove(rm_type, percent_retained, BS, 
            BF, escape_time, node_degree)
        if screen:
            print(f'Percent eliminated from basin: {100*rm_reg[BS].sum()/BS.sum()}')
        #perform the GT for this basin alone
        B, D, Q, N, retry = gt.gt_seq(N=N,rm_reg=rm_reg,B=B,escape_rates=D,trmb=10,retK=True,Ndense=50,screen=False)
        #update free energies, escape times, and equilibrium occupation probabilities
        BF = BF[~rm_reg]
        BF -= BF.min()
        #peq in reduced system
        #rK = convert.K_from_Q(Q)
        #nu, vr = spla.eig(rK)
        #qsdo = np.abs(nu.real).argsort()
        #pi = vr[:, qsdo[0]].real
        escape_time = 1./D
        node_degree = B.indptr[1:] - B.indptr[:-1]
        #community selectors in reduced network -- update shapes
        for com in communities:
            communities[com] = communities[com][~rm_reg]        
    #return final network 
    print(f'Removed {N_original-N} of {N_original} nodes,retained: {100 * N/N_original}') 
    return B, D, Q, N, BF, communities
    
