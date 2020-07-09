# -*- coding: utf-8 -*-
r"""
Calculate first passage time statistics with graph transformation
-----------------------------------------------------------------
Contains wrappers to `PyGT.gt_tools` to calculate the mean first passage times
and phenomenological rate constants between endpoint macrostates
:math:`\mathcal{A}` and :math:`\mathcal{B}`.

.. note::

    Install the `pathos` package to parallelize MFPT computations.

"""

import numpy as np
from io import StringIO
import time,os, importlib
#from tqdm import tqdm
np.set_printoptions(linewidth=160)
from . import ktn_io as kio
from . import gt_tools as gt
from . import conversion as convert
from scipy.sparse import save_npz,load_npz, diags, eye, csr_matrix,bmat
from scipy.sparse.linalg import eigs,inv,spsolve
from scipy.sparse.csgraph import connected_components
import scipy as sp
import scipy.linalg as spla
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool

def compute_passage_stats(AS, BS, BF, Q, dopdf=True):
    r"""Compute the A->B and B->A first passage time distribution,
    first moment, and second moment using eigendecomposition.

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
        First and second moments of first passage time distribution for A->B and B->A [:math:`\mathcal{T}_{\mathcal{B}\mathcal{A}}`, :math:`\mathcal{V}_{\mathcal{B}\mathcal{A}}`, :math:`\mathcal{T}_{\mathcal{A}\mathcal{B}}`, :math:`\mathcal{V}_{\mathcal{A}\mathcal{B}}`]
    pt : (4, 400) array-like
        time in multiples of :math:`\left<t\right>` and first passage time distribution :math:`p(t)\left<t\right>` for A->B and B->A

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
    r"""Compute escape time distribution and first and second moment
    from the basin specified by `BS` using eigendecomposition.

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
        First and second moments of escape time distribution, [:math:`\left<t\right>_{\mathcal{B}}`, :math:`\left<t^2 \right>_{\mathcal{B}}`]
    pt : (2, 400) array-like
        time in multiples of :math:`\left<t\right>` and escape time distribution :math:`p(t)\left<t\right>`

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

def get_intermicrostate_mfpts_GT(temp, data_path, pool_size=None, **kwargs):
    r"""Compute matrix of inter-microstate MFPTs with GT.

    Parameters
    ----------
    temp : float
        Effective temperature :math:`k_B T`.
    data_path : str or Path
        Path to data containing min.data, ts.data files.
    pool_size : int
        Number of cores over which to parallelize computation.

    Returns
    -------
    mfpt : np.ndarray[float64] (N,N)
        matrix of inter-microstate MFPTs between all pairs of nodes
    rho : np.ndarray[float64] (N,)
        stationary distribution of Markov chain

    """

    beta = 1./temp
    #GT setup
    B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=data_path,beta=beta,Emax=None,Nmax=None,screen=False)
    D = np.ravel(K.sum(axis=0))
    BF = beta*u-s
    BF -= BF.min()
    rho = np.exp(-BF)
    rho /= rho.sum()
    mfpt = np.zeros((N,N))

    matrix_elements = []
    for i in range(N):
        for j in range(N):
            if i < j:
                matrix_elements.append((i,j))

    def given_ij(ij):
        i, j = ij
        MFPTAB, MFPTBA = compute_MFPTAB(i, j, B, D, **kwargs)
        return MFPTAB, MFPTBA

    if pool_size is None:
        for ij in matrix_elements:
            i, j = ij
            mfpt[i][j], mfpt[j][i] = given_ij(ij)
    else:
        with Pool(processes=pool_size) as p:
            results = p.map(given_ij, matrix_elements)
        for k, result in enumerate(results):
            i, j = matrix_elements[k]
            mfpt[i][j], mfpt[j][i] = result

    return mfpt, rho

def compute_MFPTAB(i, j, B, escape_rates=None, K=None, **kwargs):
    r"""Compute the inter-microstate :math:`i\leftrightarrow j` MFPT using GT.

    Unlike ``compute_rates()`` function, which assumes there is at least 2 microstates
    in the absorbing macrostate, this function does not require knowledge of equilibrium
    occupation probabilities since :math:`\mathcal{T}_{ij}=\tau_j^\prime/B_{ij}^\prime`.

    Parameters
    ----------
    i : int
        node-ID (0-indexed) of first microstate.
    j : int
        node-ID (0-indexed) of second microstate.
    B : matrix (N,N)
        branching probability matrix. If dense, kwarg dense=True should be set.
    escape_rates : array-like (N,)
        vector of escape rates, i.e. inverse waiting times from each node.
    K : matrix (N,N)
        rate matrix with off-diagonal elements :math:`K_{ij}` and diagonal
        elements :math:`K_{ii} = 0`. If specified, `escape_rates` keyword
        overridden with sum of columns of `K`. Defaults to None.

    Returns
    -------
    MFPTij : float
        mean first passage time :math:`i \leftarrow j`
    MFPTji : float
        mean first passage time :math:`j \leftarrow i`

    """

    if K is None and escape_rates is None:
        raise ValueError('Either escape_rates or K must be specified.')

    if escape_rates is not None:
        D = escape_rates
    if K is not None:
        D = np.ravel(K.sum(axis=0))

    N = B.shape[0]
    AS = np.zeros(N, bool)
    AS[i] = True
    BS = np.zeros(N, bool)
    BS[j] = True
    #GT away all I states
    inter_region = ~(AS+BS)
    #left with a 2-state network
    rB, tau_Fs, rQ, rN, retry = gt.GT(rm_vec=inter_region,B=B,tau=1.0/D,retK=True,block=1,**kwargs)
    rB = rB.todense()
    #escape time tau_F
    rD = 1.0/tau_Fs
    #remaining network only has 1 in A and 1 in B = 2 states
    r_AS = AS[~inter_region]
    r_BS = BS[~inter_region]
    #tau_a^F / P_Ba^F
    P_BA = rB[r_BS, :][:, r_AS]
    P_AB = rB[r_AS, :][:, r_BS]
    MFPTBA = tau_Fs[r_AS]/P_BA
    MFPTAB = tau_Fs[r_BS]/P_AB
    return MFPTAB[0,0], MFPTBA[0,0]

def compute_rates(AS, BS, B, escape_rates=None, K=None, initA=None, initB=None, BF=None,
    MFPTonly=True, fullGT=False, pool_size=None, **kwargs):
    r""" Calculate kSS, kNSS, kF, k*, kQSD, and MFPT for the transition path
    ensemble AS --> BS from rate matrix K. K can be the matrix of an original
    Markov chain, or a partially graph-transformed Markov chain.

    Differs from ``compute_passage_stats()`` in that this function removes all intervening states
    using GT before computing fpt stats and rates on the fully reduced network
    with state space :math:`(\mathcal{A} \cup \mathcal{B})`. This implementation also does not rely on a full
    eigendecomposition of the non-absorbing matrix; it instead performs a matrix inversion,
    or if `fullGT` is specified, all nodes in the set :math:`(\mathcal{A} \cup b)^\mathsf{c}` are removed
    for each :math:`b \in \mathcal{B}` so that the
    MFPT is given by an average:

    .. math::

        \begin{equation}
        \mathcal{T}_{\mathcal{A}\mathcal{B}} = \frac{1}{\sum_{b \in
        \mathcal{B}} p_b(0)} \sum_{b \in \mathcal{B}} \frac{p_b(0)
        \tau^\prime_b}{1-P^\prime_{bb}}
        \end{equation}

    If the MFPT is less than :math:`10^{20}`, `fullGT` does not need to be
    specified since the inversion of the non-absorbing matrix is numerically
    stable. However, for extremely metastable systems, `fullGT` should be
    specified to ensure numerical stability of the operation.

    TODO: include equations for kSS, kNSS, etc.
    TODO: debug case where A and B each only contain one state.

    Parameters
    ----------
    AS : np.ndarray[bool] (N,)
        selects out noes in the :math:`\mathcal{A}` set.
    BS : np.ndarray[bool] (N,)
        selects out nodes in the :math:`\mathcal{B}` set.
    B : matrix (N,N)
        branching probability matrix. If dense, an additional keyword
        `dense=True` should be specified.
    escape_rates : array-like (N,)
        vector of escape rates, i.e. inverse waiting times from each node.
    K : matrix (N,N)
        rate matrix with off-diagonal elements :math:`K_{ij}` and diagonal
        elements :math:`K_{ii} = 0`. If specified, `escape_rates` keyword
        overridden with sum of columns of `K`. Defaults to None.
    initA : array-like (N,)
        normalized initial occupation probabilities in :math:`\mathcal{A}` set.
        Defaults to Boltzmann distribution if BF is specified.
    initB : array-like (N,)
        normalized initial occupation probabilities in :math:`\mathcal{B}` set.
        Defaults to Boltzmann distribution if BF is specified.
    BF : array-like (N,)
        Free energies of nodes, used to compute Boltzmann :math:`\pi_i =\rm{exp}(-\beta F_i)/\sum_i \rm{exp}(-\beta F_i)`.
    MFPTonly : bool
        If True, only MFPTs are calculated (rate calculations ignored).
    fullGT : bool
        If True, all source nodes are isolated with GT to obtain the average
        MFPT.
    pool_size : int
        Number of cores over which to parallelize fullGT computation.

    Returns
    -------
    df : pandas DataFrame
        single row, columns are 'MFPTAB', 'kSSAB', 'kNSSAB', 'kQSDAB', 'k*AB', 'kFAB',
        and the same quantities for B<-A


    """

    N = len(AS)
    assert(N==len(BS))

    if AS.sum()==1 and BS.sum()==1:
        raise NotImplementedError('There must be at least 2 microstates in A and B.')

    if K is None and escape_rates is None:
        raise ValueError('Either escape_rates or K must be specified.')

    if escape_rates is not None:
        D = escape_rates
    if K is not None:
        D = np.ravel(K.sum(axis=0))

    inter_region = ~(AS+BS)
    r_AS = AS[~inter_region]
    r_BS = BS[~inter_region]
    rDSS = D[~inter_region]

    if BF is not None:
        r_BF = BF[~inter_region]
        rhoA = np.exp(-r_BF[r_AS])
        initA = rhoA/rhoA.sum()
        rhoB = np.exp(-r_BF[r_BS])
        initB = rhoB/rhoB.sum()

    #use GT to renormalize away all I states

    rB, rtau, rQ, rN, retry = gt.GT(rm_vec=inter_region,B=B,tau=1.0/D,retK=True,block=1,**kwargs)
    rD = 1.0/rtau

    #first do A->B direction, then B->A
    #r_s is the non-absorbing region (A first, then B)
    df = pd.DataFrame()
    dirs = ['BA', 'AB']
    inits = [initA, initB]

    for i, r_s in enumerate([r_AS, r_BS]) :
        #local equilibrium distribution in r_s
        rho = inits[i]
        #MFPTs to B from each source microstate a
        T_Ba = np.zeros(r_s.sum())
        if not fullGT:
            #MFPT calculation via matrix inversion
            invQ = spla.inv(rQ[r_s,:][:,r_s].todense())
            tau = invQ.dot(rho).sum(axis=0)
            T_Ba = invQ.sum(axis=0)
        #compare to individual T_Ba quantities from further GT compression
        else:
            def disconnect_sources(s):
                #disconnect all source nodes except for `s`
                rm_reg = np.zeros(rN, bool)
                rm_reg[r_s] = True
                aind = r_s.nonzero()[0][s]
                #print(f'Disconnecting source node {aind}')
                rm_reg[r_s.nonzero()[0][s]] = False

                rfB, tau_Fs, rfQ, rfN, retry = gt.GT(rm_vec=rm_reg,B=rB,tau=1.0/rD,retK=True,block=1,Ndense=1)
                rfB = rfB.todense()
                #escape time tau_F
                rfD = 1./tau_Fs
                #remaining network only as 1 in A and 1 in B = 2 states
                rf_s = r_s[~rm_reg]
                #tau_a^F / P_Ba^F
                P_Ba = np.ravel(rfB[~rf_s,:][:,rf_s].sum(axis=0))[0]
                return tau_Fs[rf_s][0]/P_Ba

            if pool_size is None:
                for s in range(r_s.sum()):
                    T_Ba[s] = disconnect_sources(s)
            else:
                with Pool(processes=pool_size) as p:
                    T_Ba = p.map(disconnect_sources, [s for s in range(r_s.sum())])
            #MFPT_BA = (T_Ba@rho)
            tau = T_Ba@rho
        df[f'MFPT{dirs[i]}'] = [tau]
        """
            Rates: SS, NSS, QSD, k*, kF
        """
        if not MFPTonly:
            #eigendecomposition of rate matrix in non-abosrbing region
            #for A, full_RK[r_A, :][:, r_A] is just a 5x5 matrix
            l, v = spla.eig(rQ[r_s,:][:,r_s].todense())
            #order the eigenvalues from smallest to largest -- they are positive since Q = D - K instead of K-D
            qsdo = np.abs(l.real).argsort()
            nu = l.real[qsdo]
            #v[:, qsdo[0]] is the eigenvector corresponding to smallest eigenvalue
            #aka quasi=stationary distribution
            qsd = v[:,qsdo[0]]
            qsd /= qsd.sum()
            #committor C^B_A: probability of reaching B before A: 1_B.B_BA^I (eqn 6 of SwinburneW20)
            C = np.ravel(rB[~r_s,:][:,r_s].sum(axis=0))
            #for SS, we use waiting times from non-reduced network (DSS)
            df[f'kSS{dirs[i]}'] = [C.dot(np.diag(rDSS[r_s])).dot(rho)]
            #for NSS, we use waiting times D^I_s from reduced network
            df[f'kNSS{dirs[i]}'] = [C.dot(np.diag(rD[r_s])).dot(rho)]
            #kQSD is same as NSS except using qsd instead of boltzmann
            df[f'kQSD{dirs[i]}'] = [C.dot(np.diag(rD[r_s])).dot(qsd)]
            #k* is just 1/MFPT
            df[f'k*{dirs[i]}'] = [1./tau]
            #and kF is <1/T_Ab>
            df[f'kF{dirs[i]}'] = [(rho/T_Ba).sum()]
    return df


def rates_cycle(temps, data_path, suffix='', **kwargs):
    """Compute rates and mean first passage times between A and B sets for a range of temperatures.

    Parameters
    ----------
    temps : (ntemps, ) array-like
        list of temperatures at which to compute A<->B rates
    data_path : str or Path object
        path to data
    suffix : str
        suffix for output file name `ratecycle{suffix}.csv`

    Returns
    -------
    df : pandas DataFrame
        rows are temperatures, columns are 'MFPTAB', 'kSSAB', 'kNSSAB', 'kQSDAB', 'k*AB', 'kFAB',
        and the same quantities for B<-A

    """
    dfs = []
    for temp in temps:
        beta = 1./temp
        B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=data_path,beta=beta,Emax=None,Nmax=None,screen=False)
        D = np.ravel(K.sum(axis=0))
        BF = beta*u-s
        BF -= BF.min()
        AS,BS = kio.load_AB(data_path,index_sel)
        IS = np.zeros(N, bool)
        IS[~(AS+BS)] = True
        df = compute_rates(AS, BS, B, escape_rates=D, K=K, BF=BF, **kwargs)
        df['T'] = [temp]
        dfs.append(df)
    bigdf = pd.concat(dfs)
    bigdf.to_csv(f'csvs/ratescycle{suffix}.csv')
    return bigdf
