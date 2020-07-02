r""" 
Optimal Markovian coarse-graining given a community structure
-------------------------------------------------------------

This module provides functions to analyze and estimate coarse-grained continuous-time
Markov chains given a partiioning :math:`\mathcal{C} = \{I, J, ...\}` of the :math:`V` nodes into :math:`N<V` communities.
Various formulations for the inter-community rates are implemented, including the local
equilibrium approximation, Hummer-Szabo relation, and other routes to obtain the optimal
coarse-grained Markov chain for a given community structure. [1]_

.. [1] D. Kannan, D. J. Sharpe, T. D. Swinburne, D. J. Wales, "Dimensionality reduction of Markov chains from mean first passage times using graph transformation." *J. Chem. Phys.* (2020)


"""
from .. import fpt_stats as fpt

import numpy as np
from numpy.linalg import inv
import scipy 
import scipy.linalg as spla 
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.linalg import eig
from scipy.linalg import expm
from pathlib import Path
import pandas as pd
import os
import subprocess


class Analyze_KTN(object):
    r""" Analyze a kinetic transition network (KTN) with a specified community structure.

    Attributes
    ----------
    path : str or Path object
        path to directory with all relevant files
    K : array-like (nnodes, nnodes)
        Rate matrix with elements :math:`K_{ij}` corresponding to the `i \leftarrow j` transition rate, 
        and diagonal elements :math:`K_{ii} = -\sum_\gamma K_{\gamma i}` such that the columns sum to zero.
    pi : array-like (nnodes,)
        Stationary distribution of nodes, :math:`\pi_i`, i.e. vector of equilibrium occupation probabilities.
    commpi : array-like (ncomms,)
        Stationary distribution of communities, :math:`\Pi_J = \sum_{j \in J} \pi_j`.
    communities : dict
        dictionary mapping community IDs (1-indexed) to node IDs (1-indexed).
    commdata : str
        Filename, located in directory specified by `path`, of a single-column file where 
        each line contains the community ID (0-indexed) of the node specified by the line number in the file

    Note
    ----
    Either `communities` or `commdata` must be specified.

    """

    def __init__(self, path, K=None, pi=None, commpi=None, communities=None,
                 commdata=None, temp=None, thresh=None):
        self.path = Path(path) 
        self.K = K
        self.pi = pi
        self.commpi = commpi
        if K is not None and pi is not None:
            commpi = self.get_comm_stat_probs(np.log(pi), log=False)
            self.commpi = commpi
        if temp is not None:
            logpi, Kmat = read_ktn_info(f'T{temp:.3f}', log=True)
            pi = np.exp(logpi)
            self.K = Kmat
            self.pi = pi/pi.sum()
            commpi = self.get_comm_stat_probs(logpi, log=False)
            self.commpi = commpi/commpi.sum()
        if communities is not None:
            self.communities = communities
        elif commdata is not None:
            self.communities = read_communities(self.path/commdata)
        else:
            if thresh is not None and temp is not None:
                commdata=f'communities_G{thresh:.2f}_T{temp:.3f}.dat'
                self.communities = read_communities(self.path/commdata)
            else:
                raise AttributeError('Either communities or commdata must' \
                                    'be specified.')

    def construct_coarse_rate_matrix_LEA(self, temp=None):
        """Calculate the coarse-grained rate matrix obtained using the local
        equilibrium approximation (LEA)."""

        if self.K is None:
            if temp is not None:
                logpi, Kmat = read_ktn_info(f'T{temp:.3f}', log=True)
                pi = np.exp(logpi)
                self.K = Kmat
                self.pi = pi/pi.sum()
                commpi = self.get_comm_stat_probs(logpi, log=False)
                self.commpi = commpi/commpi.sum()
            else:
                raise ValueError("The attributes K, pi, and commpi must be specified.")
        
        N = len(self.communities)
        Rlea = np.zeros((N,N))

        for i in range(N):
            for j in range(N):
                if i < j:
                    ci = np.array(self.communities[i+1]) - 1
                    cj = np.array(self.communities[j+1]) - 1
                    Rlea[i, j] = np.sum(self.K[np.ix_(ci, cj)]@self.pi[cj]) / self.commpi[j]
                    Rlea[j, i] = np.sum(self.K[np.ix_(cj, ci)]@self.pi[ci]) / self.commpi[i]
        
        for i in range(N):
            Rlea[i, i] = -np.sum(Rlea[:, i])
        return Rlea

    def construct_coarse_rate_matrix_Hummer_Szabo(self, temp=None):
        r""" Calculate the optimal coarse-grained rate matrix using the Hummer-Szabo
        relation, aka Eqn. (12) in Hummer & Szabo *J. Phys. Chem. B.* (2015)."""

        if self.K is None:
            if temp is not None:
                logpi, Kmat = read_ktn_info(f'T{temp:.3f}', log=True)
                pi = np.exp(logpi)
                self.K = Kmat
                self.pi = pi/pi.sum()
                commpi = self.get_comm_stat_probs(logpi, log=False)
                self.commpi = commpi/commpi.sum()
            else:
                raise ValueError("The attributes K, pi, and commpi must be specified.")

        N = len(self.communities)
        V = len(self.pi)
        D_N = np.diag(self.commpi)
        D_V = np.diag(self.pi)
        #construct clustering matrix M from community assignments
        M = np.zeros((V, N))
        for ci in self.communities:
            col = np.zeros((V,))
            comm_idxs = np.array(self.communities[ci]) - 1
            col[comm_idxs] = 1.0
            M[:, ci-1] = col

        Pi_col = self.commpi.reshape((N, 1))
        pi_col = self.pi.reshape((V, 1))
        mat_to_invert = pi_col@np.ones((1,V)) - self.K
        first_inverse = spla.inv(mat_to_invert)
        #check that Pi = M^T pi
        Pi_calc = M.T@self.pi
        for entry in np.abs(self.commpi - Pi_calc):
            assert(entry < 1.0E-10)

        #H-S relation
        second_inversion = spla.inv(M.T@first_inverse@D_V@M)
        R_HS = Pi_col@np.ones((1,N)) - D_N@second_inversion
        if not check_detailed_balance(self.commpi, R_HS):
            print(f'HS does not satisfy detailed balance')
        return R_HS

    def construct_coarse_rate_matrix_KKRA(self, mfpt=None, GT=False, temp=None, **kwargs):
        r"""Calculate optimal coarse-grained rate matrix using Eqn. (79) 
        of Kells et al. *J. Chem. Phys.* (2020), aka the KKRA expression
        in Eqn. (10) of Kannan et al. *J. Chem. Phys.* (2020).
        
        Parameters
        ----------
        mfpt : (nnodes, nnodes)
            Matrix of inter-microstate MFPTs between all pairs of nodes. Defaults to None.
        GT : bool
            If True, matrix of inter-microstate MFPTs is computed with GT using the `graph_tran.fpt_stats` module.
            Kwargs can then be specified for GT (such as the pool_size for parallelization). Defaults to False.

        """

        if self.K is None:
            if temp is not None:
                logpi, Kmat = read_ktn_info(f'T{temp:.3f}', log=True)
                pi = np.exp(logpi)
                self.K = Kmat
                self.pi = pi/pi.sum()
                commpi = self.get_comm_stat_probs(logpi, log=False)
                self.commpi = commpi/commpi.sum()
            else:
                raise ValueError("The attributes K, pi, and commpi must be specified.")
        N = len(self.communities)
        n = len(self.pi)
        D_N = np.diag(self.commpi)
        D_n = np.diag(self.pi)

        #construct clustering matrix M from community assignments
        M = np.zeros((n, N))
        for ci in self.communities:
            col = np.zeros((n,))
            comm_idxs = np.array(self.communities[ci]) - 1
            col[comm_idxs] = 1.0
            M[:, ci-1] = col

        Pi_col = self.commpi.reshape((N, 1))
        pi_col = self.pi.reshape((n, 1))
        if GT:
            mfpt = fpt.get_intermicrostate_mfpts_GT(temp, self.path, **kwargs)
        elif mfpt is None:
            mfpt = self.get_MFPT_from_Kmat(self.K)
        R = Pi_col@np.ones((1,N)) - D_N@spla.inv(Pi_col@Pi_col.T +
                                                 M.T@D_n@mfpt@pi_col@Pi_col.T -
                                                 M.T@D_n@mfpt@D_n@M)
        if not check_detailed_balance(self.commpi, R):
            print(f'KKRA does not satisfy detailed balance')
        return R

    def get_intermicrostate_mfpts_linear_solve(self):
        r"""Calculate the matrix of inter-microstate MFPTs between all pairs of nodes
        by solving a system of linear equations given by Eq.(8) of 
        Kannan et al. *J. Chem. Phys.* (2020)."""

        K = ktn.K
        n = K.shape[0]
        mfpt = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i==j:
                    mfpt[i][j] = 0.
                else:
                    try:
                        mfpt[i][j] = -spla.solve(K[np.arange(n)!=i, :][:, np.arange(n)!=i],
                                                (np.arange(n)==j)[np.arange(n)!=i]).sum()
                    except scipy.linalg.LinAlgWarning as err:
                        raise Exception('LinAlgWarning') 
        return mfpt

    def get_intermicrostate_mfpts_fundamental_matrix(self):
        r"""Calculate the matrix of inter-microstate MFPTs between all pairs of nodes
        using Eq. (6) of Kannan et al. *J. Chem. Phys.* (2020). """

        K = ktn.K
        pi = ktn.pi
        nmin = K.shape[0]
        pioneK = spla.inv(pi.reshape((nmin,1))@np.ones((1,nmin)) + K)
        zvec = np.diag(pioneK)
        mfpt = np.diag(1./pi)@(pioneK - zvec.reshape((nmin,1))@np.ones((1,nmin)))
        return mfpt

    def get_intercommunity_MFPTs_linear_solve(self):
        r"""Calculate the true MFPTs between communities by inverting the non-absorbing
        rate matrix. Equivalent to Eqn. (14) in Swinbourne & Wales *JCTC* (2020)."""
       
        K = ktn.K
        pi = ktn.pi
        n = K.shape[0]
        N = len(self.communities)
        mfpt = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i==j:
                    mfpt[i][j] = 0.
                else:
                    ci = np.array(self.communities[i+1]) - 1
                    cj = np.array(self.communities[j+1]) - 1
                    cond = np.ones((n,), dtype=bool)
                    cond[ci] = False
                    initial_cond = np.zeros((n,))
                    #initialize probability density to local boltzman in cj
                    initial_cond[cj] = pi[cj]/pi[cj].sum()
                    mfpt[i][j] = -spla.solve(K[cond, :][:, cond],
                                            initial_cond[cond]).sum()
        return mfpt

    def get_intercommunity_weighted_MFPTs(self, pi, commpi, mfpt):
        r"""Comppute the matrix :math:`\textbf{t}_{\rm C}` of appropriately weighted
        inter-community MFPTs, as defined in Eq. (18) in Kannan et al. *J. Chem. Phys.*
        (2020)."""

        N = len(self.communities)
        tJI = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                ci = np.array(self.communities[i+1]) - 1
                cj = np.array(self.communities[j+1]) - 1
                tJI[j][i] = pi[cj]@ mfpt[cj,:][:,ci] @ pi[ci] / (commpi[i]*commpi[j])
                tJI[j][i] -= pi[cj] @ mfpt[cj,:][:,cj] @ pi[cj] / (commpi[j])**2

        return tJI

    def get_thetaIJ(self, pi, commpi, mfpt):
        r"""Compute the matrix with elements :math:`\theta_{IJ}` as defined in Eq. (16) in Kannan et al.
        *J. Chem. Phys.* (2020).
        """

        N = len(self.communities)
        tJI = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                ci = np.array(self.communities[i+1]) - 1
                cj = np.array(self.communities[j+1]) - 1
                #find local min i* in ci and j* in cj (sort by energy)
                #tji = get_mfpt_between_states_GT()
                tJI[j][i] = pi[cj]@ mfpt[cj,:][:,ci] @ pi[ci] / (commpi[i]*commpi[j])
                #tJI[j][i] -= pi[cj] @ mfpt[cj,:][:,cj] @ pi[cj] / (commpi[j])**2

        return tJI

    def construct_coarse_rate_matrix_from_MFPTs(self, temp=None):
        """ Calculate a rate matrix using  MFPTs between communities.
        Note that the true-intercommunity MFPTs cannot be used to construct
        a reduced Markov chain that satisfies detailed balance."""
        
        if self.K is None:
            if temp is not None:
                logpi, Kmat = read_ktn_info(f'T{temp:.3f}', log=True)
                pi = np.exp(logpi)
                self.K = Kmat
                self.pi = pi/pi.sum()
                commpi = self.get_comm_stat_probs(logpi, log=False)
                self.commpi = commpi/commpi.sum()
            else:
                raise ValueError("The attributes K, pi, and commpi must be specified.")
        # get R from matrix of MFPTs
        N = len(self.communities)
        D_N = np.diag(self.commpi)
        mfpt = self.get_intercommunity_MFPTs_linear_solve(self)
        
        matrix_of_ones = np.ones((N,1))@np.ones((1,N))
        #R_MFPT = inv(MFPT)@(inv(D_N) - matrix_of_ones)
        R_MFPT = spla.solve(mfpt,np.diag(1.0/self.commpi) - matrix_of_ones)
        check_detailed_balance(self.commpi, R_MFPT)
        return R_MFPT

    def get_free_energies_from_rates(self, R, thresh, temp, kB=1.0, planck=1.0):
        """ Estimate free energies of all macrostates and transition states
        between macrostates from an arbitrary rate matrix R. Write out a `min.data`
        and `ts.data` file for coarse-grained network.
        
        TODO: Get largest connected component of coarse-grained rate matrix.

        Warning
        -------
        It is better to write the coarse-grained matrix to a `rate_matrix.dat`
        file specifying all of the entries of the matrix in dense format, and then
        use this file as input to PATHSAMPLE as opposed to attepting to write out `min.data`
        or `ts.data` files representing the coarse-grained network.

        """

        N = len(self.communities)
        #subtract diagonal from R to recover only postive/zero entries
        R_nodiag = R - np.diag(np.diag(R))
        #replace any negative, near-zero entries with zero
        if np.any(R_nodiag < 0.0):
            R_nodiag = np.where(np.abs(R_nodiag) < 1.E-13, np.zeros_like(R_nodiag), R_nodiag)
        #print(R_nodiag)
        if np.any(R_nodiag < 0.0):
            raise ValueError('The rate matrix R has negative entries.')
        if np.all(R_nodiag == 0.0):
            raise ValueError('The rate matrix R has all zero entries.')
        #TODO: first get connected set of R_nodiag (not the same as nonzero)
        #TODO: make sure A is connected to B
        #set a reference minima's p_eq to be 1
        idx, idy = np.nonzero(R_nodiag)
        commpi_renorm = {}
        commpi_renorm[idx[0]] = 1.0
        for i in range(len(idx)):
            if idy[i] not in commpi_renorm:
                dbalance = R_nodiag[idy[i], idx[i]] / R_nodiag[idx[i], idy[i]]
                if idx[i] not in commpi_renorm:
                    raise ValueError('The rate matrix R has unconnected \
                                     components.')
                commpi_renorm[idy[i]] = commpi_renorm[idx[i]]*dbalance 
        #calculate free energies for this connected set
        min_nrgs = {}
        if temp==3.0:
            print(commpi_renorm)
        for key in commpi_renorm:
            if commpi_renorm[key] <= 0.0:
                min_nrgs[key] = np.inf
            else:
                min_nrgs[key] = -kB*temp*np.log(commpi_renorm[key])
        if len(min_nrgs) != N:
            raise ValueError('The rate matrix R has unconnected components.')
        #create min.data file
        minfree = np.loadtxt(self.path/f'G{thresh:.1f}/min.data.regrouped.{temp:.10f}.G{thresh:.1f}')
        correction_factor = minfree[0, 1]
        with open(self.path/f'min.data.T{temp:.3f}','w') as f:
            for i in range(N):
                f.write(f'{min_nrgs[i]} {correction_factor} 1 1.0 1.0 1.0\n') 

        ts_nrgs_LJ = []
        Ls = []
        Js = []
        df = pd.DataFrame(columns=['nrg','fvibts','pointgroup','min1','min2','itx','ity','itz'])
        #loop over connected minima
        for i in range(len(idx)):
            L = idx[i]
            J = idy[i]
            if L < J:
                #for nonzeros rates, calculate free energy
                free = min_nrgs[J] - kB*temp*np.log(planck*R[L,J]/(kB*temp))
                ts_nrgs_LJ.append(free)
                Ls.append(L)
                Js.append(J)
        df = pd.DataFrame()
        df['nrg'] = ts_nrgs_LJ
        df['fvibts'] = 0.0
        df['pointgroup'] = 1
        df['min1'] = np.array(Ls)+1 #community ID is 1-indexed
        df['min2'] = np.array(Js)+1
        df['itx'] = 1.0
        df['ity'] = 1.0
        df['itz'] = 1.0
        df.to_csv(self.path/f'ts.data.T{temp:.3f}',header=False, index=False, sep=' ')


    def get_timescale_error(self, m, K, R):
        """ Calculate the ith timescale error for i in {1,2,...m} of a
        coarse-grained rate matrix R compared to the full matrix K.
        
        Parameters
        ----------
        m : int
            Number of dominant eigenvalues (m < N)
        K : np.ndarray[float] (V, V)
            Rate matrix for full network
        R : np.ndarray[float] (N, N)
            Coarse-grained rate matrix

        Returns
        -------
        timescale_errors : np.ndarray[float] (m-1,)
            Errors for m-1 slowest timescales
        
        """
        
        ncomms = len(self.communities)
        if m >= ncomms:
            raise ValueError('The number of dominant eigenvectors must be' \
                             'less than the number of communities.')
        Kevals, Kevecs = calc_eigenvectors(K, m, which_eig='SM')
        Revals, Revecs = calc_eigenvectors(R, m, which_eig='SM')
        #the 0th eigenvalue corresponds to infinite time
        Ktimescales = -1./Kevals[1:]
        Rtimescales = -1./Revals[1:]
        timescale_errors = np.abs(Rtimescales - Ktimescales)
        return timescale_errors

    def calculate_eigenfunction_error(self, m, K, R):
        r""" Calculate the :math:`i^{\rm th}` eigenvector approximation error for :math:`i \in {1, 2,
        ... m}` of a coarse-grained rate matrix `R` by comparing its eigenvector 
        to the correspcorresponding eigenvector of the full matrix.
        """
        
        ncomms = len(self.communities)
        if m >= ncomms:
            raise ValueError('The number of dominant eigenvectors must be' \
                             'less than the number of communities.')
        Kevals, Kevecs = calc_eigenvectors(K, m, which_eig='SM')
        Revals, Revecs = calc_eigenvectors(R, m, which_eig='SM')
        errors = np.zeros((m,))
        for i in range(0, m):
            print(i)
            for ck in self.communities:
                #minima in community ck
                minima = np.array(self.communities[ck]) - 1
                coarse_evec = np.tile(Revecs[i, ck-1], len(minima)) #scalar
                errors[i] += np.linalg.norm(coarse_evec - Kevecs[i, minima])
        return errors

    def get_comm_stat_probs(self, logpi, log=True):
        """ Calculate the community stationary probabilities by summing over
        the stationary probabilities of the nodes in each community.
        
        Parameters
        ----------
        logpi : list (nnodes,)
            log stationary probabilities of node in original Markov chain

        Returns
        -------
        commpi : list (ncomms,)
            stationary probabilities of communities in coarse coarse_network

        TODO: account for unconnected minima (set their occupation probs to 0)
        """

        pi = np.exp(logpi)
        if (np.sum(pi) - 1.0) > 1.E-10:
            pi = pi/np.sum(pi)
            logpi = np.log(pi)
        ncomms = len(self.communities)
        logcommpi = np.zeros((ncomms,))
        for ci in self.communities:
            #zero-indexed list of minima in community ci
            nodelist = np.array(self.communities[ci]) - 1 
            logcommpi[ci-1] = -np.inf
            for node in nodelist:
                logcommpi[ci-1] = np.log(np.exp(logcommpi[ci-1]) + np.exp(logpi[node]))
        commpi = np.exp(logcommpi)
        assert abs(np.sum(commpi) - 1.0) < 1.E-10
        if log:
            return logcommpi
        else:
            return commpi


"""Functions to analyze KTNs without any community structure."""

def read_ktn_info(self, suffix, log=False):
    """ Read in input files `stat_prob_{suffix}.dat`, `ts_weights_{suffix}.dat`, and `ts_conns_{suffix}.dat` 
    describing the stationary probabilities and connectivity of the nodes in the Markov chain to be analyzed.
    Files must be located in the directory specified by `self.path`."""

    logpi = np.loadtxt(self.path/f'stat_prob_{suffix}.dat')
    pi = np.exp(logpi)
    nnodes = len(pi)
    assert(abs(1.0 - np.sum(pi)) < 1.E-10)
    logk = np.loadtxt(self.path/f'ts_weights_{suffix}.dat', 'float')
    k = np.exp(logk)
    tsconns = np.loadtxt(self.path/f'ts_conns_{suffix}.dat', 'int')
    Kmat = np.zeros((nnodes, nnodes))
    for i in range(tsconns.shape[0]):
        Kmat[tsconns[i,1]-1, tsconns[i,0]-1] = k[2*i]
        Kmat[tsconns[i,0]-1, tsconns[i,1]-1] = k[2*i+1]
    #set diagonals
    for i in range(nnodes):
        Kmat[i, i] = 0.0
        Kmat[i, i] = -np.sum(Kmat[:, i])
    #identify isolated minima (where row of Kmat = 0)
    #TODO: identify unconnected minima
    Kmat_nonzero_rows = np.where(~np.all(Kmat==0, axis=1))
    Kmat_connected = Kmat[np.ix_(Kmat_nonzero_rows[0],
                                    Kmat_nonzero_rows[0])]
    pi = pi[Kmat_nonzero_rows]/np.sum(pi[Kmat_nonzero_rows])
    logpi = np.log(pi)
    assert(len(pi) == Kmat_connected.shape[0])
    assert(Kmat_connected.shape[0] == Kmat_connected.shape[1])
    assert(np.all(np.abs(Kmat_connected@pi)<1.E-10))

    if log:
        return logpi, Kmat_connected 
    else:
        return pi, Kmat_connected

def calc_eigenvectors(K, k, which_eig='SM', norm=False):
    """ Calculate `k` dominant eigenvectors and eigenvalues of sparse matrix
     using the implictly restarted Arnoldi method."""
    evals, evecs = eigs(K, k, which=which_eig)
    evecs = np.transpose(evecs)
    evecs = np.array([evec for _,evec in sorted(zip(list(evals),list(evecs)),
                                key=lambda pair: pair[0], reverse=True)],dtype=float)
    evals = np.array(sorted(list(evals),reverse=True),dtype=float)
    if norm:
        row_sums = evecs.sum(axis=1)
        evecs = evecs / row_sums[:, np.newaxis] 
    return evals, evecs

def construct_transition_matrix(K, tau_lag):
    r""" Return column-stochastic transition matrix :math:`\textbf{T} = \rm{exp}(\textbf{K}\tau)`
    where :math:`\tau` is the lag time `tau_lag`.
    """
    T = expm(tau_lag*K)
    for x in np.sum(T, axis=0):
        #assert( abs(x - 1.0) < 1.0E-10) 
        print(f'Transition matrix is not column-stochastic at' \
                f'tau={tau_lag}')
    return T

def get_timescales(K, m, tau_lag):
    """ Return characteristic timescales obtained from the `m` dominant 
    eigenvalues of the transition matrix constructed from `K` at lag time
    `tau_lag`."""
    T = construct_transition_matrix(K, tau_lag)
    evals, evecs = calc_eigenvectors(T, m, which_eig='LM')
    char_times = np.zeros((np.shape(evals)[0]),dtype=float)
    # note that we ignore the zero eigenvalue, associated with
    # infinite time (stationary distribution)
    for i, eigval in enumerate(evals[1:]):
        char_times[i+1] = -tau_lag/np.log(eigval)
    return char_times

def calculate_spectral_error(m, Rs, labels):
    """ Calculate spectral error, where `m` is the number of dominant
    eigenvalues in both the reduced and original transition networks, as a
    function of lag time for various coarse grained rate-matrices
    specified in the array `Rs`. Plots the decay. """

    tau_lags = np.logspace(-4, 4, 1000)
    colors = sns.color_palette("BrBG", 5)
    colors = [colors[0], colors[-1]]
    fig, ax = plt.subplots()

    for j,R in enumerate(Rs):
        spectral_errors = np.zeros(tau_lags.shape)
        for i, tau in enumerate(tau_lags):
            T = construct_transition_matrix(R, tau)
            Tevals, Tevecs = calc_eigenvectors(T, m+1, which_eig='LM')
            # compare m+1 th eigenvalue (first fast eigenvalue) to slowest
            # eigenmodde
            spectral_errors[i] = Tevals[m]/Tevals[1]
        ax.plot(tau_lags, spectral_errors, label=labels[j], color=colors[j])

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\eta(\tau)$')
    plt.yscale('log')
    plt.legend()
    fig.tight_layout()

def check_detailed_balance(pi, K):
    """ Check if Markov chain satisfies detailed balance condition, 
    :math:`k_{ij} \pi_j = k_{ji} \pi_i` for all :math:`i,j`.

    Parameters
    ----------
    pi : array-like (nnodes,) or (ncomms,)
        stationary probabilities
    K : array-like (nnodes, nnodes) or (ncomms, ncomms)
        transition rate matrix

    """
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if i < j:
                left = K[i,j]*pi[j]
                right = K[j,i]*pi[i]
                diff = abs(left - right)
                if (diff > 1.E-10):
                    #print(f'Detailed balance not satisfied for i={i}, j={j}')
                    return False
    return True

def read_communities(commdat):
    """Read in a single column file called communities.dat where each line
    is the community ID (zero-indexed) of the minima given by the line
    number.
    
    Parameters
    ----------
    commdat : .dat file
        single-column file containing community IDs of each minimum

    Returns
    -------
    communities : dict
        mapping from community ID (1-indexed) to minima ID (1-indexed)
    """

    communities = {}
    with open(commdat, 'r') as f:
        for minID, line in enumerate(f, 1):
            groupID =  int(line) + 1
            if groupID in communities:
                communities[groupID].append(minID)
            else:
                communities[groupID] = [minID]
    return communities



