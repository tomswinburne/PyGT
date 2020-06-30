""" This module provides functions to analyze and estimate coarse-grained continuous-time
Markov chains given a partiion of the :math:`V` nodes into :math:`N<V` communities.
Various formulations for the inter-community rates are implemented, including the local
equilibrium approximation, Hummer-Szabo relation, and other expressions. 

The module is
designed to interface with both the Python and Fortran implementations of the GT algorithm.

Deepti Kannan 2020 """

from .code_wrapper import ParsedPathsample
from .code_wrapper import ScanPathsample

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

PATHSAMPLE = "/home/dk588/svn/PATHSAMPLE/build/gfortran/PATHSAMPLE"

"""Functions to analyze KTNs without any community structure."""

def read_ktn_info(self, suffix, log=False):
    #read in Daniel's files stat_prob.dat and ts_weights.dat
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
    # calculate k dominant eigenvectors and eigenvalues of sparse matrix
    # using the implictly restarted Arnoldi method
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
    """ Return column-stochastic transition matrix T = expm(K*tau).
    Columns sum to 1. """
    T = expm(tau_lag*K)
    for x in np.sum(T, axis=0):
        #assert( abs(x - 1.0) < 1.0E-10) 
        print(f'Transition matrix is not column-stochastic at' \
                f'tau={tau_lag}')
    return T

def get_timescales(K, m, tau_lag):
    """ Return characteristic timescales obtained from the m dominant 
    eigenvalues of the transition matrix constructed from K at lag time
    tau_lag."""
    T = construct_transition_matrix(K, tau_lag)
    evals, evecs = calc_eigenvectors(T, m, which_eig='LM')
    char_times = np.zeros((np.shape(evals)[0]),dtype=float)
    # note that we ignore the zero eigenvalue, associated with
    # infinite time (stationary distribution)
    for i, eigval in enumerate(evals[1:]):
        char_times[i+1] = -tau_lag/np.log(eigval)
    return char_times

def calculate_spectral_error(m, Rs, labels):
    """ Calculate spectral error, where m is the number of dominant
    eigenvalues in both the reduced and original transition networks, as a
    function of lag time. Plots the decay. """

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
    """ Check if network satisfies detailed balance condition, which is
    thatthat :math:`k_{ij} \pi_j = k_{ji} \pi_i` for all :math:`i,j`.

    Parameters
    ----------
    pi : list (nnodes,) or (ncomms,)
        stationary probabilities
    K : np.ndarray (nnodes, nnodes) or (ncomms, ncomms)
        inter-minima rate constants in matrix form

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

class Analyze_KTN(object):
    """ Analyze a KTN with a specified community structure."""

    def __init__(self, path, communities=None,
                 commdata=None, temp=None, thresh=None, pathsample=None):
        self.path = Path(path) #path to directory with all relevant files
        self.K = None
        self.pi = None
        self.commpi = None
        if temp is not None:
            logpi, Kmat = read_ktn_info(f'T{temp:.3f}', log=True)
            pi = np.exp(logpi)
            self.K = Kmat
            self.pi = pi/pi.sum()
        if communities is not None:
            self.communities = communities
        elif commdata is not None:
            self.communities = read_communities(self.path/commdata)
        else:
            if thresh is not None and temp is not None:
                commdata=f'communities_G{thresh:.2f}_T{temp:.3f}.dat'
                self.communities = read_communities(self.path/commdata)
                commpi = self.get_comm_stat_probs(logpi, log=False)
                self.commpi = commpi/commpi.sum()
            else:
                raise AttributeError('Either communities or commdata must' \
                                    'be specified.')
        #for analyzing rate matrices generated from PATHSAMPLE
        if pathsample is not None:
            self.parse = pathsample

    def calc_inter_community_rates_NGT(self, C1, C2):
        """Calculate k_{C1<-C2} using NGT. Here, C1 and C2 are community IDs
        (i.e. groups identified in DUMPGROUPS file from REGROUPFREE). This
        function isolates the minima in C1 union C2 and the transition states
        that connect them and feeds this subnetwork into PATHSAMPLE, using the
        NGT keyword to calculate inter-community rates."""

        #minima to isolate
        mintoisolate = self.communities[C1] + self.communities[C2]
        #parse min.data and write a new min.data file with isolated minima
        #also keep track of the new minIDs based on line numbers in new file
        newmin = {}
        j = 1
        with open(self.path/f'min.data.{C1}.{C2}', 'w') as newmindata:
            with open(self.path/'min.data','r') as ogmindata:
                #read min.data and check if line number is in C1 U C2
                for i, line in enumerate(ogmindata, 1):
                    if i in mintoisolate:
                        #save mapping from old minIDs to new minIDs
                        newmin[i] = j
                        #NOTE: these min have new line numbers now
                        #so will have to re-number min.A,min.B,ts.data
                        newmindata.write(line)
                        j += 1
                    
        #exclude transition states in ts.data that connect minima not in C1/2
        ogtsdata = pd.read_csv(self.path/'ts.data', sep='\s+', header=None,
                               names=['nrg','fvibts','pointgroup','min1','min2','itx','ity','itz'])
        newtsdata = []
        noconnections = True #flag for whether C1 and C2 are disconnected
        for ind, row in ogtsdata.iterrows():
            min1 = int(row['min1'])
            min2 = int(row['min2'])
            if min1 in mintoisolate and min2 in mintoisolate:
                # turn off noconnections flag as soon as one TS between C1 and
                # C2 is found
                if ((min1 in self.communities[C1] and min2 in self.communities[C2]) or
                (min1 in self.communities[C2] and min2 in self.communities[C1])):
                    noconnections = False
                #copy line to new ts.data file, renumber min
                modifiedrow = pd.DataFrame(row).transpose()
                modifiedrow['min1'] = newmin[min1]
                modifiedrow['min2'] = newmin[min2]
                modifiedrow['pointgroup'] = int(modifiedrow['pointgroup'])
                newtsdata.append(modifiedrow)
        if noconnections or len(newtsdata)==0:
            #no transition states between these minima, return 0
            print(f"No transition states exist between communities {C1} and {C2}")
            return 0.0, 0.0
        newtsdata = pd.concat(newtsdata)
        #write new ts.data file
        newtsdata.to_csv(self.path/f'ts.data.{C1}.{C2}',header=False, index=False, sep=' ')
        #write new min.A/min.B files with nodes in C1 and C2 (using new
        #minIDs of course)
        numInC1 = len(self.communities[C1])
        minInC1 = []
        for min in self.communities[C1]:
            minInC1.append(newmin[min] - 1)
        numInC2 = len(self.communities[C2])
        minInC2 = []
        for j in self.communities[C2]:
            minInC2.append(newmin[j] - 1)
        self.minA = minInC1
        self.minB = minInC2
        self.numInA = numInC1
        self.numInB = numInC2
        self.write_minA_minB(self.path/f'min.A.{C1}', self.path/f'min.B.{C2}')
        #run PATHSAMPLE
        files_to_modify = [self.path/'min.A', self.path/'min.B',
                           self.path/'min.data', self.path/'ts.data']
        for f in files_to_modify:
            os.system(f'mv {f} {f}.old')
        os.system(f"cp {self.path}/min.A.{C1} {self.path}/min.A")
        os.system(f"cp {self.path}/min.B.{C2} {self.path}/min.B")
        os.system(f"cp {self.path}/min.data.{C1}.{C2} {self.path}/min.data")
        os.system(f"cp {self.path}/ts.data.{C1}.{C2} {self.path}/ts.data")
        outfile = open(self.path/f'out.{C1}.{C2}.T{temp}','w')
        subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
        #parse output
        self.parse_output(outfile=self.path/f'out.{C1}.{C2}.T{temp}')
        for f in files_to_modify:
            os.system(f'mv {f}.old {f}')
        #return rates k(C1<-C2), k(C2<-C1)
        return self.output['kAB'], self.output['kBA']


    def construct_coarse_rate_matrix_NGT(self):
        """ Calculate inter-community rate constants using communities defined
        by minima_groups file at specified temperature. Returns a NxN rate
        matrimatrix where N is the number of communities."""

        N = len(self.communities.keys())
        print(N)
        R = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i < j:
                    try:
                        Rij, Rji = self.calc_inter_community_rates_NGT(i+1, j+1)
                    except:
                        print(f'PATHSAMPLE errored out for communities {i} and {j}')
                        continue
                    R[i, j] = Rij
                    R[j, i] = Rji
        for i in range(N):
            R[i, i] = -np.sum(R[:, i])
        return R

    def construct_coarse_rate_matrix_LEA(self, temp):
        """Calculate the coarse-grained rate matrix obtained using the local
        equilibrium approximation (LEA)."""

        if self.K is None:
            logpi, Kmat = read_ktn_info(f'T{temp:.3f}', log=True)
            pi = np.exp(logpi)
            self.K = Kmat
            self.pi = pi/pi.sum()
            commpi = self.get_comm_stat_probs(logpi, log=False)
            self.commpi = commpi/commpi.sum()
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

    def construct_coarse_matrix_Hummer_Szabo(self, temp):
        """ Calculate the coarse-grained rate matrix using the Hummer-Szabo
        relation, aka eqn. (12) in Hummer & Szabo (2015) J.Phys.Chem.B."""

        if self.K is None:
            logpi, Kmat = read_ktn_info(f'T{temp:.3f}', log=True)
            pi = np.exp(logpi)
            self.K = Kmat
            self.pi = pi/pi.sum()
            commpi = self.get_comm_stat_probs(logpi, log=False)
            self.commpi = commpi/commpi.sum()
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
            print(f'HS does not satisfy detailed balance at T={temp}')
        return R_HS

    def hummer_szabo_from_mfpt(self, temp, GT=True, mfpt=None):
        """Calculate Hummer-Szabo coarse-grained rate matrix using Eqn. (72)
        of Kells et al. (2019) paper on correlation functions and the Kemeny
        constant."""
        if self.K is None:
            logpi, Kmat = read_ktn_info(f'T{temp:.3f}', log=True)
            pi = np.exp(logpi)
            self.K = Kmat
            self.pi = pi/pi.sum()
            commpi = self.get_comm_stat_probs(logpi, log=False)
            self.commpi = commpi/commpi.sum()
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
            mfpt = self.get_MFPT_between_states_GT(temp)
        elif mfpt is None:
            mfpt = self.get_MFPT_from_Kmat(self.K)
        R = Pi_col@np.ones((1,N)) - D_N@spla.inv(Pi_col@Pi_col.T +
                                                 M.T@D_n@mfpt@pi_col@Pi_col.T -
                                                 M.T@D_n@mfpt@D_n@M)
        if not check_detailed_balance(self.commpi, R):
            print(f'KRA does not satisfy detailed balance at T={temp}')
        return R

    def get_MFPT_from_Kmat(self, K):
        """Use Tom's linear solver method to extract MFPTs from a rate matrix
        K. Equivalent to Eqn. (14) in Swinburne & Wales (2020)."""

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

    def mfpt_from_correlation(self, K, pi):
        """Calculate the matrix of mean first passage times using Eq. (49) of KRA
        JCP paper. """
        nmin = K.shape[0]
        pioneK = spla.inv(pi.reshape((nmin,1))@np.ones((1,nmin)) + K)
        zvec = np.diag(pioneK)
        mfpt = np.diag(1./pi)@(pioneK - zvec.reshape((nmin,1))@np.ones((1,nmin)))
        return mfpt

    def get_MFPT_between_communities(self, K, pi):
        """Use Tom's linear solver method to extract MFPTs between communities
        using a rate matrix K. Equivalent to Eqn. (14) in Swinbourne & Wales
        2020."""
       
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

    def get_MFPT_AB(self, A, B, temp, n):
        """Run a single GT calculation using the READRATES keyword in PATHSAMPLE
        to get the A<->B mean first passage time at specified temperatire."""

        parse = ParsedPathsample(self.path/'pathdata')
        files_to_modify = [self.path/'min.A', self.path/'min.B']
        for f in files_to_modify:
            if not f.exists():
                print(f'File {f} does not exists')
                raise FileNotFoundError 
            os.system(f'mv {f} {f}.original')
        communities = self.communities
        #update min.A and min.B with nodes in A and B
        parse.minA = np.array(communities[A+1]) - 1
        parse.numInA = len(communities[A+1])
        parse.minB = np.array(communities[B+1]) - 1
        parse.numInB = len(communities[B+1])
        parse.write_minA_minB(self.path/'min.A', self.path/'min.B')
        parse.append_input('NGT', '0 T')
        parse.append_input('TEMPERATURE', f'{temp}')
        parse.append_input('READRATES', f'{n}')
        parse.write_input(self.path/'pathdata')
        #run PATHSAMPLE
        outfile = open(self.path/f'out.{A+1}.{B+1}.T{temp}', 'w')
        subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
        #parse output
        parse.parse_output(outfile=self.path/f'out.{A+1}.{B+1}.T{temp}')
        return parse.output['MFPTAB'], parse.output['MFPTBA']

    def get_MFPT_between_communities_GT(self, temp):
        """Use PATHSAMPLE to compute MFPTs between communities
        using a rate matrix K."""

        parse = ParsedPathsample(self.path/'pathdata')
        files_to_modify = [self.path/'min.A', self.path/'min.B']
        for f in files_to_modify:
            if not f.exists():
                print(f'File {f} does not exists')
                raise FileNotFoundError 
            os.system(f'mv {f} {f}.original')
        communities = self.communities
        N = len(communities)
        MFPT = np.zeros((N,N))
        for ci in range(N):
            for cj in range(N):
                if ci < cj:
                    parse.minA = np.array(communities[ci+1]) - 1
                    parse.numInA = len(communities[ci+1])
                    parse.minB = np.array(communities[cj+1]) - 1
                    parse.numInB = len(communities[cj+1])
                    parse.write_minA_minB(self.path/'min.A', self.path/'min.B')
                    #os.system(f'cat {self.path}/min.A')
                    #os.system(f'cat {self.path}/min.B')
                    parse.append_input('NGT', '0 T')
                    parse.append_input('TEMPERATURE', f'{temp}')
                    parse.write_input(self.path/'pathdata')
                    #run PATHSAMPLE
                    outfile = open(self.path/f'out.{ci+1}.{cj+1}.T{temp}', 'w')
                    subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
                    #parse output
                    parse.parse_output(outfile=self.path/f'out.{ci+1}.{cj+1}.T{temp}')
                    MFPT[ci, cj] = parse.output['MFPTAB']
                    MFPT[cj, ci] = parse.output['MFPTBA']

        #restore original min.A and min.B files
        for f in files_to_modify:
            os.system(f'mv {f}.original {f}')

        return MFPT

    def get_MFPT_between_states_GT(self, temp):
        """Use PATHSAMPLE to compute MFPTs between states
        using a rate matrix K."""

        parse = ParsedPathsample(self.path/'pathdata')
        files_to_modify = [self.path/'min.A', self.path/'min.B']
        for f in files_to_modify:
            if not f.exists():
                print(f'File {f} does not exists')
                raise FileNotFoundError 
            os.system(f'mv {f} {f}.original')
        
        parse.append_input('NGT', '0 T')
        parse.comment_input('WAITPDF')
        parse.append_input('TEMPERATURE', f'{temp}')
        parse.write_input(self.path/'pathdata')
        n = len(self.pi)
        mfpt = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    parse.minA = [i] 
                    parse.numInA = 1
                    parse.minB = [j]
                    parse.numInB = 1
                    parse.write_minA_minB(self.path/'min.A', self.path/'min.B')
                    #run PATHSAMPLE
                    outfile_name = self.path/f'out.{i+1}.{j+1}.T{temp}'
                    outfile = open(outfile_name, 'w')
                    subprocess.run(f"{PATHSAMPLE}", stderr=subprocess.STDOUT, stdout=outfile, cwd=self.path)
                    #parse output
                    parse.parse_output(outfile=self.path/f'out.{i+1}.{j+1}.T{temp}')
                    mfpt[i, j] = parse.output['MFPTAB']
                    mfpt[j, i] = parse.output['MFPTBA']
                    Path(outfile_name).unlink()
                    print(f'Calculated MFPT for states ({i}, {j})')

        #restore original min.A and min.B files
        for f in files_to_modify:
            os.system(f'mv {f}.original {f}')

        return mfpt

    def get_kells_cluster_passage_times(self, pi, commpi, mfpt):
        """Comppute the t_JI as defined in Eqn. 66 in Kells, Rosta,
        Annibale (2019)."""

        N = len(self.communities)
        tJI = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                ci = np.array(self.communities[i+1]) - 1
                cj = np.array(self.communities[j+1]) - 1
                tJI[j][i] = pi[cj]@ mfpt[cj,:][:,ci] @ pi[ci] / (commpi[i]*commpi[j])
                tJI[j][i] -= pi[cj] @ mfpt[cj,:][:,cj] @ pi[cj] / (commpi[j])**2

        return tJI

    def get_approx_kells_cluster_passage_times(self, pi, commpi, mfpt):
        """Compute an approximation to t_JI as defined in Eqn. 66 in Kells, Rosta,
        Annibale (2019) in which we assume that all mfpt's tji from any state i
        in I to any state j in J are the same. In this approximation, whose
        accuracy depends on the degree of metastability, we can approximate
        t_JI by a single mfpt between i* and j* where i* and j* are the local minima of I
        and J respectively.
        NOTE: problem with local min is that they may have very little
        occupation probability, in which case it doesnt make sense to choose
        it.
        Other possibility: keep choosing (i,j) pairs and montecarlo sample them
        and take an average or something?
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

    def construct_coarse_rate_matrix_from_MFPTs(self, temp, GT=False):
        """ Calculate a rate matrix using  MFPTs between communities."""
        if self.K is None:
            logpi, Kmat = read_ktn_info(f'T{temp:.2f}', log=True)
            pi = np.exp(logpi)
            self.K = Kmat
            self.pi = pi/pi.sum()
            commpi = self.get_comm_stat_probs(logpi, log=False)
            self.commpi = commpi/commpi.sum()
        # get R from matrix of MFPTs
        N = len(self.communities)
        D_N = np.diag(self.commpi)
        if GT:
            mfpt = self.get_MFPT_between_communities_GT()
        else:
            mfpt = self.get_MFPT_between_communities(self.K, self.pi)
        
        matrix_of_ones = np.ones((N,1))@np.ones((1,N))
        #R_MFPT = inv(MFPT)@(inv(D_N) - matrix_of_ones)
        R_MFPT = spla.solve(mfpt,np.diag(1.0/self.commpi) - matrix_of_ones)
        check_detailed_balance(self.commpi, R_MFPT)
        return R_MFPT

    def get_free_energies_from_rates(self, R, thresh, temp, kB=1.0, planck=1.0):
        """ Estimate free energies of all super states and transition states
        between super states from an arbitrary rate matrix R. """

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
        """ Calculate the ith eigenvector approximation error for i in {1, 2,
        ... m} of a coarse-grained rate matrix R by comparing its eigenvector 
        to the correspcorresponding eigenvector of the full matrix.
        TODO: test on small system for which we are confident in eigenvectors.
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
        thethe stationary probabilities of the nodes in each community.
        
        Parameters
        ----------
        pi : list (nnodes,)
            stationary probabilities of all minima in full network

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

 
"""Functions that use the Analyze_KTN class to perform useful tasks."""

def compare_HS_LEA(temps, nrgthreshs):
    """ Calculate coarse-grained rate matrices using the Hummer-Szabo, NGT, and LEA
    methods and compute kAB/kBA using NGT to be compared to the rates on the
    fulfull network. """

    #temps = [0.7, 0.8, 0.9, 1.0, 1.2, 1.33, 1.5,
    #         2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    #         10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]
    #nrgthreshs = [1, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 100]
    bigdfs = []
    for temp in temps:
        dfs = []
        for thresh in nrgthreshs:
            df = pd.DataFrame()
            df['T'] = [temp]
            df['Gthresh'] = [thresh]
            ktn = Analyze_KTN('/scratch/dk588/databases/LJ38.2010/4000.minima',
                              thresh=thresh, temp=temp)
            labels = []
            matrices = []
            try:
                Rhs_mfpt = ktn.hummer_szabo_from_mfpt(temp, GT=False,
                                                      mfpt=np.load(f'csvs/mfpt_GT_LJ38_900minima_T{temp}.npy'))
                matrices.append(Rhs_mfpt)
                labels.append('KRA')
            except Exception as e:
                print(f'Hummer Szabo Kells had the following error: {e}')
            """
            try:
                Rhs = ktn.construct_coarse_matrix_Hummer_Szabo(temp)
                matrices.append(Rhs)
                labels.append('HS')
            except Exception as e:
                print(f'Hummer Szabo had the following error: {e}')
            """
            try:
                Rlea = ktn.construct_coarse_rate_matrix_LEA(temp)
                matrices.append(Rlea)
                labels.append('LEA')
            except Exception as e:
                print(f'LEA had the following error: {e}')

            if len(matrices)==0:
                continue

            for i, R in enumerate(matrices):
                """ get A->B and B->A mfpt on coarse network"""
                ABparse = ParsedPathsample(ktn.path/'pathdata')
                ABparse.parse_minA_and_minB(ktn.path/f'G{thresh:.1f}/min.A.regrouped.{temp:.10f}.G{thresh:.1f}',
                                ktn.path/f'G{thresh:.1f}/min.B.regrouped.{temp:.10f}.G{thresh:.1f}')

                #first do A<-B
                N = len(ktn.communities)
                pi = ktn.commpi
                cond = np.ones((N,), dtype=bool)
                cond[ABparse.minA] = False
                initial_cond = np.zeros((N,))
                #initialize probability density to local boltzman in B
                initial_cond[ABparse.minB] = pi[ABparse.minB]/pi[ABparse.minB].sum()
                mfptAB = -spla.solve(R[cond, :][:, cond],
                                        initial_cond[cond]).sum()
                #then do B<-A
                cond = np.ones((N,), dtype=bool)
                cond[ABparse.minB] = False
                initial_cond = np.zeros((N,))
                #initialize probability density to local boltzman in A
                initial_cond[ABparse.minA] = pi[ABparse.minA]/pi[ABparse.minA].sum()
                mfptBA = -spla.solve(R[cond, :][:, cond],
                                        initial_cond[cond]).sum()

                df[f'MFPTAB_{labels[i]}'] = [mfptAB]
                df[f'MFPTBA_{labels[i]}'] = [mfptBA]
                """
                try:
                    ktn.get_free_energies_from_rates(R, thresh, temp)
                except ValueError as e:
                    print(f'{labels[i]}: {e} at T={temp}, G={thresh}')
                    continue
                #run NGT on regrouped minima using HS rates instead of LEA
                og_input_files = [ktn.path/'min.A', ktn.path/'min.B',
                            ktn.path/'min.data', ktn.path/'ts.data']
                new_input_files = [ktn.path/f'G{thresh:.1f}/min.A.regrouped.{temp:.10f}.G{thresh:.1f}',
                                ktn.path/f'G{thresh:.1f}/min.B.regrouped.{temp:.10f}.G{thresh:.1f}',
                                ktn.path/f'min.data.T{temp:.3f}',
                                ktn.path/f'ts.data.T{temp:.3f}']
                for j, f in enumerate(og_input_files):
                    os.system(f'mv {f} {f}.original')
                    os.system(f'cp {new_input_files[j]} {f}')
                #run PATHSAMPLE
                parse = ParsedPathsample(ktn.path/'pathdata')
                parse.append_input('TEMPERATURE', temp)
                parse.comment_input('REGROUPFREE')
                parse.comment_input('DUMPGROUPS')
                #parse.comment_input('WAITPDF')
                parse.append_input('NGT', '0 T')
                parse.write_input(ktn.path/'pathdata')
                outfile = open(parse.path/f'out.{thresh:.1f}.T{temp:.1f}.ngt','w')
                subprocess.run(f"{PATHSAMPLE}", stderr=subprocess.STDOUT, stdout=outfile, cwd=ktn.path)
                #parse output
                parse.parse_output(outfile=parse.path/f'out.{thresh:.1f}.T{temp:.1f}.ngt')
                #return rates k(C1<-C2), k(C2<-C1)
                df[f'kAB_{labels[i]}'] = [parse.output['kAB']]
                df[f'MFPTAB_{labels[i]}'] = [parse.output['MFPTAB']]
                df[f'kBA_{labels[i]}'] = [parse.output['kBA']]
                df[f'MFPTBA_{labels[i]}'] = [parse.output['MFPTBA']]
                #now run waitpdf on the coarse network
                parse.comment_input('NGT')
                parse.append_input('WAITPDF', '')
                parse.write_input(ktn.path/'pathdata')
                outfile = open(parse.path/f'out.{thresh:.1f}.T{temp:.1f}.waitpdf','w')
                try:
                    subprocess.run(f"{PATHSAMPLE}", stderr=subprocess.STDOUT,
                                   stdout=outfile, cwd=ktn.path, timeout=5)
                except subprocess.TimeoutExpired:
                    print('WAITPDF expired 5s timeout. Setting tau*AB and' +
                          ' tau*BA to NaN')
                #parse output
                parse.parse_output(outfile=parse.path/f'out.{thresh:.1f}.T{temp:.1f}.waitpdf')
                if 'tau*AB' in parse.output:
                    #df[f'k*AB_{labels[i]}'] = [1.0/parse.output['tau*AB']]
                    df[f'tau*AB_{labels[i]}'] = [parse.output['tau*AB']]
                else:
                    #df[f'k*AB_{labels[i]}'] = np.nan
                    df[f'tau*AB_{labels[i]}'] = np.nan
                if 'tau*BA' in parse.output:
                    #df[f'k*BA_{labels[i]}'] = [1.0/parse.output['tau*BA']]
                    df[f'tau*BA_{labels[i]}'] = [parse.output['tau*BA']]
                else:
                    #df[f'k*BA_{labels[i]}'] = np.nan
                    df[f'tau*BA_{labels[i]}'] = np.nan
                #return original min.data / ts.data files
                for j, f in enumerate(og_input_files):
                    os.system(f'mv {f}.original {f}')
                """
            dfs.append(df)
        bigdf = pd.concat(dfs, ignore_index=True, sort=False)
        scan = ScanPathsample(ktn.path/'pathdata', suffix='scan_MFPT_exact')
        #scan.parse.comment_input('WAITPDF')
        rates = scan.run_NGT_exact(temp)
        bigdf['MFPTexactAB'] = rates['MFPTAB']
        bigdf['MFPTexactBA'] = rates['MFPTBA']
        #bigdf['kABexact'] = rates['kAB']
        #bigdf['kBAexact'] = rates['kBA']
        bigdfs.append(bigdf)
    biggerdf = pd.concat(bigdfs, ignore_index=True)
    #if file exists, append to existing data
    csv = Path('csvs/rates_LEA_HSK_LJ38_MFPT.csv')
    #if csv.is_file():
    #    olddf = pd.read_csv(csv)
    #    bigdf = olddf.append(bigdf)
    #write updated file to csv
    biggerdf.to_csv(csv, index=False)

def calc_mfpt_AB_tom(temps):
    """Calculate the mean first passage time A<-B and B<-A on the full network
    for each of the temperatures specified using Tom's absorbing Markov chain
    method."""
    
    data = np.zeros((len(temps), 3))
    data[:,0] = temps
    n=8
    for i, temp in enumerate(temps):
        #free energy threshold doesn't matter since we only need K, pi
        ktn = Analyze_KTN('/scratch/dk588/databases/modelA',
                        commdata=f'communities_G1.00_T{temp:.2f}.dat')
        logpi, Kmat = read_ktn_info(f'T{temp:.2f}', log=True)
        pi = np.exp(logpi)
        pi = pi/pi.sum()
        minA = np.array([1, 7]) - 1
        minB = np.array([6, 8]) - 1
        #first do A<-B
        cond = np.ones((n,), dtype=bool)
        cond[minA] = False
        initial_cond = np.zeros((n,))
        #initialize probability density to local boltzman in B
        initial_cond[minB] = pi[minB]/pi[minB].sum()
        mfptAB = -spla.solve(Kmat[cond, :][:, cond],
                                initial_cond[cond]).sum()
        #then do B<-A
        cond = np.ones((n,), dtype=bool)
        cond[minB] = False
        initial_cond = np.zeros((n,))
        #initialize probability density to local boltzman in A
        initial_cond[minA] = pi[minA]/pi[minA].sum()
        mfptBA = -spla.solve(Kmat[cond, :][:, cond],
                                initial_cond[cond]).sum()
        data[i,1] = mfptAB
        data[i,2] = mfptBA
    df = pd.DataFrame(data, columns=['T', 'MFPTAB', 'MFPTBA'])
    return df


def compute_HS_matrices(temp):

    ktn = Analyze_KTN('/scratch/dk588/databases/modelA',
                      commdata=f'communities_G1.00_T{temp:.2f}.dat')

    logpi, Kmat = read_ktn_info(f'T{temp:.2f}')
    Rhs = ktn.construct_coarse_matrix_Hummer_Szabo(temp)
    Rhs_mfpt = ktn.hummer_szabo_from_mfpt(temp)
    return

if __name__ == '__main__':
    #df = pd.read_csv('csvs/rates_LEA_HS_HSK_modelA_MFPT_waitpdf333.csv')
    #temps = np.unique(df['T'])
    #df2 = calc_mfpt_AB_tom(temps)
    #temps = [0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.110, 0.115, 0.120,
    #         0.125, 0.130, 0.135, 0.140, 0.145, 0.150, 0.160, 0.165, 0.170,
    #         0.175, 0.180, 0.185, 0.190, 0.195, 0.2, 0.25, 0.3, 0.35, 0.4]
    #start from highest to lowest
    #temps = [0.4, 0.3, 0.2, 0.1, 0.05]
    #temps = temps[::-1]
    #nrgthreshs = [5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 100]
    #nrgthreshs = nrgthreshs[::-1]
    #compare_HS_LEA(temps, nrgthreshs)
    #temp = 100
    #ktn = Analyze_KTN('/scratch/dk588/databases/LJ38.2010/4000.minima',
    #                  thresh=100.0, temp=0.4)
    #Kmat = ktn.K
    #pi = ktn.pi
    #commpi = ktn.commpi
    #mfpt_comms_GT = ktn.get_MFPT_between_communities_GT(temp)
    #mfpt_comms_tom = ktn.get_MFPT_between_communities(Kmat, pi)
    #Rhs = ktn.construct_coarse_matrix_Hummer_Szabo(temp)
    #Rhs_mfpt = ktn.hummer_szabo_from_mfpt(temp)
    #Rhs_mfpt_gt = ktn.hummer_szabo_from_mfpt(temp, GT=True)
    
    temps = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 
           20.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 
           1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
