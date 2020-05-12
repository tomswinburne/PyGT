"""File Conversions

The KTN_python module includes python GT tools, KTN analysis tools, and
ways to talk to the external programs PATHSAMPLE and DISCOTRESS. 
To unify these four software packages, this script converts between 
different file formats for representing KTNs.

Community Files
---------------
communities.dat : single-column (nnnodes,)
    used by DISCOTRESS and ktn_tools; each line contains community ID (0-indexed)
    of the node specified by the line number in the file
minima.groups : output of DUMPGROUPS keyword from REGROUPFREE routine in PATHSAMPLE
    used by ParsePathsample class in ktn package to write a communities.dat file. 
    So conversion is absent from this file.

Community data structures:
--------------------------
ktn.communities : dictionary mapping community IDs (1-indexed) to minima IDs (1-indexed)
    file format used in ktn_analysis
pgt.communities : dictionary mapping community IDs (0-indexed) to boolean list (nnodes,)
    Boolean list selects out the nodes in that community; used in gt_tools

Rate matrix files
-----------------
rate_matrix.dat : (nnodes, nnodes) rate matrix entries printed in dense format
min.data, ts.data : input to PATHSAMPLE (unnecessary if rate_matrix.dat is present)
ts_weights.dat : single column, ln(kij), (2*nts, )
    used in DISCOTRESS, output of DUMPINFO, kij first line, kji second line, etc.
ts_conns.dat   : double columns, i <-> j, (nts, )
    used in DISCOTRESS, output of DUMPINFO

Rate matrix variables
---------------------
Q : in GT code, Q is actually -K and is typically in sparse format
ktn.K : in ktn code, K is the same as -Q and is in dense format

"""
import numpy as np
import scipy as sp
from pathlib import Path 


""" Rate matrix file conversion """

def K_from_Q(Q):
    """ Convert the sparse, gt_tools `Q` matrix to the dense, `K` matrix used 
    in ktn_analysis by multiplying nonzero entries by -1. This same function
    can also be used to get Q from K, although that's a less useful conversion."""
    rate_mat = Q
    if sp.sparse.issparse(rate_mat):
        rate_mat = rate_mat.todense()
    ix, iy = np.nonzero(rate_mat)
    for j in range(len(ix)):
        rate_mat[ix[j],iy[j]] *= -1
    return rate_mat

def dump_rate_mat(K, data_path, fmt='%.20G'):
    """ Dumps a rate_matrix.dat file for input to PATHSAMPLE. 

    Parameters
    ----------
    Q : (nnodes, nnodes)
        rate matrix in sparse or dense format
    data_path : str or Path
        path to folder with pathdata file where rate_matrix.dat will be dumped
    fmt : format string
        Defaults to '%.20G'

    """
    np.savetxt(Path(data_path)/'rate_matrix.dat', K, fmt=fmt)

def ts_weights_conns_from_K(K, data_path, suffix=''):
    """ Write a ts_weights.dat and ts_conns.dat file from a rate matrix.
    K can actually be K or Q since we only care about off-diagonal entries. Ensures
    the rates are positive before writing to file.

    TODO: test
    TODO: figure out how to get ts_weights.dat from sparse matrix
    """

    #if K is sparse (i.e. Q), can just get from data structure

    ts_conns = open(Path(data_path)/f'ts_conns{suffix}.dat', 'w')
    ts_weights = open(Path(data_path)/f'ts_weights{suffix}.dat', 'w')
    if sp.sparse.issparse(K):
        #tsweights = []
        for i in range(K.shape[0]):
            cols = K.indices[K.indptr[i]:K.indptr[i+1]]
            kijs = K.data[K.indptr[i]:K.indptr[i+1]]
            for j in cols:
                if i < j:
                    ts_conns.write(f'{i} {j}\n')
                #append k i <- j, later we'll sort it so j <- i comes right after
                #ts_weights.append(np.log(kijs[j]))
        #tsweights_sorted = np.array(tsweights)
    else:
        ix, iy = np.nonzero(K)
        for i in range(len(ix)):
            if ix[i] < iy[i]:
                #i --> j, where i < j
                ts_conns.write(f'{ix[i]} {iy[i]}\n')
                # Kji (i -> j)
                ts_weights.write(f'{np.log(K[iy[i], ix[i]])}\n')
                # Kij (j -> i)
                ts_weights.write(f'{np.log(K[ix[i], iy[i]])}\n')

def dump_stat_probs(pi, data_path, suffix='', fmt='%.20G'):
    """ Dump a stat_prob.dat file."""
    pi = pi/pi.sum()
    np.savetxt(Path(data_path)/'stat_prob{suffix}.dat', np.log(pi), fmt=fmt)

def read_ktn_info(data_path, suffix='', log=False):
    """ Read in Daniel's files stat_prob.dat, ts_weights.dat, and ts_conns.dat
    and return a rate matrix and vector of stationary probabilities."""

    logpi = np.loadtxt(data_path/f'stat_prob{suffix}.dat')
    pi = np.exp(logpi)
    nnodes = len(pi)
    assert(abs(1.0 - np.sum(pi)) < 1.E-10)
    logk = np.loadtxt(self.path/f'ts_weights{suffix}.dat', 'float')
    k = np.exp(logk)
    tsconns = np.loadtxt(self.path/f'ts_conns{suffix}.dat', 'int')
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

""" Converting between community data structures. """

def ktn_comms_from_gt_comms(gt_comms):
    """ Convert the GT-style dictionary of communities to a format
    compatible with the ktn_analysis class.

    Parameters
    ----------
    communities: dict
        community IDs (0-indexed) to boolean array selectors (nnodes,)

    Returns
    -------
    communities: dict
        community IDs (1-indexed) to minimia IDs (1-indexed) (nmin, )

    """
    ktn_comms = {}
    for ci in gt_comms:
        ktn_comms[ci+1] = np.array(gt_comms[ci].nonzero()[0]) + 1
    return ktn_comms

def write_AB_communities_from_gt(gt_comms, A, B, data_path, suffix=''):
    """ Write a communities.dat file containing just 2 communities;
    all nodes in B U I are assigned to community 0, and all nodes in
    absorbing A assigned to community 1. Useful for simulating A<-B
    transition paths with DISCOTRESS."""

    AS = gt_comms[A]
    BS = gt_comms[B]
    IS = ~(AS+BS)
    AB_comms = {}
    AB_comms[0] = BS+IS
    AB_comms[1] = AS
    write_gt_comms(AB_comms, data_path, suffix=suffix)

def write_gt_comms(gt_comms, data_path, suffix=''):
    """ Write a communities.dat file from GT-style communities dictionary."""
    if len(gt_comms) < 1:
        raise ValueError('gt_comms is empty.')
    #all community selector arrays have same length
    nnodes = len(gt_comms[0])
    #file to write: each line has commID of that node
    commIDs = np.zeros(nnodes, int)
    for ci in gt_comms:
        min_in_ci = np.array(gt_comms[ci].nonzero()[0])
        commIDs[min_in_ci] = ci
    np.savetxt(Path(data_path)/f'communities{suffix}.dat', commIDs, fmt='%d')

def write_ktn_comms(ktn_comms, data_path, suffix=''):
    """ Write a single-column file `commdat` where each line is the
    community ID (zero-indexed) of the minima given by the line
    number.

    Parameters
    ----------
    communities : dict
        mapping from community ID (1-indexed) to minima ID (1-indexed)
    commdat : .dat file name
        file to which to write communitiy assignments (0-indexed)

    """
    commdat = Path(data_path)/f'communities{suffix}.dat'
    if len(ktn_comms) < 1:
        raise ValueError('ktn_comms is empty.')
    #count number of nodes total
    nnodes = 0
    for ci in ktn_comms:
        nnodes += len(ktn_comms[ci])
    #file to write: each line has commID of that node
    commIDs = np.zeros(nnodes, int)
    for ci in ktn_comms:
        min_in_ci = np.array(ktn_comms[ci]) - 1
        commIDs[min_in_ci] = ci - 1
    np.savetxt(commdat, commIDs, fmt='%d')



