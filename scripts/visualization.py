""" Script to visualize escape time and first passage time distributions
in full and graph-transformed networks.

Plots figures in the partial GT manuscript.


Deepti Kannan, 2020"""

import numpy as np
from lib import partialGT as pgt
from ktn.ktn_analysis import *
import lib.ktn_io as kio
import lib.gt_tools as gt
import lib.conversion as convert
import scipy as sp
from scipy.sparse import save_npz,load_npz, diags, eye, csr_matrix, bmat
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import seaborn as sns
from pathlib import Path
import sys
from scipy import interpolate
from copy import deepcopy

sns.set()
textwidth = 6.47699
pres_params = {'axes.edgecolor': 'black',
                  'axes.facecolor':'white',
                  'axes.grid': False,
                  'axes.linewidth': 0.5,
                  'backend': 'ps',
                  'savefig.format': 'pdf',
                  'axes.titlesize': 24,
                  'axes.labelsize': 20,
                  'legend.fontsize': 20,
                  'xtick.labelsize': 18,
                  'ytick.labelsize': 18,
                  'text.usetex': True,
                  'figure.figsize': [7, 5],
                  'font.family': 'sans-serif',
                  #'mathtext.fontset': 'cm',
                  'xtick.bottom':True,
                  'xtick.top': False,
                  'xtick.direction': 'out',
                  'xtick.major.pad': 3,
                  'xtick.major.size': 3,
                  'xtick.minor.bottom': False,
                  'xtick.major.width': 0.2,

                  'ytick.left':True,
                  'ytick.right':False,
                  'ytick.direction':'out',
                  'ytick.major.pad': 3,
                  'ytick.major.size': 3,
                  'ytick.major.width': 0.2,
                  'ytick.minor.right':False,
                  'lines.linewidth':2}

params = {'axes.edgecolor': 'black',
                  'axes.facecolor':'white',
                  'axes.grid': False,
                  'axes.linewidth': 0.5,
                  'backend': 'ps',
                  'savefig.format': 'ps',
                  'axes.titlesize': 11,
                  'axes.labelsize': 9,
                  'legend.fontsize': 9,
                  'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'text.usetex': True,
                  'figure.figsize': [7, 5],
                  'font.family': 'sans-serif',
                  #'mathtext.fontset': 'cm',
                  'xtick.bottom':True,
                  'xtick.top': False,
                  'xtick.direction': 'out',
                  'xtick.major.pad': 3,
                  'xtick.major.size': 3,
                  'xtick.minor.bottom': False,
                  'xtick.major.width': 0.2,

                  'ytick.left':True,
                  'ytick.right':False,
                  'ytick.direction':'out',
                  'ytick.major.pad': 3,
                  'ytick.major.size': 3,
                  'ytick.major.width': 0.2,
                  'ytick.minor.right':False,
                  'lines.linewidth':2}
plt.rcParams.update(pres_params)

''' read positions stored in file min.pos.dummy and ts.pos.dummy '''
def read_pos(posf):

    pos = []
    with open(posf,"r") as pf:
        for line in pf.readlines():
            pos.append([float(x) for x in line.split()])
    return np.array(pos,dtype=float)

''' read energy and connectivity data stored in min.data.dummy and ts.data.dummy '''
def read_data(dataf,sp_type):

    ens = []
    if sp_type==2: conns = []
    with open(dataf,"r") as df:
        for line in df.readlines():
            ens.append(float(line.split()[0]))
            if sp_type==2:
                conns.append([int(line.split()[3]),int(line.split()[4])])
    if sp_type==1:
        return np.array(ens,dtype=float)
    elif sp_type==2:
        return np.array(ens,dtype=float), np.array(conns,dtype=int)


''' make a 2D plot of the data '''
def plot_network_landscape(min_ens,min_pos,ts_conns,plot_nodes=True):

    xi, yi = np.linspace(-1.5,1.5,100), np.linspace(-1.5,1.5,100)
    xi, yi = np.meshgrid(xi,yi)
    rbfi = interpolate.Rbf(min_pos[:,0],min_pos[:,1],min_ens,function="gaussian")
    approx_ens = rbfi(xi,yi)
    fig, ax = plt.subplots(figsize=(textwidth/2, textwidth/2))
    ax.imshow(approx_ens,cmap="bwr",extent=[xi.min(),xi.max(),yi.min(),yi.max()], origin="lower")
    if plot_nodes:
        # plot vertices
        ax.scatter(min_pos[:,0],min_pos[:,1], c='k', s=3)
        for conn in ts_conns:
            pt1, pt2 = min_pos[conn[0]-1,:], min_pos[conn[1]-1,:]
            ax.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"k-", lw=0.5)
    # plot
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    plt.show()
    fig.tight_layout()
    plt.savefig('plots/9state_network_landscape.pdf')

def plot_network_communities(min_pos, ts_conns, communities=None, plot_edges=True):
    data_path = Path('KTN_data/9state')
    B, K, tau, N, u, s, Emin, index_sel = kio.load_ktn(path=data_path,beta=0.1,Emax=None,Nmax=None,screen=False)
    xi, yi = np.linspace(-1.5,1.5,100), np.linspace(-1.5,1.5,100)
    xi, yi = np.meshgrid(xi,yi)
    if communities is None:
        communities = np.loadtxt(data_path/'communities_bace9.dat', dtype=int)
    colors = sns.color_palette('Paired', 11)
    node_colors = []
    for i in range(len(communities)):
        node_colors.append(colors[communities[i]])
    fig, ax = plt.subplots(figsize=(textwidth/2, textwidth/2))
    #plot endpoints
    #plt.plot([min_pos[endpt1-1,0],min_pos[endpt2-1,0]],[min_pos[endpt1-1,1],min_pos[endpt2-1,1]], \
    #         "ro",markersize=10,zorder=10)
    # plot points pairwise such that connected points are joined
    if plot_edges:
        for conn in ts_conns:
            pt1, pt2 = min_pos[conn[0]-1,:], min_pos[conn[1]-1,:]
            #if communities[conn[0]-1] == communities[conn[1]-1]:
                #color the edge according to that community
            #    ax.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"-", color=node_colors[communities[conn[0]-1]], lw=2)
            ax.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"k-", alpha=0.25,lw=0.5)
    # plot vertices
    ax.scatter(min_pos[:,0],min_pos[:,1], c=node_colors, alpha=1.0, s=10)
    # plot
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    #plt.show()
    fig.tight_layout()
    plt.savefig('plots/9state_network_pgtcomms_bace9.pdf')

def plot_pgt_network(min_pos, **kwargs):
    """ Plot a partially graph-transformed network."""
    data_path = Path('KTN_data/9state')
    #GT-reduced network
    r_B, r_D, r_Q, r_N, r_BF, r_communities, rm_vec = pgt.prune_all_basins(beta=1.0,
        data_path=data_path, **kwargs)
    convert.ts_weights_conns_from_K(r_Q.todense(), data_path, suffix='_GT')
    ts_conns = np.loadtxt(data_path/'ts_conns_GT.dat', dtype=int)
    r_min_pos = min_pos[~rm_vec, :]
    communities = np.loadtxt(data_path/'communities_bace9.dat', dtype=int)
    r_comms = communities[~rm_vec]
    plot_network_communities(r_min_pos, ts_conns, communities=r_comms, plot_edges=True)

def illustrate_gt_gephi(temp, data_path = Path('KTN_data/32state'), suffix='gt', **kwargs):
    """ Show a network where (A U B)^c has been removed. """

    #GT-reduced
    beta = 1./temp
    B, K, tau, N, u, s, Emin, index_sel = kio.load_ktn(path=data_path,beta=beta,Emax=None,Nmax=None,screen=False)
    D = np.ravel(K.sum(axis=0))
    BF = beta*u-s
    BF -= BF.min()
    AS,BS = kio.load_ktn_AB(data_path,index_sel)
    #remove all of intervening region
    rm_vec = ~AS
    rm_vec[24] = False #don't remove attractor node of B
    r_B, r_tau, r_Q = gt.GT(rm_vec=rm_vec,B=B,tau=tau,rates=True,block=1,**kwargs)
    r_D = 1.0 /r_tau
    r_N = r_tau.size
    convert.ts_weights_conns_from_K(r_Q, data_path, suffix='_GT')
    ts_conns = np.loadtxt(data_path/'ts_conns_GT.dat', dtype=int)

    #get nodes in remaining network + their attributes
    mindata = np.loadtxt(data_path/'min.data')
    nnodes = mindata.shape[0]
    node_ids = np.arange(1, nnodes+1, 1) #all node IDs
    pgt_node_ids = node_ids[~rm_vec] #node IDs remaining in network
    pgt_fake_ids = np.arange(1, r_N+1, 1) #what the new ts_conns file thinks the IDs are
    pgt_id_dict = {}
    for i, id in enumerate(pgt_fake_ids):
        pgt_id_dict[id] = pgt_node_ids[i]

    #record whether node remains or not
    df = pd.DataFrame(columns=['Id', 'remains'])
    df['Id'] = node_ids
    df['remains'] = ~rm_vec #true if remains
    df.to_csv('csvs/nodes_to_remove_32state_Ab.csv', index=False)

    #load edges from original network
    edge_df = pd.DataFrame(columns=['source', 'target', 'weight'])
    for i in range(ts_conns.shape[0]):
        #convert to original node IDs
        ts_conns[i, 0] = pgt_id_dict[ts_conns[i, 0]]
        ts_conns[i, 1] = pgt_id_dict[ts_conns[i, 1]]
    edge_df['source'] = ts_conns[:, 0]
    edge_df['target'] = ts_conns[:, 1]
    edge_df['weight'] = np.tile(1.0, ts_conns.shape[0])
    edge_df.to_csv(f'csvs/edge_list_{suffix}.csv', index=False)


def dump_pgt_network_gephi(min_data, suffix='pgt', **kwargs):
    """ Plot a partially graph-transformed network."""
    data_path = Path('KTN_data/9state')
    #GT-reduced network
    r_B, r_D, r_Q, r_N, r_BF, r_communities, rm_vec = pgt.prune_all_basins(beta=1.0,
        data_path=data_path, **kwargs)
    convert.ts_weights_conns_from_K(r_Q, data_path, suffix='_GT')
    ts_conns = np.loadtxt(data_path/'ts_conns_GT.dat', dtype=int)
    communities = np.loadtxt(data_path/'communities_bace9.dat', dtype=int)
    r_comms = communities[~rm_vec]

    #get nodes in remaining network + their attributes
    mindata = np.loadtxt(min_data)
    nnodes = mindata.shape[0]
    node_ids = np.arange(1, nnodes+1, 1) #all node IDs
    pgt_node_ids = node_ids[~rm_vec] #node IDs remaining in network
    pgt_fake_ids = np.arange(1, r_N+1, 1) #what the new ts_conns file thinks the IDs are
    pgt_id_dict = {}
    for i, id in enumerate(pgt_fake_ids):
        pgt_id_dict[id] = pgt_node_ids[i]
    mindata = mindata[~rm_vec, :]
    node_df = pd.DataFrame(columns=['Id', 'Energy', 'community'])
    node_df['Id'] = pgt_node_ids
    node_df['Energy'] = mindata[:, 0].astype('float')
    node_df['community'] = r_comms
    node_df.to_csv(f'csvs/node_list_{suffix}.csv', index=False)

    #record whether node remains or not
    df = pd.DataFrame(columns=['Id', 'remains'])
    df['Id'] = node_ids
    df['remains'] = ~rm_vec #true if remains
    df.to_csv('csvs/nodes_to_remove_9state.csv', index=False)

    #load edges from original network
    edge_df = pd.DataFrame(columns=['source', 'target', 'weight'])
    for i in range(ts_conns.shape[0]):
        #convert to original node IDs
        ts_conns[i, 0] = pgt_id_dict[ts_conns[i, 0]]
        ts_conns[i, 1] = pgt_id_dict[ts_conns[i, 1]]
    edge_df['source'] = ts_conns[:, 0]
    edge_df['target'] = ts_conns[:, 1]
    edge_df['weight'] = np.tile(1.0, ts_conns.shape[0])
    edge_df.to_csv(f'csvs/edge_list_{suffix}.csv', index=False)

def plot_networks():
    endpt1 = 585
    endpt2 = 827
    data_path = Path('KTN_data/9state')
    min_ens = read_data(data_path/'min.data',1)
    ts_ens, ts_conns = read_data(data_path/'ts.data',2)
    min_pos = read_pos(data_path/'min.pos.dummy')
    ts_pos = read_pos(data_path/'ts.pos.dummy')
#    min_ens = [x for (y,x) in sorted(zip(min_ens,
    plot_network_communities(min_pos, ts_conns, plot_edges=True)
    #plot_pgt_network(min_pos, percent_retained=73, rm_type='hybrid')
    plot_network_landscape(min_ens,min_pos,ts_conns,plot_nodes=True)

#calculate ratio of MFPT in reduced network to full network for all pairs of communities
#same for the second moment
def get_first_second_moment_ratios_reduced_full(beta, r_BF, r_Q, r_comms, data_path=Path('KTN_data/9state')):

    #first compute c1<->c2 passage time distributions on full network
    B, K, tau, N, u, s, Emin, index_sel = kio.load_ktn(path=data_path,beta=beta,Emax=None,Nmax=None,screen=False)
    D = np.ravel(K.sum(axis=0))
    Q = diags(D)-K
    BF = beta*u-s
    BF -= BF.min()
    communities = pgt.read_communities(data_path/'communities.dat', index_sel)
    ncomms = len(communities)
    #i <- j <tau> and <tau^2> ratios for reduced/full
    mfpt_mat = np.ones((ncomms, ncomms))
    std_mat = np.ones((ncomms, ncomms))
    for c1 in communities:
        for c2 in communities:
            if c1 < c2:
                #update matrices
                tau_full = pgt.compute_passage_stats(communities[c1], communities[c2], BF, Q, dopdf=False)
                #now compute c1<->c2 passage time distributions on reduced network
                tau = pgt.compute_passage_stats(r_comms[c1], r_comms[c2], r_BF,
                                                r_Q, dopdf=False)
                #c2 <- c1
                mfpt_mat[c2][c1] = tau[0]/tau_full[0]
                std_mat[c2][c1] = tau[1]/tau_full[1]
                #c1 <-c2
                mfpt_mat[c1][c2] = tau[2]/tau_full[2]
                std_mat[c1][c2] = tau[3]/tau_full[3]
    return mfpt_mat, std_mat

def mfpt_reduced_full_GT(betas=np.linspace(0.1, 10.0, 20), c1=7, c2=3,
                                           data_path=Path('KTN_data/9state')):
    """ For each temperature, compute MFPT c1<->c2 in reduced and full networks.
    Plot T_AB vs. 1/T. """

    tauAB_gt = np.zeros((2,len(betas)))
    tauAB_full = np.zeros(len(betas))
    tauBA_gt = np.zeros((2, len(betas)))
    tauBA_full = np.zeros(len(betas))
    for i, beta in enumerate(betas):
        #first compute c1<->c2 passage time distributions on full network
        B, K, tau, N, u, s, Emin, index_sel = kio.load_ktn(path=data_path,beta=beta,Emax=None,Nmax=None,screen=False)
        D = np.ravel(K.sum(axis=0))
        Q = diags(D) - K
        BF = beta*u-s
        BF -= BF.min()
        temp = 1./beta
        #first calculate MFPT on full network using PATHSAMPLE
        ktn = Analyze_KTN(path=data_path,
                          commdata='communities_bace.dat')
        convert.dump_rate_mat(convert.K_from_Q(Q), data_path)
        #TODO: update this function to use READRATES keyword in pathsample
        MFPTAB, MFPTBA = ktn.get_MFPT_AB(c1, c2, temp, N)
        tauAB_full[i] = MFPTAB
        tauBA_full[i] = MFPTBA
        #then repeat on reduced network retaining 75% of states
        r_B, r_D, r_Q, r_N, r_BF, r_comms = pgt.prune_all_basins(beta=beta, data_path=data_path,
                                                                rm_type='hybrid', percent_retained=53)
        #dump reduced rate matrix into file that PATHSAMPLE can read
        convert.dump_rate_mat(convert.K_from_Q(r_Q), data_path)
        ktn = Analyze_KTN(path=data_path, communities=convert.ktn_comms_from_gt_comms(r_comms))
        MFPTAB, MFPTBA = ktn.get_MFPT_AB(c1, c2, temp, r_N)
        tauAB_gt[0, i] = MFPTAB
        tauBA_gt[0, i] = MFPTBA
        #then repeat on reduced network retaining 25% of states
        r_B, r_D, r_Q, r_N, r_BF, r_comms = pgt.prune_all_basins(beta=beta, data_path=data_path,
                                                                rm_type='hybrid', percent_retained=13)
        #dump reduced rate matrix into file that PATHSAMPLE can read
        convert.dump_rate_mat(convert.K_from_Q(r_Q), data_path)
        ktn = Analyze_KTN(path=data_path, communities=convert.ktn_comms_from_gt_comms(r_comms))
        MFPTAB, MFPTBA = ktn.get_MFPT_AB(c1, c2, temp, r_N)
        tauAB_gt[1, i] = MFPTAB
        tauBA_gt[1, i] = MFPTBA
    fig, ax = plt.subplots(1, 2, figsize=(textwidth, textwidth/3))
    colors = sns.color_palette("bright", 10)
    ax[0].plot(betas, tauAB_full, '-', color=colors[0],
               label=r'$\mathcal{T}_{\mathcal{A}\mathcal{B}}$')
    ax[0].plot(betas, tauAB_gt[0,:], '--', color=colors[1],
               label=r'$\mathcal{T}_{\mathcal{A}\mathcal{B}}$ (retaining 75\%)')
    ax[0].plot(betas, tauAB_gt[1,:], '--', color=colors[2],
               label=r'$\mathcal{T}_{\mathcal{A}\mathcal{B}}$ (retaining 25\%)')
    ax[0].set_xlabel(r'$1/T$')
    ax[0].set_ylabel('Time')
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[1].plot(betas, tauBA_full, '-', color=colors[0],
               label=r'$\mathcal{T}_{\mathcal{B}\mathcal{A}}$')
    ax[1].plot(betas, tauBA_gt[0,:], '--', color=colors[1],
               label=r'$\mathcal{T}_{\mathcal{B}\mathcal{A}}$ (retaining 75\%)')
    ax[1].plot(betas, tauBA_gt[1,:], '--', color=colors[2],
               label=r'$\mathcal{T}_{\mathcal{B}\mathcal{A}}$ (retaining 25\%)')
    ax[1].set_xlabel(r'$1/T$')
    ax[1].set_ylabel('Time')
    ax[1].set_yscale('log')
    ax[1].legend()
    fig.tight_layout()
    return tauAB_full, tauAB_gt, tauBA_full, tauBA_gt

def compare_pgt_network_pres(beta=1.0, percent_retained=30, mfpt=True, data_path=Path('KTN_data/9state')):
    """ Make a single heatmap with just the MFPT ratios for a given coarse-graining amount."""
    fig, ax = plt.subplots(figsize=(4.5,4.2))
    #perform GT-reduction
    r_B, r_D, r_Q, r_N, r_BF, r_comms = pgt.prune_basins_sequentially(beta=beta, data_path=data_path,
                                                rm_type='hybrid', percent_retained=percent_retained)

    mfpt_mat, std_mat = get_first_second_moment_ratios_reduced_full(beta, r_BF, r_Q, r_comms)
    #mfpt_labels = [][]
    #[f'{ratio:.1f}' for ratio in mfpt_mat]
    sns.heatmap(std_mat, center=1.0, linewidths=0.25, linecolor='k', square=True, robust=True,
                annot=True, annot_kws={'fontsize':16, 'family':'sans-serif'}, cmap='coolwarm', ax=ax)
    ax.set_title(r'$\mathcal{T}_{IJ}^{\rm GT} / \mathcal{T}_{IJ}$, retaining 50\%')
    fig.tight_layout()
    plt.savefig('plots/heatmap_2nd_T1.0_hybrid50_sequential.pdf')

def compare_pgt_networks(beta=1.0, data_path=Path('KTN_data/9state')):
    """ Plot 4 panel figure where top two are MFPT and std heatmaps from removing
    25% of nodes and the bottom two are MFPT and std heatmaps from removing 75% of the nodes."""

    fig, (ax0, ax1) = plt.subplots(2, 2, figsize=(textwidth*(2.5/3), textwidth*(2/3)))
    #start by conservatively removing 25% of the nodes at T=1
    #recall, that these nodes do not have a high tpp density
    r_B, r_D, r_Q, r_N, r_BF, r_comms = pgt.prune_basins_sequentially(beta=beta, data_path=data_path,
                                                            rm_type='hybrid', percent_retained=30)

    mfpt_mat, std_mat = get_first_second_moment_ratios_reduced_full(beta, r_BF, r_Q, r_comms)
    #mfpt_labels = [][]
    #[f'{ratio:.1f}' for ratio in mfpt_mat]
    sns.heatmap(mfpt_mat, center=1.0, linewidths=0.25, linecolor='k', square=True, robust=True,
                annot=True, annot_kws={'fontsize':6, 'family':'sans-serif'}, cmap='coolwarm', ax=ax0[0])
    sns.heatmap(std_mat, center=1.0, linewidths=0.25, linecolor='k', square=True, robust=True,
                annot=True, annot_kws={'fontsize':6, 'family':'sans-serif'}, cmap='coolwarm', ax=ax0[1])
    ax0[0].set_title(r'$\mathcal{T}_{IJ}^{\mathcal{Z}} / \mathcal{T}_{IJ}$, retaining 50\%')
    ax0[1].set_title(r'$\sigma_{IJ}^{\mathcal{Z}} / \sigma_{IJ}$, retaining 50\%')

    #this time, RETAIN 25%
    r_B, r_D, r_Q, r_N, r_BF, r_comms = pgt.prune_basins_sequentially(beta=beta, data_path=data_path,
                                                               rm_type='hybrid', percent_retained=5)
    mfpt_mat, std_mat = get_first_second_moment_ratios_reduced_full(beta, r_BF, r_Q, r_comms)
    minnorm = min(mfpt_mat.min(), std_mat.min())
    maxnorm = max(mfpt_mat.max(), std_mat.max())
    lognorm = LogNorm(minnorm, maxnorm)
    sns.heatmap(mfpt_mat, center=1.0, linewidths=0.25, linecolor='k', cmap='coolwarm', robust=True,
                square=True, annot=True, annot_kws={'fontsize':6, 'family':'sans-serif'}, ax=ax1[0])

    sns.heatmap(std_mat, center=1.0, linewidths=0.25, linecolor='k', cmap='coolwarm', robust=True,
                square=True, annot=True, annot_kws={'fontsize':6, 'family':'sans-serif'}, ax=ax1[1])
    ax1[0].set_title(r'$\mathcal{T}_{IJ}^{\mathcal{Z}} / \mathcal{T}_{IJ}$, retaining 10\%')
    ax1[1].set_title(r'$\sigma_{IJ}^{\mathcal{Z}} / \sigma_{IJ}$, retaining 10\%')
    fig.tight_layout()
    plt.savefig('plots/heatmaps_T1.0_hybrid_50_10_sequential.pdf')

def rank_nodes_to_eliminate(beta=1.0/0.25, escape_time_upper_bound=1000):
    """ Scatter nodes by node degree, free energy, and escape time,
    coloring by A, B, and I.
    """
    Nmax = None
    data_path = "KTN_data/LJ38/4k"
    B, K, escape_times, N, u, s, Emin, index_sel = kio.load_ktn(path=data_path,beta=beta,Emax=None,Nmax=Nmax,screen=False)
    D = np.ravel(K.sum(axis=0)) #array of size (N,) containing escape rates for each min
    BF = beta*u-s
    BF -= BF.min()
    node_degree = B.indptr[1:] - B.indptr[:-1]
    print(f'Number of nodes with >1 connection: {len(node_degree[node_degree > 1])}')
    AS,BS = kio.load_ktn_AB(data_path,index_sel)
    IS = np.zeros(N, bool)
    IS[~(AS+BS)] = True
    print(f'A: {AS.sum()}, B: {BS.sum()}, I: {IS.sum()}')

    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(15,4))
    colors = sns.color_palette("Paired")
    #node degree vs escape time
    ax.scatter(node_degree[IS], escape_times[IS], color=colors[8], alpha=0.4, label='I')
    ax.scatter(node_degree[AS], escape_times[AS], color=colors[5], alpha=0.8, label='A')
    ax.scatter(node_degree[BS], escape_times[BS], color=colors[1], alpha=0.4, label='B')
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Escape Time')
    #ax.set_yscale('log')
    ax.set_ylim([-1.0, escape_time_upper_bound])
    ax.legend()
    #free energy vs escape time
    ax1.scatter(BF[IS], escape_times[IS], color=colors[8], alpha=0.4, label='I')
    ax1.scatter(BF[AS], escape_times[AS], color=colors[5], alpha=0.8, label='A')
    ax1.scatter(BF[BS], escape_times[BS], color=colors[1], alpha=0.4, label='B')
    ax1.set_xlabel('Free Energy')
    ax1.set_ylabel('Escape Time')
    #ax1.set_yscale('log')
    ax1.set_ylim([-1.0, escape_time_upper_bound])
    ax1.legend()
    #node degree vs free energy
    ax2.scatter(BF[IS], node_degree[IS], color=colors[8], alpha=0.4, label='I')
    ax2.scatter(BF[AS], node_degree[AS], color=colors[5], alpha=0.8, label='A')
    ax2.scatter(BF[BS], node_degree[BS], color=colors[1], alpha=0.4, label='B')
    ax2.set_xlabel('Free Energy')
    ax2.set_ylabel('Node Degree')
    ax2.legend()
    plt.show()
    fig.tight_layout()

def scatter_visitation_prob(beta = 1.0, rm_type='hybrid', percent_retained=75):
    data_path = Path("KTN_data/9state")
    temp = 1.
    beta = 1./temp
    B, K, escape_times, N, u, s, Emin, index_sel = kio.load_ktn(path=data_path,beta=beta,Emax=None,Nmax=None,screen=True)
    D = np.ravel(K.sum(axis=0))
    #free energy of minima
    BF = beta*u-s
    #rescaled
    BF -= BF.min()
    #node degree
    node_degree = B.indptr[1:] - B.indptr[:-1]
    # AB regions
    AS,BS = kio.load_ktn_AB(data_path,index_sel)
    IS = np.zeros(N, bool)
    IS[~(AS+BS)] = True
    tp_densities = np.loadtxt(Path(data_path)/'tp_stats.dat')[:,3]
    rm_vec = np.zeros(N,bool)
    #color nodes that we would propose to remove
    if rm_type == 'node_degree':
        rm_vec[node_degree < 2] = True
        rm_vec[(AS+BS)] = False #only remove the intermediate nodes this time

    if rm_type == 'escape_time':
        #remove nodes with the smallest escape times
        #retain nodes in the top percent_retained percentile of escape time
        rm_vec[IS] = escape_times[IS] < np.percentile(escape_times[IS], 100.0 - percent_retained)

    if rm_type == 'free_energy':
        rm_vec[IS] = BF[IS] > np.percentile(BF[IS], percent_retained)

    if rm_type == 'combined':
        rho = np.exp(-BF)
        rho /= rho.sum()
        combo_metric = escape_times * rho
        rm_vec[IS] = combo_metric[IS] < np.percentile(combo_metric[IS], 100.0 - percent_retained)

    if rm_type == 'hybrid':
        #remove nodes in the top percent_retained percentile of escape time
        time_sel = (escape_times[IS] < np.percentile(escape_times[IS], 100.0 - percent_retained))
        bf_sel = (BF[IS]>np.percentile(BF[IS],percent_retained))
        sel = np.bitwise_and(time_sel, bf_sel)
        #that are also in the lowest percent_retained percentile of free energy
        rm_vec[IS] = sel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.71))
    colors = sns.color_palette("Paired")
    #tp_density vs escape time
    ax1.scatter(tp_densities[~rm_vec], escape_times[~rm_vec], color=colors[8],
        s=15, alpha=0.5, label='intervening nodes')
    #ax1.scatter(tp_densities[AS], escape_times[AS], color=colors[5],
    #    s=3, alpha=0.8, label='A')
    #ax1.scatter(tp_densities[BS], escape_times[BS], color=colors[1],
    #    s=3, alpha=0.8, label='B')
    ax1.scatter(tp_densities[rm_vec], escape_times[rm_vec], color=colors[9],
        s=15, alpha=0.8, label='nodes to remove')
    ax1.set_xlabel(r'$\mathcal{A}\leftarrow \mathcal{B}$ visitation prob.')
    ax1.set_ylabel('Escape Time')
    ax1.set_yscale('log')
    handles, labels = ax1.get_legend_handles_labels()
    #ax1.legend(ncol=2, loc=(0, 1.05))
    #tp_density vs free energy
    ax2.scatter(BF[~rm_vec], tp_densities[~rm_vec], color=colors[8], alpha=0.5, s=15, label='intervening nodes')
    #ax2.scatter(BF[AS], tp_densities[AS], color=colors[5], alpha=0.8, s=3, label='A')
    #ax2.scatter(BF[BS], tp_densities[BS], color=colors[1], alpha=0.8, s=3, label='B')
    ax2.scatter(BF[rm_vec], tp_densities[rm_vec], color=colors[9], alpha=0.8, s=15, label='nodes to remove')
    ax2.set_xlabel('Free Energy')
    ax2.set_ylabel(r'$\mathcal{A}\leftarrow \mathcal{B}$ visitation prob.')
    fig.tight_layout()
    fig.legend(handles, labels, loc=(0, 1.0), ncol=2)
    plt.savefig('plots/visitation_probability_scatter.pdf')

def plot_AB_waiting_time(beta, data_path='KTN_data/LJ38/4k/', size=[5, 128], percent_retained=10, **kwargs):
    """ Plot two panels, A->B first passage time in full and reduced networks,
    and B->A first passage time in full and reduced networks. Reduce networks
    with three different heuristics (escape_time, free_energy, hybrid).
    """
    beta, tau, gttau_time, pt, gtpt_time = pgt.prune_intermediate_nodes(beta, data_path=data_path, dopdf=True,
                                            percent_retained=percent_retained, rm_type='escape_time', **kwargs)
    beta, tau, gttau_bf, pt, gtpt_bf = pgt.prune_intermediate_nodes(beta, data_path=data_path, dopdf=True, percent_retained=percent_retained,
                                                        rm_type='free_energy', **kwargs)
    beta, tau, gttau_hybrid, pt, gtpt_hybrid = pgt.prune_intermediate_nodes(beta, data_path=data_path, dopdf=True,
                                                        rm_type='combined', percent_retained=percent_retained, **kwargs)
    fig, ax = plt.subplots(1, 2, figsize=(textwidth, textwidth/2.25))
    colors = sns.color_palette("bright", 10)
    names = ["A", "B"]
    for j in range(2):
        ax[j].set_title(r"%s$\to$%s (from %d$\to$%d states, T=%2.2g)" % (names[j], names[1-j], size[j],size[1-j],1.0/beta))
        #full
        ax[j].plot(pt[2*j]/tau[2*j],pt[1+2*j],'-', color=colors[0], lw=2,
            label=r"Full $p_\mathcal{%s\to{%s}}(t)$" % (names[j],names[1-j]))
        #by escape time
        ax[j].plot(gtpt_time[2*j]/gttau_time[2*j],gtpt_time[1+2*j],'--', color=colors[-1], lw=2,
            label=r"$p^{GT}_\mathcal{%s\to{%s}}(t)$, time" % (names[j],names[1-j]))
        #by free energy
        ax[j].plot(gtpt_bf[2*j]/gttau_bf[2*j],gtpt_bf[1+2*j],'--',color=colors[-2],lw=2,
            label=r"$p^{GT}_\mathcal{%s\to{%s}}(t), \beta F$" % (names[j],names[1-j]))
        #by hybrid approach
        ax[j].plot(gtpt_hybrid[2*j]/gttau_hybrid[2*j],gtpt_hybrid[1+2*j],'-.',color=colors[1], lw=2,
            label=r"$p^{GT}_\mathcal{%s\to{%s}}(t)$, combined" % (names[j],names[1-j]))
        ax[j].set_xlabel(r"$t/\langle t \rangle$")
        if j==0:
            ax[j].set_ylabel(r"$\langle t \rangle p(t/\langle t \rangle)$")
        #if j==1:
        ax[j].legend()
        ax[j].set_xscale('log')
        ax[j].set_yscale('log')
        ax[j].set_xlim(0.001,100.)
        ax[j].set_ylim(pt[1+2*j].min()/10.0,10.0)
    fig.tight_layout()
    plt.savefig('plots/LJ38_partialGT_ABwaitpdf_combined.pdf')

def plot_escapeB(betas=[4.0, 7.0], data_path='KTN_data/LJ38/4k/', percent_retained=25, **kwargs):
    """Plot a four panel figure. Top two panels show escape time distributions
    from the icosahedral funnel of LJ38 in the full network and in 3 reduced
    networks at two different temperatures.
    Bottom two panels depict the mean and variance of the escape time
    distribution as a function of temperature."""


    fig, (ax, ax1) = plt.subplots(2, 2, figsize=(textwidth, textwidth*(2/3)))
    colors = sns.color_palette("bright", 10)
    names = ["A", "B"]
    for j in range(2):
        beta, tau, gttau_bf, pt, gtpt_bf = pgt.prune_source(betas[j], data_path=data_path, dopdf=True,
                                                        percent_retained_in_B=percent_retained,
                                                        rm_type='free_energy', **kwargs)
        beta, tau, gttau_time, pt, gtpt_time = pgt.prune_source(betas[j], data_path=data_path, dopdf=True,
                                                        percent_retained_in_B=percent_retained,
                                                        rm_type='escape_time', **kwargs)
        beta, tau, gttau_hybrid, pt, gtpt_hybrid = pgt.prune_source(betas[j], data_path=data_path, dopdf=True,
                                                            rm_type='combined',
                                                            percent_retained_in_B=percent_retained, **kwargs)
        ax[j].set_title(r"Escape from $\mathcal{B}$, T=%2.2g" % (1.0/betas[j]))
        #by escape time
        ax[j].plot(gtpt_time[0]/tau[0],gtpt_time[1],'-',
                   color=colors[-1], lw=1.5,
            label=r"$p^{GT}_\mathcal{B}(t)$, time")
        #by free energy
        ax[j].plot(gtpt_bf[0]/tau[0],gtpt_bf[1],'-',color=colors[-2],lw=1.5,
            label=r"$p^{GT}_\mathcal{B}(t), \beta F$" )
        #full
        ax[j].plot(pt[0]/tau[0],pt[1],'-', color=colors[0], lw=2,
            label=r"Full $p_\mathcal{B}(t)$")
        #by hybrid approach
        ax[j].plot(gtpt_hybrid[0]/tau[0],gtpt_hybrid[1],'-.',color=colors[1], lw=2,
            label=r"$p^{GT}_\mathcal{B}(t)$, combined")
        ax[j].set_xlabel(r"$t/\mathcal{T}_\mathcal{B}$")
        if j==0:
            ax[j].set_ylabel(r'$p(t/\mathcal{T}_\mathcal{B})\mathcal{T}_{\mathcal{B}}$')
        #if j==1:
        handles, labels = ax[j].get_legend_handles_labels()
        order = [2,0,1,3]
        ax[j].legend([handles[id] for id in order], [labels[id] for id in order])
        ax[j].set_xscale('log')
        ax[j].set_yscale('log')
        ax[j].set_xlim(0.001,1000.)
        ax[j].set_ylim(pt[1].min()/10.0,100.0)

    #now compute first and second moments and plot
    data = np.zeros((2, 20, 5))
    beta_range = np.linspace(2.5, 8.5, 20)
    for i, beta in enumerate(beta_range):
        data[0][i][0], data[0][i][1:3], data[0][i][3:5] = pgt.prune_source(beta, data_path=data_path, dopdf=False,
                                percent_retained_in_B=percent_retained,
                                rm_type='free_energy', **kwargs)
        data[1][i][0], data[1][i][1:3], data[1][i][3:5] = pgt.prune_source(beta, data_path=data_path, dopdf=False,
                                percent_retained_in_B=percent_retained,
                                rm_type='combined', **kwargs)
    names=[r"$\beta F$", "combined"]
    for j in range(2):
        #full mean escape time and second moment
        ax1[j].plot(data[j,:,0], data[j,:,1], '-', lw=2, color=colors[0],
                    label=r"$\mathcal{T}_\mathcal{B}$")
        ax1[j].plot(data[j,:,0], np.sqrt(data[j,:,2]-data[j,:,1]**2), '-', lw=2,
                    color=colors[2], label=r"$\sigma_\mathcal{B}$")
        #reduced mean escape time and second moment
        ax1[j].plot(data[j,:,0], data[j,:,3], '-.', lw=2, color=colors[1],
                    label=r"$\mathcal{T}_\mathcal{B}$ (GT, %s)" % (names[j]))
        ax1[j].plot(data[j,:,0], np.sqrt(data[j,:,4]-data[j,:,3]**2), '-.', lw=2,
                    color=colors[3], label=r"$\sigma_\mathcal{B}$ (GT, %s)" % (names[j]))
        ax1[j].set_xlabel(r"$1/T$")
        if j==0:
            ax1[j].set_ylabel("Time")
        handles, labels = ax1[j].get_legend_handles_labels()
        order = [2, 3, 0, 1]
        ax1[j].legend([handles[id] for id in order], [labels[id] for id in
                                                      order], ncol=2)
        ax1[j].set_yscale("log")
    fig.tight_layout()
    plt.savefig('plots/LJ38_partialGT_escapeB_combined.pdf')
