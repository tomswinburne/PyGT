""" Script to visualize escape time and first passage time distributions
in full and graph-transformed networks.

Plots figures in the partial GT manuscript.


Deepti Kannan, 2020"""

import numpy as np
from lib import partialGT as pgt
import lib.ktn_io as kio
import lib.gt_tools as gt
from scipy.sparse import save_npz,load_npz, diags, eye, csr_matrix, bmat
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import seaborn as sns
from pathlib import Path
sns.set()
textwidth = 6.47699

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
plt.rcParams.update(params)


#calculate ratio of MFPT in reduced network to full network for all pairs of communities
#same for the second moment
def get_first_second_moment_ratios_reduced_full(beta, r_BF, r_Q, r_comms, data_path=Path('KTN_data/9state')):

    #first compute c1<->c2 passage time distributions on full network
    B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=data_path,beta=beta,Emax=None,Nmax=None,screen=False)
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

def compare_pgt_networks(beta=1.0, data_path=Path('KTN_data/9state')):
    """ Plot 4 panel figure where top two are MFPT and std heatmaps from removing
    25% of nodes and the bottom two are MFPT and std heatmaps from removing 75% of the nodes."""

    fig, (ax0, ax1) = plt.subplots(2, 2, figsize=(textwidth*(2.5/3), textwidth*(2/3)))
    #start by conservatively removing 25% of the nodes at T=1
    #recall, that these nodes do not have a high tpp density
    r_B, r_D, r_Q, r_N, r_BF, r_comms = pgt.prune_all_basins(beta=beta, data_path=data_path,
                                                            rm_type='hybrid', percent_retained=53)

    mfpt_mat, std_mat = get_first_second_moment_ratios_reduced_full(beta, r_BF, r_Q, r_comms)
    sns.heatmap(mfpt_mat, center=1.0, linewidths=0.25, linecolor='k', square=True, robust=True,
                cmap='coolwarm', ax=ax0[0])
    sns.heatmap(std_mat, center=1.0, linewidths=0.25, linecolor='k', square=True, robust=True,
                cmap='coolwarm', ax=ax0[1])
    ax0[0].set_title(r'$\mathcal{T}_{IJ}^{\mathcal{Z}} / \mathcal{T}_{IJ}$, retaining 75\%')
    ax0[1].set_title(r'$\sigma_{IJ}^{\mathcal{Z}} / \sigma_{IJ}$, retaining 75\%')

    #this time, RETAIN 25%
    r_B, r_D, r_Q, r_N, r_BF, r_comms = pgt.prune_all_basins(beta=beta, data_path=data_path,
                                                               rm_type='hybrid', percent_retained=13)
    mfpt_mat, std_mat = get_first_second_moment_ratios_reduced_full(beta, r_BF, r_Q, r_comms)
    minnorm = min(mfpt_mat.min(), std_mat.min())
    maxnorm = max(mfpt_mat.max(), std_mat.max())
    lognorm = LogNorm(minnorm, maxnorm)
    sns.heatmap(mfpt_mat, center=1.0, linewidths=0.25, linecolor='k', cmap='coolwarm', robust=True,
                square=True, norm=lognorm, cbar_kws={"ticks":[0,1,10,1e2,1e3, 1e4]}, ax=ax1[0])
                
    sns.heatmap(std_mat, center=1.0, linewidths=0.25, linecolor='k', cmap='coolwarm', robust=True,
                square=True, norm=lognorm, cbar_kws={"ticks":[0,1,10,1e2,1e3, 1e4]}, ax=ax1[1])
    ax1[0].set_title(r'$\mathcal{T}_{IJ}^{\mathcal{Z}} / \mathcal{T}_{IJ}$, retaining 25\%')
    ax1[1].set_title(r'$\sigma_{IJ}^{\mathcal{Z}} / \sigma_{IJ}$, retaining 25\%')
    fig.tight_layout()
    plt.savefig('plots/heatmaps_T1.0_hybrid.pdf')

def rank_nodes_to_eliminate(beta=1.0/0.25, escape_time_upper_bound=1000):
    """ Scatter nodes by node degree, free energy, and escape time, 
    coloring by A, B, and I.
    """
    Nmax = None
    data_path = "KTN_data/LJ38/4k"
    B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=data_path,beta=beta,Emax=None,Nmax=Nmax,screen=False)
    D = np.ravel(K.sum(axis=0)) #array of size (N,) containing escape rates for each min
    BF = beta*u-s
    BF -= BF.min()
    escape_times = 1.0/D
    node_degree = B.indptr[1:] - B.indptr[:-1]
    print(f'Number of nodes with >1 connection: {len(node_degree[node_degree > 1])}')
    AS,BS = kio.load_AB(data_path,index_sel)
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

def scatter_committors_tpd(beta = 1.0, rm_type='hybrid', percent_retained=75):
    data_path = Path("KTN_data/9state")
    temp = 1.
    beta = 1./temp
    B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=data_path,beta=beta,Emax=None,Nmax=None,screen=True)
    D = np.ravel(K.sum(axis=0)) 
    #escape time
    escape_times = 1./D
    #free energy of minima
    BF = beta*u-s
    #rescaled
    BF -= BF.min()
    #node degree
    node_degree = B.indptr[1:] - B.indptr[:-1]
    # AB regions
    #AS,BS = kio.load_AB(data_path,index_sel)
    Amin = 585
    Bmin = 827
    AS = np.zeros(N, bool)
    BS = np.zeros(N, bool)
    AS[Amin-1] = True
    BS[Bmin-1] = True
    IS = np.zeros(N, bool)
    IS[~(AS+BS)] = True
    tp_densities = np.loadtxt(Path(data_path)/'tp_densities.dat')
    rm_reg = np.zeros(N,bool)
    #color nodes that we would propose to remove
    if rm_type == 'node_degree':
        rm_reg[node_degree < 2] = True
        rm_reg[(AS+BS)] = False #only remove the intermediate nodes this time

    if rm_type == 'escape_time':
        #remove nodes with the smallest escape times
        #retain nodes in the top percent_retained percentile of escape time
        rm_reg[IS] = escape_times[IS] < np.percentile(escape_times[IS], 100.0 - percent_retained)

    if rm_type == 'free_energy':
        rm_reg[IS] = BF[IS] > np.percentile(BF[IS], percent_retained)

    if rm_type == 'combined':
        rho = np.exp(-BF)
        rho /= rho.sum()
        combo_metric = escape_times * rho
        rm_reg[IS] = combo_metric[IS] < np.percentile(combo_metric[IS], 100.0 - percent_retained)
        
    if rm_type == 'hybrid':
        #remove nodes in the top percent_retained percentile of escape time
        time_sel = (escape_times[IS] < np.percentile(escape_times[IS], 100.0 - percent_retained))
        bf_sel = (BF[IS]>np.percentile(BF[IS],percent_retained))
        sel = np.bitwise_and(time_sel, bf_sel)
        #that are also in the lowest percent_retained percentile of free energy
        rm_reg[IS] = sel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(textwidth*(2/3), textwidth/3))
    colors = sns.color_palette("Paired")
    #tp_density vs escape time
    ax1.scatter(tp_densities[~rm_reg], escape_times[~rm_reg], color=colors[8], 
        s=3, alpha=0.4, label='I')
    ax1.scatter(tp_densities[AS], escape_times[AS], color=colors[5], 
        s=3, alpha=0.8, label='A')
    ax1.scatter(tp_densities[BS], escape_times[BS], color=colors[1], 
        s=3, alpha=0.8, label='B')
    ax1.scatter(tp_densities[rm_reg], escape_times[rm_reg], color=colors[9], 
        s=3, alpha=0.8, label='rm')
    ax1.set_xlabel('tpp density')
    ax1.set_ylabel('Escape Time')
    ax1.set_yscale('log')
    ax1.legend(ncol=2)
    #tp_density vs free energy
    ax2.scatter(BF[~rm_reg], tp_densities[~rm_reg], color=colors[8], alpha=0.4, s=3, label='I')
    ax2.scatter(BF[AS], tp_densities[AS], color=colors[5], alpha=0.8, s=3, label='A')
    ax2.scatter(BF[BS], tp_densities[BS], color=colors[1], alpha=0.8, s=3, label='B')
    ax2.scatter(BF[rm_reg], tp_densities[rm_reg], color=colors[9], alpha=0.8, s=3, label='rm')
    ax2.set_xlabel('Free Energy')
    ax2.set_ylabel('tpp density')
    fig.tight_layout()
    plt.savefig('plots/tpp_bf_tau_scatter.pdf')

def plot_AB_waiting_time(beta, size=[5, 128], percent_retained=10, **kwargs):
    """ Plot two panels, A->B first passage time in full and reduced networks,
    and B->A first passage time in full and reduced networks. Reduce networks
    with three different heuristics (escape_time, free_energy, hybrid).
    """
    beta, tau, gttau_time, pt, gtpt_time = prune_intermediate_nodes(beta, dopdf=True, 
                                                        rm_type='escape_time', **kwargs)
    beta, tau, gttau_bf, pt, gtpt_bf = prune_intermediate_nodes(beta, dopdf=True, 
                                                        rm_type='free_energy', **kwargs)
    beta, tau, gttau_hybrid, pt, gtpt_hybrid = prune_intermediate_nodes(beta, dopdf=True, 
                                                        rm_type='hybrid', percent_retained=percent_retained/2, **kwargs)
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
            label=r"$p^{GT}_\mathcal{%s\to{%s}}(t)$, hybrid" % (names[j],names[1-j]))
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
    plt.savefig('plots/LJ38_partialGT_ABwaitpdf.pdf')

def plot_escapeB(betas=[4.0, 7.0], percent_retained=25, **kwargs):
    """Plot a four panel figure. Top two panels show escape time distributions
    from the icosahedral funnel of LJ38 in the full network and in 3 reduced
    networks at two different temperatures.
    Bottom two panels depict the mean and variance of the escape time
    distribution as a function of temperature."""

    
    fig, (ax, ax1) = plt.subplots(2, 2, figsize=(textwidth, textwidth*(2/3)))
    colors = sns.color_palette("bright", 10)
    names = ["A", "B"]
    for j in range(2):
        beta, tau, gttau_bf, pt, gtpt_bf = prune_source(betas[j], dopdf=True,
                                                        percent_retained_in_B=percent_retained, 
                                                        rm_type='free_energy', **kwargs)
        beta, tau, gttau_time, pt, gtpt_time = prune_source(betas[j], dopdf=True,
                                                        percent_retained_in_B=percent_retained, 
                                                        rm_type='escape_time', **kwargs)
        beta, tau, gttau_hybrid, pt, gtpt_hybrid = prune_source(betas[j], dopdf=True, 
                                                            rm_type='hybrid',
                                                            percent_retained_in_B=percent_retained/2, **kwargs)
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
            label=r"$p^{GT}_\mathcal{B}(t)$, hybrid")
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
        data[0][i][0], data[0][i][1:3], data[0][i][3:5] = prune_source(beta, dopdf=False,
                                percent_retained_in_B=percent_retained,
                                rm_type='free_energy', **kwargs)
        data[1][i][0], data[1][i][1:3], data[1][i][3:5] = prune_source(beta, dopdf=False,
                                percent_retained_in_B=12,
                                rm_type='hybrid', **kwargs)
    names=[r"$\beta F$", "hybrid"]    
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
    plt.savefig('plots/LJ38_partialGT_escapeB.pdf')



