""" Plotting script for DTMC manuscript.


Deepti Kannan, 2020"""

#library code
import lib.ktn_io as kio
import lib.gt_tools as gt
import lib.partialGT as pgt
import lib.conversion as convert
from ktn.ktn_analysis import *
#other modules
import numpy as np
import scipy as sp
import scipy.linalg as spla 
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.linalg import eig
from scipy.linalg import expm
from scipy.interpolate import CubicSpline
from pathlib import Path
import pandas as pd
import os
import subprocess
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
sns.set()

textwidth = 6.47699
columnwidth = 246.0/72
purple = '#BB6DBA'
orange = '#F57937'
green = '#6CB41C'
blue = '#07B4CC'
grey = '#2D3A3B'

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

def compare_HS_LEA(temps, data_path=Path('KTN_data/32state'), theta=False):
    """ Calculate coarse-grained rate matrices using the Hummer-Szabo and LEA
    methods and compute MFPTAB/BA using NGT to be compared to the rates on the
    full network. """

    dfs = []
    for temp in temps:
        df = pd.DataFrame()
        df['T'] = [temp]
        ktn = Analyze_KTN(data_path, commdata='communities.dat')
        #KTN input
        beta = 1./temp
        B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=data_path,beta=beta,Emax=None,Nmax=None,screen=False)
        Q = K.todense() - D.todense()
        AS, BS = kio.load_AB(data_path,index_sel) 
        D = np.ravel(K.sum(axis=0))
        BF = beta*u-s
        BF -= BF.min()
        pi = np.exp(-BF)
        pi /= pi.sum()
        #ktn setup
        commpi = ktn.get_comm_stat_probs(np.log(pi), log=False)
        ktn.K = Q
        ktn.pi = pi
        ktn.commpi = commpi
        ncomms = len(commpi)
        #MFPT calculations on full network
        full_df = pgt.compute_rates(AS, BS, BF, B, D, K=K, fullGT=True)
        df['MFPTAB'] = full_df['MFPTAB']
        df['MFPTBA'] = full_df['MFPTBA']
        
        #compute coarse-grained networks: 4 versions of Hummer-Szabo + LEA
        mfpt, pi = pgt.get_intermicrostate_mfpts_GT(temp, data_path)
        labels = []
        matrices = []
        try:
            Rhs = ktn.construct_coarse_matrix_Hummer_Szabo(temp)
            matrices.append(Rhs)
            labels.append('HS')
        except Exception as e:
            print(f'Hummer Szabo had the following error: {e}')
        try:
            Rhs_kkra = ktn.hummer_szabo_from_mfpt(temp, GT=False, mfpt=mfpt)
            matrices.append(Rhs_kkra)
            labels.append('KKRA')
        except Exception as e:
            print(f'KKRA had the following error: {e}')

        #try inter-microstate MFPT calculations
        
        #weighted-MFPT with eigendecomposition for computer inter-microstate MFPTs
        try:
            mfpt = ktn.get_MFPT_from_Kmat(Q)
            if theta:
                pt = ktn.get_approx_kells_cluster_passage_times(pi, commpi, mfpt)
            else:
                pt = ktn.get_kells_cluster_passage_times(pi, commpi, mfpt)
            Rhs_invert = spla.inv(pt)@(np.diag(1./commpi) - np.ones((ncomms,ncomms)))
            matrices.append(Rhs_invert)
            labels.append('PTinvert_eig')
        except Exception as e:
            print(f'Inversion of weighted-MFPTs had the following error: {e}')

        #weighted-MFPT with fundamental matrix for computing inter-microstate MFPTs
        try:
            mfpt = ktn.mfpt_from_correlation(Q, pi)
            if theta:
                pt = ktn.get_approx_kells_cluster_passage_times(pi, commpi, mfpt)
            else:
                pt = ktn.get_kells_cluster_passage_times(pi, commpi, mfpt)
            Rhs_invert = spla.inv(pt)@(np.diag(1./commpi) - np.ones((ncomms,ncomms)))
            matrices.append(Rhs_invert)
            labels.append('PTinvert_fund')
        except Exception as e:
            print(f'Inversion of weighted-MFPTs had the following error: {e}')

        #weighted-MFPT with GT for computing inter-microstate MFPTs
        try:
            mfpt, pi = pgt.get_intermicrostate_mfpts_GT(temp, data_path)
            if theta:
                pt = ktn.get_approx_kells_cluster_passage_times(pi, commpi, mfpt)
            else:
                pt = ktn.get_kells_cluster_passage_times(pi, commpi, mfpt)
            Rhs_invert = spla.inv(pt)@(np.diag(1./commpi) - np.ones((ncomms,ncomms)))
            matrices.append(Rhs_invert)
            labels.append('PTinvert_GT')
        except Exception as e:
            print(f'Inversion of weighted-MFPTs had the following error: {e}')
        
        #try:
        #    Rhs_solve = spla.solve(pt, np.diag(1.0/commpi) - np.ones((ncomms,ncomms)))
        #    matrices.append(Rhs_solve)
        #    labels.append('PTsolve')
        #except Exception as e:
        #    print(f'Linear solve with weighted-MFPTs had the following error: {e}')
        
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
            rK = R - np.diag(np.diag(R))
            escape_rates = -1*np.diag(R)
            D = np.diag(escape_rates)
            B = rK@np.diag(1./escape_rates)
            Acomm = 0
            Bcomm = 3
            MFPTAB, MFPTBA = pgt.compute_MFPTAB(Acomm, Bcomm, B, D, K=rK, dense=True)
            df[f'AB_{labels[i]}'] = [MFPTAB]
            df[f'BA_{labels[i]}'] = [MFPTBA]
        dfs.append(df)
    bigdf = pd.concat(dfs, ignore_index=True, sort=False)
    return bigdf

def plot_ratios_16state(df, log=True, excludeHS=False):
    """Plot ratio of MFPT on coarse-grained networks to exact MFPT on full network."""
    colors = sns.color_palette("Dark2", 5)
    df.replace([np.inf, -np.inf], np.nan)
    df2= df.sort_values('T')
    symbols = ['-s', '-o', '--^', '-x', '--x']
    rates = ['LEA', 'KKRA', 'HS', 'PTinvert', 'PTsolve']
    if excludeHS:
        symbols = ['-s', '-o', '--^']
        rates = ['LEA', 'PTinvert', 'PTsolve']
    labels = rates
    denom = 'MFPT'
    #first plot A<-B direction
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=[10, 4])
    for j, CG in enumerate(rates):
        #then only plot HSK for temperatures that are not NaN
        df2CG = df2[-df2[f'BA_{CG}'].isna()]
        ax.plot(1./df2CG['T'], df2CG[f'BA_{CG}']/df2CG[f'{denom}BA'],
                symbols[j], label=labels[j], color=colors[j], linewidth=1,
                markersize=4)
    ax.set_xlabel(r'$1/T$')
    if log:
        ax.set_yscale('log')
    ax.set_ylabel('Ratio MFPTBA')
    ax.legend(frameon=True)
    for j, CG in enumerate(rates):
        #then only plot HSK for temperatures that are not NaN
        df2CG = df2[-df2[f'AB_{CG}'].isna()]
        ax2.plot(1./df2CG['T'], df2CG[f'AB_{CG}']/df2CG[f'{denom}AB'],
                symbols[j], label=labels[j], color=colors[j], linewidth=1,
                markersize=4)
    ax2.set_xlabel(r'$1/T$')
    ax2.set_ylabel('Ratio MFPTAB')
    if log:
        ax2.set_yscale('log')
    ax2.legend(frameon=True)
    fig.tight_layout()
    #fig.subplots_adjust(left=0.12, top=0.97, right=0.99, bottom=0.11,
    #                    wspace=0.325)

def plot_mfpts_32state(df, insetdf=None, theta=False, mfpt_eig=False, log=True):
    """Plot MFPTs computed on coarse-grained networks against true MFPT from full network."""
    #colors = sns.color_palette("Dark2", 4)
    colors=[green, purple, orange, blue]
    order = [0, 1, 2, 3]
    df.replace([np.inf, -np.inf], np.nan)
    df2= df.sort_values('T')
    symbols = ['-s', '--o', '-o', '--^']
    if theta:
        labels = ['LEA', r'$\theta^{-1}$', 'KKRA', 'HS']
    else:
        labels = ['LEA', r'$\textbf{t}_{\rm{C}}^{-1}$', 'KKRA', 'HS']

    if mfpt_eig:
        colors=[green, 'b', purple, orange, blue]
        order = [0, 1, 2, 3, 4]
        symbols = ['-s', '-x', '--o', '-o', '--^']
        rates = ['LEA', 'PTinvert_eig', 'PTinvert_fund', 'KKRA', 'HS']
        labels = ['LEA', r'$\textbf{t}_{\rm{C}}^{-1}$ (eigendecomposition)', r'$\textbf{t}_{\rm{C}}^{-1}$ (fundamental matrix)', 'KKRA', 'HS']

    denom = 'MFPT'
    #first plot A<-B direction
    fig, ax = plt.subplots(figsize=[1.2*columnwidth, 0.95*columnwidth])
    ax.plot(1./df2['T'], df2['MFPTAB'], '-', color='k', label='Exact', lw=1, markersize=4)
    for j, CG in enumerate(rates):
        #then only plot HSK for temperatures that are not NaN
        df2CG = df2[-df2[f'AB_{CG}'].isna()]
        if CG == 'HS' or CG == 'KKRA':
            df2CG = df2CG[df2CG['T'] > 0.125]
        ax.plot(1./df2CG['T'], df2CG[f'AB_{CG}'],
                symbols[j], label=labels[j], color=colors[order[j]], linewidth=1,
                markersize=4)
    ax.set_xlabel(r'$1/T$')
    if log:
        ax.set_yscale('log')
    ax.set_ylabel('MFPT ($\mathcal{A} \leftarrow \mathcal{B}$)')
    #now plot inset with super low temps to show that t_c^-1 breaks
    if not mfpt_eig:
        ax.legend(loc=4, frameon=True)
        axins = inset_axes(ax, width="40%", height="40%", 
                        bbox_to_anchor=(.12, 0.12, 0.85, .85),
                        bbox_transform=ax.transAxes, loc=2)
        insetdf.replace([np.inf, -np.inf], np.nan)
        insetdf2  = insetdf.sort_values('T')
        insetdf2 = insetdf2[insetdf2['T'] > (1./160)]
        axins.plot(1./insetdf2['T'], insetdf2['MFPTAB'], '-', color='k', lw=1)
        for j, CG in enumerate(['LEA', 'PTinvert']):
            insetdf2CG = insetdf2[-insetdf2[f'AB_{CG}'].isna()]
            axins.plot(1./insetdf2CG['T'], insetdf2CG[f'AB_{CG}'],
                    symbols[j], label=labels[j], color=colors[order[j]], linewidth=1,
                    markersize=3)
        #axins.set_xlim([1./insetdf2['T'].max(), 155.])
        #axins.set_ylim([10**175])
        axins.tick_params('both', pad=1.5)
        axins.set_yscale('log')
    else:
        ax.legend(loc=2, frameon=True)
    fig.tight_layout()
    #plt.savefig('plots/hummer_szabo_32state.pdf')

def calculate_condition_numbers(invtemps=np.linspace(1, 40, 20), data_path=Path('KTN_data/32state')):
    # why is KKRA so bad? Let's plot the condition number of the 4-dimensional matrix that it inverts.
    dfs = []
    for temp in 1./invtemps:
        df = pd.DataFrame()
        df['T'] = [temp]
        beta = 1./temp
        ktn = Analyze_KTN(data_path, commdata='communities.dat')
        B, K, D, N, u, s, Emin, index_sel = kio.load_mat(path=data_path,beta=beta,Emax=None,Nmax=None,screen=False)
        Q = K.todense() - D.todense()
        mfpt, pi = pgt.get_intermicrostate_mfpts_GT(temp, data_path)
        #compute weighted-MFPT between communities
        commpi = ktn.get_comm_stat_probs(np.log(pi), log=False)
        pt = ktn.get_kells_cluster_passage_times(pi, commpi, mfpt)
        df['weighted-MFPT'] = [np.linalg.cond(pt)]
        #KKRA calculation
        N = len(ktn.communities)
        n = len(pi)
        D_N = np.diag(commpi)
        D_n = np.diag(pi)
        #construct clustering matrix M from community assignments
        M = np.zeros((n, N))
        for ci in ktn.communities:
            col = np.zeros((n,))
            comm_idxs = np.array(ktn.communities[ci]) - 1
            col[comm_idxs] = 1.0
            M[:, ci-1] = col

        Pi_col = commpi.reshape((N, 1))
        pi_col = pi.reshape((n, 1))
        KKRAmat = Pi_col@Pi_col.T + M.T@D_n@mfpt@pi_col@Pi_col.T - M.T@D_n@mfpt@D_n@M
        df['KKRA_invert'] = [np.linalg.cond(KKRAmat)]
        try:
            RKKRA = D_N@spla.inv(KKRAmat)
            df['KKRA_2nd_term'] = [np.linalg.cond(RKKRA)]
            RKKRA = Pi_col@np.ones((1,N)) - RKKRA
            df['KKRA_rates'] = [np.linalg.cond(RKKRA)]
        except Exception as e:
            df['KKRA_2nd_term'] = [np.nan]
            df['KKRA_rates'] = [np.nan]
        
        #fundamental matrix (in original Hummer-Szabo expression)
        fund_mat = pi_col@np.ones((1,n)) - Q
        df['Fund_matrix'] = [np.linalg.cond(fund_mat)]
        try:
            HSmat = M.T@spla.inv(fund_mat)@D_n@M
            df['HS_invert'] = [np.linalg.cond(HSmat)]
        except Exception as e:
            df['HS_invert'] = [np.nan]
        try:
            R_HS = D_N@spla.inv(HSmat)
            df['HS_2nd_term'] = [np.linalg.cond(HSmat)]
            R_HS = Pi_col@np.ones((1,N)) - R_HS
            df['HS_rates'] = [np.linalg.cond(R_HS)]
        except Exception as e:
            df['HS_2nd_term'] = [np.nan]
            df['HS_rates'] = [np.nan]
        dfs.append(df)
    bigdf = pd.concat(dfs, ignore_index=True, sort=False)
    return bigdf

def plot_condition_numbers(df):   
    #plot
    invtemps = 1./df['T']
    fig, ax = plt.subplots(figsize=[1.2*columnwidth, 0.9*columnwidth])
    colors = sns.color_palette("Dark2", 6)
    ax.plot(invtemps, df['weighted-MFPT'], 'o-', markersize=3, color=colors[0], label=r'$\textbf{t}_{\rm C}$')
    #ax.plot(invtemps, df['KKRA_invert'], 'o-', markersize=3, color=colors[1], label='KKRA invert')
    ax.plot(invtemps, df['KKRA_2nd_term'], 'o-', markersize=3, color=colors[2], label='matrix inverted \n in KKRA')
    #ax.plot(invtemps, df['HS_invert'], 'o-', markersize=3, color=colors[4], label='HS invert')
    #ax.plot(invtemps, df['HS_2nd_term'], 'o-', markersize=3, color=colors[5], label='HS 2nd term')
    ax.plot(invtemps, df['Fund_matrix'], 'o-', markersize=3, color=colors[3], label='matrix inverted \n in HS')
    ax.set_xlabel(r"$1/T$")
    ax.set_ylabel("Condition number")
    ax.set_yscale('log')
    #ax2.plot(invtemps, df['KKRA_invert'], 'o-', markersize=3, color=colors[1], label='KKRA invert')
    ax.plot(invtemps, df['KKRA_rates'], 'o-', markersize=3, color=colors[4], label=r'$\textbf{K}_{\rm C}$ (KKRA)')
    #ax2.plot(invtemps, df['HS_invert'], 'o-', markersize=3, color=colors[4], label='HS invert')
    ax.plot(invtemps, df['HS_rates'], 'o-', markersize=3, color=colors[1], label=r'$\textbf{K}_{\rm C}$ (HS)')
    #ax2.legend(loc=2)
    #ax2.set_xlabel(r"$1/T$")
    #ax2.set_yscale('log')
    ax.legend(loc=2)
    fig.tight_layout()
    plt.savefig('plots/condition_numbers_32state.pdf')

def compare_theta_approx(i, j, I, J, labels, invT = np.linspace(0.1, 50, 6), data_path=Path('KTN_data/32state'), approx=False, ax=None):
    """Plot PTAB vs. MFPT[min1, min2] to compare, where min1 is a minimum in A
    and min2 is a minimum in B."""
    comm = {0:'A', 1:'C', 2:'D', 3:'B'}
    ktn = Analyze_KTN(data_path, commdata='communities.dat')
    mfptAB = np.zeros_like(invT)
    mfptBA = np.zeros_like(invT)
    PTAB = np.zeros_like(invT)
    PTBA= np.zeros_like(invT)
    for k, invtemp in enumerate(invT):
        temp = 1./invtemp
        mfpt, pi = pgt.get_intermicrostate_mfpts_GT(temp, data_path)
        mfptAB[k] = mfpt[i, j]
        mfptBA[k] = mfpt[j, i]
        commpi = ktn.get_comm_stat_probs(np.log(pi), log=False)
        if approx:
            pt = ktn.get_approx_kells_cluster_passage_times(pi, commpi, mfpt)
        else:
            pt = ktn.get_kells_cluster_passage_times(pi, commpi, mfpt)
        PTAB[k] = pt[I,J]
        PTBA[k] = pt[J,I]
    if ax is None:
        fig, ax = plt.subplots(figsize=[textwidth/3, textwidth/3])
    ax.plot(invT, mfptAB/PTAB, '-o', label=labels[0], color=blue, lw=1, markersize=3.5)
    ax.plot(invT, mfptBA/PTBA, '-o', label=labels[1], color=orange, lw=1, markersize=3.5)
    ax.plot(invT, np.tile(1.0, len(invT)), '--', color=grey, lw=1)
    #ax.set_xlabel('1/T')
    ax.legend(loc=1, fontsize=7, frameon=False)
    return ax

def plot_theta_approx_panel(invT = np.linspace(0.1, 50, 10), data_path=Path('KTN_data/32state'), approx=False):
    """Plot all 6 inter-community weighted-MFPTs (theta)."""
    fig, axes = plt.subplots(2, 3, sharex='col', sharey='row', figsize=[0.66*textwidth, 1.15*textwidth/3])
    labels = np.array([[r'$\mathcal{T}_{ab}/[t_C]_{\mathcal{A}\mathcal{B}}$', r'$\mathcal{T}_{ac}/[t_C]_{\mathcal{A}\mathcal{C}}$', r'$\mathcal{T}_{a d}/[t_C]_{\mathcal{A}\mathcal{D}}$'],
              [r'$\mathcal{T}_{b a}/[t_C]_{\mathcal{B}\mathcal{A}}$', r'$\mathcal{T}_{c a}/[t_C]_{\mathcal{C}\mathcal{A}}$', r'$\mathcal{T}_{d a}/[t_C]_{\mathcal{D}\mathcal{A}}$']])
    labels2 = np.array([[r'$\mathcal{T}_{b c}/[t_C]_{\mathcal{B}\mathcal{C}}$', r'$\mathcal{T}_{b d}/[t_C]_{\mathcal{B}\mathcal{D}}$', r'$\mathcal{T}_{c d}/[t_C]_{\mathcal{C}\mathcal{D}}$'],
              [r'$\mathcal{T}_{c b}/[t_C]_{\mathcal{C}\mathcal{B}}$', r'$\mathcal{T}_{d b}/[t_C]_{\mathcal{D}\mathcal{B}}$', r'$\mathcal{T}_{d c}/[t_C]_{\mathcal{D}\mathcal{C}}$']])
    compare_theta_approx(0, 24, 0, 3, labels[:,0], ax=axes[0][0])
    compare_theta_approx(0, 8, 0, 1, labels[:,1], ax=axes[0][1])
    compare_theta_approx(0, 16, 0, 2, labels[:,2], ax=axes[0][2])
    compare_theta_approx(24, 8, 3, 1, labels2[:,0], ax=axes[1][0])
    compare_theta_approx(24, 16, 3, 2, labels2[:,1], ax=axes[1][1])
    compare_theta_approx(8, 16, 1, 2, labels2[:,2], ax=axes[1][2])
    axes[1][1].set_xlabel(r'$1/T$')
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.17, top=0.95, wspace=0.15, hspace=0.2)
    plt.savefig('plots/theta_approx_panel_32state.pdf')

