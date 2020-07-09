""" This script uses the Analyze_KTN class to benchmark mean first passage time
calculations on the unbranched nearest neighbor model first considered in Weiss
1967.

Deepti Kannan 2020 """

from ktn.code_wrapper import ParsedPathsample
from ktn.ktn_analysis import Analyze_KTN
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
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
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
path = Path('/Users/deepti/Documents/Wales/databases/chain/metastable')
#path = Path('/scratch/dk588/databases/chain/metastable')
#PATHSAMPLE = "/home/dk588/svn/PATHSAMPLE/build/gfortran/PATHSAMPLE"

"""Define variables needed to calculate rate as a function of
temperature."""

mindata = np.loadtxt(path/'min.data')
tsdata = np.loadtxt(path/'ts.data')

nmin = mindata.shape[0]
emin = mindata[:,0]
fvibmin = mindata[:, 1]
hordermin = mindata[:, 2]

ets = np.zeros((nmin, nmin))
fvibts = np.zeros((nmin, nmin))
horderts = np.ones((nmin, nmin))
exist = np.zeros((nmin, nmin))

for i in range(tsdata.shape[0]):
    j1 = int(tsdata[i, 3]) - 1
    j2 = int(tsdata[i, 4]) - 1
    exist[j1, j2] = 1
    exist[j2, j1] = 1
    ets[j1, j2] = tsdata[i, 0]
    ets[j2, j1] = tsdata[i, 0]
    fvibts[j1, j2] = tsdata[i, 1]
    fvibts[j2, j1] = tsdata[i, 1]
    horderts[j1, j2] = tsdata[i, 2]
    horderts[j2, j1] = tsdata[i, 2]

def Kmat(temp):
    """Return a rate matrix, nmin x nmin for specified temperature."""
    K = np.zeros((nmin, nmin), dtype=np.longdouble)
    for j in range(nmin):
        vib = np.exp((fvibmin - fvibts[:,j])/2).astype(np.longdouble)
        order = hordermin/(2*np.pi*horderts[:,j])
        nrg = np.exp(-(ets[:,j] - emin)/temp).astype(np.longdouble)
        K[:, j] = exist[:, j]*vib*order*nrg

    K = K.T
    for i in range(nmin):
        K[i, i] = -np.sum(K[:,i])
    #return transpose since ts.data assumes i->j and we want i<-j
    return K

def peq(temp):
    """Return equilibrium probabilities for specified temperature."""
    zvec = np.exp(-fvibmin/2)*np.exp(-emin/temp)/hordermin
    zvec = zvec.astype(np.longdouble)
    return zvec/np.sum(zvec)

def weiss(temp):
    """Return the matrix of mean first passage times using the recursive
    formulae in Weiss (1967) Adv. Chem. Phys. 13, 1-18."""
    K = Kmat(temp)
    def eta(j):
        if j == 0:
            return 0
        else:
            return (K[j, j-1]*eta(j-1) + 1)/K[j-1, j]

    def theta(j):
        if j==0:
            return 1
        else:
            return theta(j-1)*K[j, j-1]/K[j-1, j]

    etavec = [eta(j) for j in range(0, nmin-1)]
    thetavec = [theta(j) for j in range(0, nmin-1)]
    tmean_oneton = lambda n: (eta(n)/theta(n))*np.sum(thetavec[0:n]) - np.sum(etavec[0:n]) 

    def xeta(j):
        if j == nmin-1:
            return 0
        else:
            return (K[j, j+1]*xeta(j+1) + 1)/K[j+1, j]

    def xtheta(j):
        if j == nmin-1:
            return 1
        else:
            return xtheta(j+1)*K[j, j+1]/K[j+1, j]

    xetavec = [xeta(j) for j in range(1, nmin)]
    xthetavec = [xtheta(j) for j in range(1, nmin)]
    tmean_nmintoone = lambda n: (xeta(n)/xtheta(n))*np.sum(xthetavec[n:nmin-1]) - np.sum(xetavec[n:nmin-1]) 
    mfpt = np.zeros((nmin, nmin), dtype=np.longdouble)
    for i in range(0, nmin):
        for j in range(0, i):
            mfpt[i][j] = tmean_oneton(i) - tmean_oneton(j)
        for j in range(i+1, nmin):
            mfpt[i][j] = tmean_nmintoone(i) - tmean_nmintoone(j)
    return mfpt

def mfpt_from_correlation(temp):
    """Calculate the matrix of mean first passage times using Eq. (49) of KRA
    JCP paper. """
    pi = peq(temp)
    K = Kmat(temp)
    pioneK = spla.inv(pi.reshape((nmin,1))@np.ones((1,nmin)) + K)
    zvec = np.diag(pioneK)
    mfpt = np.diag(1./pi)@(pioneK - zvec.reshape((nmin,1))@np.ones((1,nmin)))
    return mfpt

def mfpt_between_states_GT_eigen(i, j, temps):
    """Calculate the i <-> j passage times using graph transformation and the
    WAITPDF keyword in the PATHSAMPLE program."""

    parse = ParsedPathsample(path/'pathdata')
    files_to_modify = [path/'min.A', path/'min.B']
    for f in files_to_modify:
        if not f.exists():
            print(f'File {f} does not exists')
            raise FileNotFoundError 
        os.system(f'mv {f} {f}.original')
    
    dfs = []
    for temp in temps:
        df = pd.DataFrame()
        df['T'] = [temp]
        #first get t( i<->j) from GT
        parse.append_input('NGT', '0 T')
        parse.comment_input('WAITPDF')
        parse.append_input('TEMPERATURE', f'{temp}')
        parse.write_input(path/'pathdata')
        parse.minA = [i] 
        parse.numInA = 1
        parse.minB = [j]
        parse.numInB = 1
        parse.write_minA_minB(path/'min.A', path/'min.B')
        #run PATHSAMPLE
        outfile_name = path/f'out.{i+1}.{j+1}.T{temp}'
        outfile = open(outfile_name, 'w')
        subprocess.run(f"{PATHSAMPLE}", stderr=subprocess.STDOUT, stdout=outfile, cwd=path)
        #parse output
        parse.parse_output(outfile=path/f'out.{i+1}.{j+1}.T{temp}')
        df[f'tGT{i}{j}'] = [parse.output['MFPTAB']]
        df[f'tGT{j}{i}'] = [parse.output['MFPTBA']]

        #then re-run PATHSAMPLE to get t*(i<->j) with eigendecomposition
        parse.comment_input('NGT')
        parse.append_input('WAITPDF', '')
        parse.write_input(path/'pathdata')
        outfile = open(parse.path/f'out.T{temp:.1f}.waitpdf','w')
        try:
            subprocess.run(f"{PATHSAMPLE}", stderr=subprocess.STDOUT,
                            stdout=outfile, cwd=path, timeout=5)
        except subprocess.TimeoutExpired:
            print('WAITPDF expired 5s timeout. Setting tau*AB and' +
                    ' tau*BA to NaN')
        #parse output
        parse.parse_output(outfile=parse.path/f'out.T{temp:.1f}.waitpdf')
        if 'tau*AB' in parse.output:
            df[f't*{i}{j}'] = [parse.output['tau*AB']]
        else:
            df[f't*{i}{j}'] = np.nan
        if 'tau*BA' in parse.output:
            df[f't*{j}{i}'] = [parse.output['tau*BA']]
        else:
            df[f't*{j}{i}'] = np.nan
        dfs.append(df)
    bigdf = pd.concat(dfs, ignore_index=True, sort=False)
    bigdf.to_csv(f'csvs/weiss_GT_eigendecomposition_mfpt{i}{j}.csv')
    #restore original min.A and min.B files
    for f in files_to_modify:
        os.system(f'mv {f}.original {f}')


def compare_weiss_pt(i, j, I, J, approx=False, ax=None):
    """Plot PTAB vs. weiss[min1, min2] to compare, where min1 is a minimum in A
    and min2 is a minimum in B."""
    comm = {0:'A', 1:'I', 2:'B'}
    ktn = Analyze_KTN('/Users/deepti/Documents/Wales/databases/chain/metastable', communities
                      = {1:[1,2,3], 2:[4,5,6,7,8], 3:[9,10,11]})
    invT = np.linspace(0.001, 5, 100)
    weissAB = np.zeros_like(invT)
    weissBA = np.zeros_like(invT)
    PTAB = np.zeros_like(invT)
    PTBA= np.zeros_like(invT)
    for k, invtemp in enumerate(invT):
        mfpt_weiss = weiss(1./invtemp)
        weissAB[k] = mfpt_weiss[i, j]
        weissBA[k] = mfpt_weiss[j, i]
        pi = peq(1./invtemp)
        commpi = ktn.get_comm_stat_probs(np.log(pi), log=False)
        if approx:
            pt = ktn.get_approx_kells_cluster_passage_times(pi, commpi, mfpt_weiss)
        else:
            pt = ktn.get_kells_cluster_passage_times(pi, commpi, mfpt_weiss)
        PTAB[k] = pt[I,J]
        PTBA[k] = pt[J,I]
    if ax is None:
        fig, ax = plt.subplots(figsize=[textwidth_inches/3, textwidth_inches/3])
    ax.plot(invT, weissAB/PTAB,
            label=f't$_{{{i}\leftarrow{j}}}$/PT({comm[J]}$\leftarrow${comm[I]})')
    ax.plot(invT, weissBA/PTBA,
            label=f't$_{{{j}\leftarrow{i}}}$/PT({comm[J]}$\leftarrow${comm[I]})')
    ax.set_xlabel('1/T')
    ax.legend()
    return ax
    #fig.tight_layout()

def plot_weiss_PT_comparison_panel():
    """Compare the mfpt between two states against the full cluster-cluster
    passage time defined in Kells paper. 3 different examples are shown, one
    for AB, one for AI, and one that only compares the first term of the
    passage time which we call theta."""

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=[textwidth_inches,
                                                      textwidth_inches/3])

    comm = {0:'A', 1:'I', 2:'B'}
    ktn = Analyze_KTN('/Users/deepti/Documents/Wales/databases/chain/metastable', communities
                      = {1:[1,2,3], 2:[4,5,6,7,8], 3:[9,10,11]})
    invT = np.linspace(0.001, 5, 100)
    weiss19 = np.zeros_like(invT)
    weiss91 = np.zeros_like(invT)
    weiss210 = np.zeros_like(invT)
    weiss92 = np.zeros_like(invT)
    weiss15 = np.zeros_like(invT)
    weiss51 = np.zeros_like(invT)
    PTAB = np.zeros_like(invT)
    PTBA= np.zeros_like(invT)
    PTAI = np.zeros_like(invT)
    PTIA= np.zeros_like(invT)
    thetaAB = np.zeros_like(invT)
    thetaBA= np.zeros_like(invT)
    for k, invtemp in enumerate(invT):
        mfpt_weiss = weiss(1./invtemp)
        weiss19[k] = mfpt_weiss[1, 9]
        weiss91[k] = mfpt_weiss[9, 1]
        weiss210[k] = mfpt_weiss[2, 10]
        weiss92[k] = mfpt_weiss[9, 2]
        weiss15[k] = mfpt_weiss[1, 5]
        weiss51[k] = mfpt_weiss[5, 1]
        pi = peq(1./invtemp)
        commpi = ktn.get_comm_stat_probs(np.log(pi), log=False)
        pt = ktn.get_kells_cluster_passage_times(pi, commpi, mfpt_weiss)
        PTAB[k] = pt[0,2]
        PTBA[k] = pt[2,0]
        PTAI[k] = pt[0,1]
        PTIA[k] = pt[1,0]
        theta = ktn.get_approx_kells_cluster_passage_times(pi, commpi, mfpt_weiss)
        thetaAB[k] = theta[0,2]
        thetaBA[k] = theta[2,0]
    #plot t(2<->10)/theta(A<->B)
    ax.plot(invT, weiss19/thetaAB, 
            label=r'$t_{2\leftarrow 10}/{\theta}_{A\leftarrow B}$')
    ax.plot(invT, weiss91/thetaBA, 
            label=r'$t_{10\leftarrow 2}/{\theta}_{B\leftarrow A}$')
    ax.legend()
    #plot t(3<->10)/PT(A<->B)
    ax2.plot(invT, weiss210/PTAB,
            label=r'$t_{3\leftarrow 11}/{\bf t}^{c}_{A\leftarrow B}$')
    ax2.plot(invT, weiss92/PTBA,
            label=r'$t_{10\leftarrow 3}/{\bf t}^{c}_{B\leftarrow A}$')
    ax2.set_xlabel('1/T')
    ax2.legend()
    #plot t(2<->6)/PT(A<->I)
    ax3.plot(invT, weiss15/PTAI,
            label=r'$t_{2\leftarrow 6}/{\bf t}^{c}_{A\leftarrow I}$')
    ax3.plot(invT, weiss51/PTIA,
            label=r'$t_{6\leftarrow 2}/{\bf t}^{c}_{I\leftarrow A}$')
    ax3.legend()
    fig.tight_layout()
    plt.savefig('plots/weiss_PT_mfpt_approx_panel.pdf')

def plot_weiss_landscape_mfpt_benchmark():
    """Plot energy of minima and transition states as a function of the 11
    states."""
    
    stat_pts = np.zeros((2*len(emin) - 1, )) 
    stat_pts[::2] = emin
    stat_pts[1:-1:2] = tsdata[:,0]
    states = np.arange(1, 11.5, 0.5)
    xrange = np.linspace(1, 11, 1000)
    cs = CubicSpline(states, stat_pts, bc_type='clamped')
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=[columnwidth,
                                                 columnwidth],
                                        gridspec_kw={'height_ratios': [1, 1.6]})
    ax.plot(xrange, cs(xrange), 'k')
    ax.plot(states[2], stat_pts[2], 'ro')
    ax.plot(states[[0, 4, 16, 20]], stat_pts[[0, 4, 16, 20]], 'ko')
    ax.plot(states[6:16:2], stat_pts[6:16:2], 'ko')
    ax.plot(states[18], stat_pts[18], 'ro')
    ax.set_xticks(np.arange(1, 12, 1))
    ax.set_xlabel('States')
    ax.set_ylabel('Energy')

    """ Plot 4 different calculations of mfpt: weiss analytical answer,
    eigendecomposition, GT, Kells inversion. """

    ktn = Analyze_KTN('/scratch/dk588/databases/chain/metastable', communities
                      = {1:[1,2,3], 2:[4,5,6,7,8], 3:[9,10,11]})
    invT = np.linspace(0.01, 43, 1000)
    weissBA = np.zeros_like(invT)
    weissAB = np.zeros_like(invT)
    df = pd.read_csv('csvs/weiss_GT_eigendecomposition_mfpt19.csv')
    df = df.drop(columns=['Unnamed: 0'])
    temps = df['T']
    corrBA = np.zeros_like(temps)
    corrAB = np.zeros_like(temps)
    n=11
    i=1
    j=9
    for k, invtemp in enumerate(invT):
        mfpt_weiss = weiss(1./invtemp)
        weissAB[k] = mfpt_weiss[i, j]
        weissBA[k] = mfpt_weiss[j, i]

    for k, T in enumerate(temps):
        try:
            mfpt_corr = mfpt_from_correlation(T)
            corrAB[k] = mfpt_corr[i, j]
            corrBA[k] = mfpt_corr[j, i]
        except:
            corrBA[k] = np.nan
        """
        try:
            K = Kmat(T)
            eigenAB[k] = -spla.solve(K[np.arange(n)!=i, :][:, np.arange(n)!=i],
                                    (np.arange(n)==j)[np.arange(n)!=i]).sum()
            eigenBA[k] = -spla.solve(K[np.arange(n)!=j, :][:, np.arange(n)!=j],
                                    (np.arange(n)==i)[np.arange(n)!=j]).sum()
        except:
            eigenAB[k] = np.nan
            eigenBA[k] = np.nan
        """

    #10 <- 2 direction (BA)
    ax2.plot(invT, weissBA, 'k', label='Weiss')
    ax2.plot(1./temps, df['t*91'], 'o', markersize=11, alpha=0.3,
            markeredgewidth=0, label='Eigendecomposition')
    ax2.plot(1./temps, corrBA, 'bo', markersize=5, markeredgewidth=0.25,
            markeredgecolor='k', label='Fundamental matrix')
    ax2.plot(1./temps, df['tGT91'], 'kx', markersize=3, label='GT')
    ax2.set_xlabel('1/T')
    ax2.set_ylabel(r'$\mathcal{T}_{10 \leftarrow 2}$')
    ax2.set_yscale('log')
    ax2.legend()
    #plt.subplots_adjust(left=0.06, bottom=0.13, top=0.97, right=0.99)
    plt.subplots_adjust(left=0.20, right=0.95, top=0.97, bottom=0.13, hspace=0.35)
    plt.savefig('plots/weiss_mfpt_landscape.pdf')
    #fig.tight_layout()

def plot_mfpt_benchmark():

    """ Plot 4 different calculations of mfpt: weiss analytical answer,
    eigendecomposition, GT, Kells inversion. """

    ktn = Analyze_KTN('/scratch/dk588/databases/chain/metastable', communities
                      = {1:[1,2,3], 2:[4,5,6,7,8], 3:[9,10,11]})
    invT = np.linspace(0.01, 43, 1000)
    weissBA = np.zeros_like(invT)
    weissAB = np.zeros_like(invT)
    df = pd.read_csv('csvs/weiss_GT_eigendecomposition_mfpt19.csv')
    df = df.drop(columns=['Unnamed: 0'])
    temps = df['T']
    corrBA = np.zeros_like(temps)
    corrAB = np.zeros_like(temps)
    n=11
    i=1
    j=9
    for k, invtemp in enumerate(invT):
        mfpt_weiss = weiss(1./invtemp)
        weissAB[k] = mfpt_weiss[i, j]
        weissBA[k] = mfpt_weiss[j, i]

    for k, T in enumerate(temps):
        try:
            mfpt_corr = mfpt_from_correlation(T)
            corrAB[k] = mfpt_corr[i, j]
            corrBA[k] = mfpt_corr[j, i]
        except:
            corrBA[k] = np.nan
        """
        try:
            K = Kmat(T)
            eigenAB[k] = -spla.solve(K[np.arange(n)!=i, :][:, np.arange(n)!=i],
                                    (np.arange(n)==j)[np.arange(n)!=i]).sum()
            eigenBA[k] = -spla.solve(K[np.arange(n)!=j, :][:, np.arange(n)!=j],
                                    (np.arange(n)==i)[np.arange(n)!=j]).sum()
        except:
            eigenAB[k] = np.nan
            eigenBA[k] = np.nan
        """

    #2 <- 10 direction (AB)
    fig, ax = plt.subplots(figsize=[columnwidth, 2*columnwidth/3])
    ax.plot(invT, weissAB, 'k', label='Weiss')
    ax.plot(1./temps, df['t*19'], 'o', markersize=11, alpha=0.3,
            markeredgewidth=0, label='Eigendecomposition')
    ax.plot(1./temps, corrAB, 'bo', markersize=5, markeredgewidth=0.25,
            markeredgecolor='k', label='Fundamental matrix')
    ax.plot(1./temps, df['tGT19'], 'kx', markersize=6, label='GT')
    ax.set_xlabel('1/T')
    ax.set_ylabel(r'$\mathcal{T}_{2 \leftarrow 10}$')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
