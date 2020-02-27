import numpy as np
import time,os
from tqdm import tqdm
np.set_printoptions(linewidth=160)
from lib.ktn_io import *
from lib.gt_tools import *
from scipy.sparse import save_npz,load_npz, diags, eye
from scipy.sparse.linalg import eigs,inv,spsolve
import scipy as sp
import scipy.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Do we force the generation of the KTN from data, or see if it is in the "cache"
generate = True

# where is the data
data_path = "KTN_data/LJ38/4k/"

observables = []


# inverse temperature range
for beta in np.linspace(1.0,20.,20):#np.linspace(0.01,5.0,30):#np.linspace(2.5,9.0,2):#np.linspace(0.01,5.0,30):#
    beta, B, K, D, N, u, s, kt, kcon, Emin, index_sel = load_save_mat(path=data_path,beta=beta,Emax=None,Nmax=None,generate=generate,screen=True)
    D = np.ravel(K.sum(axis=0))
    BF = beta*u-s

    """
    Boolean vectors selecting A and/or B regions
    """
    A_states,B_states = load_AB(data_path,index_sel)


    basins = B_states + A_states

    inter_region = ~basins

    print("\n\t%d A STATES <-> %d B STATES\n" % (A_states.sum(),B_states.sum()))
    print("\n\tbeta:",beta," N:",N,"N_TS:",K.data.size,"\n\n")

    """
        graph transformation to remove intermediate states
    """

    trmb = 10

    rB, rD, rK, rN, retry = gt_seq(N=N,rm_reg=inter_region,B=B,D=D,trmb=trmb,retK=True)

    r_A_states = A_states[~inter_region]
    r_B_states = B_states[~inter_region]

    oneA = np.ones(r_A_states.sum())
    oneB = np.ones(r_B_states.sum())
    r_BF = BF[~inter_region]

    res = np.zeros(5)
    res[0] = beta


    """
        calculate rates from direct matrix,
    """

    for si,s_r_s in enumerate([[A_states,r_A_states],[B_states,r_B_states]]):
        print(s_r_s[0].sum(),"STATES ->",rN-s_r_s[0].sum(),"STATES")

        """ with no GT """
        n_r_s = np.zeros(N,bool)
        n_r_s[s_r_s[0]] = True
        n_r_s[inter_region] = True

        rho = np.zeros(N)
        rho[s_r_s[0]] = np.exp(-BF[s_r_s[0]])
        rho = rho[n_r_s]
        rho /= rho.sum()

        dtau = np.ravel(spsolve((eye(N)-B)[n_r_s,:][:,n_r_s].transpose(),1.0/kt[n_r_s]))

        res[1+2*si+0] = (1.0/dtau).dot(rho)
        res[1+2*si+1] = 1.0/(dtau.dot(rho))
        print("no GT,  :",res[2*si+1:2*(si+1)+1])



        """ with intermediate states GT'ed away """
        r_s = s_r_s[1]
        rho = np.exp(-r_BF[r_s])
        rho /= rho.sum()
        dtau = np.ravel(spsolve(rK[r_s,:][:,r_s].transpose(),np.ones(r_s.sum())))
        res[1+2*si+0] = (1.0/dtau).dot(rho)
        res[1+2*si+1] = 1.0/(dtau.dot(rho))
        print("GT-I, kF, 1/tau:",res[2*si+1:2*(si+1)+1])


        """ with final states GT'ed to a single state """
        rm_reg = np.ones(rN,bool)
        rm_reg[(~r_s).nonzero()[0][0]] = False # (~r_s).nonzero()[0] == final states, don't remove only the first
        rm_reg[r_s.nonzero()[0]] = False # (~r_s).nonzero()[0] == initial states, keep all
        rrB, rrD, rrK, rrN, retry = gt_seq(N=rN,rm_reg=rm_reg,B=rB,D=rD,trmb=1,Ndense=1,retK=True)
        rr_s = r_s[~rm_reg]

        dtau = np.ravel(spsolve(rrK[rr_s,:][:,rr_s].transpose(),np.ones(rr_s.sum())))

        res[1+2*si+0] = (1.0/dtau).dot(rho)
        res[1+2*si+1] = 1.0/(dtau.dot(rho))
        print("GT-(I+final), kF, 1/tau:",res[2*si+1:2*(si+1)+1])




        """ GT'ed to 2 state system for each initial state:"""
        tau_gt = np.zeros(r_s.nonzero()[0].size)
        pbar = tqdm(total=tau_gt.size,leave=False,mininterval=0.0)
        for str in range(tau_gt.size):
            rm_reg = np.ones(rN,bool)
            rm_reg[(~r_s).nonzero()[0][0]] = False # (~r_s).nonzero()[0] == final states, don't remove only the first
            rm_reg[r_s.nonzero()[0][str]] = False # (~r_s).nonzero()[0] == initial states, don't remove only the "strth"
            bs=1
            if rm_reg.sum()>10:
                bs=10
            rrB, rrD, rrK, rrN, retry = gt_seq(N=rN,rm_reg=rm_reg,B=rB,D=rD,trmb=bs,Ndense=1,retK=True)
            rr_s = r_s[~rm_reg]
            print(rr_s)
            print(rrK.todense())
            tau_gt[str] = np.ravel(spsolve(rrK[rr_s,:][:,rr_s].transpose(),np.ones(rr_s.sum())))
            print(tau_gt[str])
            pbar.update(1)
        pbar.close()
        res[1+2*si+0] = rho.dot(1.0/tau_gt)
        res[1+2*si+1] = 1.0/(tau_gt.dot(rho))
        print("GT-(I+final+initial), kF, 1/tau:",res[2*si+1:2*(si+1)+1])


        print("\n---\n")
    print("\n*******\n")
    observables.append(res)

    print(observables[-1])

ob = np.r_[observables]


fig, ax = plt.subplots(1,1,figsize=(8,6),dpi=100)

ll = [r"$k^F_\mathcal{A\leftarrow B}$",r"$k^*_\mathcal{A\leftarrow B}$",\
        #r"$k^{SS}_\mathcal{A\leftarrow B}$",r"$k^{NSS}_\mathcal{A\leftarrow B}$"]
        r"$k^F_\mathcal{B\leftarrow A}$",r"$k^*_\mathcal{B\leftarrow A}$"]

for j in range(4):
    ax.semilogy(ob[:,0],ob[:,1+j],'o-',label=ll[j])
ax.legend()
plt.show()







"""


    r_B_states.copy()
    rm_re

    C_AB = np.ravel(rB[r_A_states,:][:,r_B_states].sum(axis=0))
    C_BA = np.ravel(rB[r_B_states,:][:,r_A_states].sum(axis=0))

    C_AB_D = C_AB.dot(D.data[B_states])
    C_BA_D = C_BA.dot(D.data[A_states])

    C_AB_rD = C_AB.dot(rD[B_states])
    C_BA_rD = C_BA.dot(rD[A_states])

    l_AB, v_AB = eigs(K_AB)

    K_AB = rK[r_B_states,:][:,r_B_states]

    C_AB_D = np.ravel(C_AB.dot(D[B_states,:][:,B_states].todense()))
    C_AB_rD = np.ravel(C_AB.dot(diags(rD[r_B_states]).todense()))

    l_AB, v_AB = spla.eig(K_AB.todense(),right=True)

    qsdo = np.abs(l_AB.real).argsort()

    nu = l_AB.real[qsdo]
    #print("GAP:",nu[1]/nu[0])
    qsd = v_AB[:,qsdo[0]].real
    qsd /= qsd.sum()

    uB = u[B_states]-u[B_states].min()
    sB = s[B_states]-s[B_states].min()
    rho = np.exp(-uB*beta+sB)
    rho /= rho.sum()


    kqsd = C_AB_rD.dot(qsd)



    T_AB = np.ravel(spsolve(K_AB.transpose(),np.ones(rho.size)))#  iK_BA.sum(axis=0)
    trho = T_AB.dot(rho)
    tqsd = T_AB.dot(qsd)
    k_F = (1.0/T_AB).dot(rho)

    observables.append([beta,kqsd,\
                        C_AB_D.dot(rho),C_AB_rD.dot(rho),\
                        nu[0],k_F,\
                        1.0/trho,1.0/tqsd
                        ])

    print(observables[-1])

    K_BA = rK[r_A_states,:][:,r_A_states]

    C_BA = np.ravel(rB[r_B_states,:][:,r_A_states].sum(axis=0))

    C_BA_rD = np.ravel(C_BA.dot(diags(rD[r_A_states]).todense()))

    l_BA, v_BA = spla.eig(K_BA.todense(),right=True)



    qsdo = np.abs(l_BA.real).argsort()

    nu = l_BA.real[qsdo]
    #print("GAP:",nu[1]/nu[0])

    qsd = v_BA[:,qsdo[0]].real
    qsd /= qsd.sum()

    uA = u[A_states]-u[A_states].min()
    sA = s[A_states]-s[A_states].min()
    rho = np.exp(-uA*beta+sA)
    rho /= rho.sum()

    C_BA_D = np.ravel(C_BA.dot(D[A_states,:][:,A_states].todense()))

    kqsd = C_BA_rD.dot(qsd)

    #iK_BA = spla.inv(K_BA.todense())
    T_BA = spsolve(K_BA.transpose(),np.ones(rho.size))#  iK_BA.sum(axis=0)
    trho = T_BA.dot(rho)
    tqsd = T_BA.dot(qsd)
    k_F = (1.0/T_BA).dot(rho)



    #rr_rm_reg = r_B_states.copy()
    #rr_rm_reg[r_B_states.nonzero()[0][-1]] = False
    #rr_rm_reg[0] = False
    #rrB, rrD, rrN, retry = gt_seq(N=rN,rm_reg=rr_rm_reg,B=rB,D=rD,trmb=1)
    #print(rrD[r_A_states[~rr_rm_reg]])
    #rrK = diags(rrD)-rrB.dot(diags(rrD))
    robservables.append([beta,kqsd,\
                        C_BA_D.dot(rho),C_BA_rD.dot(rho),\
                        nu[0],k_F,\
                        1.0/trho,1.0/tqsd
                        ])

    print(robservables[-1])

exit()

# kSS, kNSS, nu_0, k_F, trho, tqsd

ll = [r"$k_{QSD}$",r"$k_{SS}$",r"$k_{NSS}$",r"$\nu_0$",r"$k_F$",r"$1/\langle\tau|\pi\rangle$",r"$1/\langle\tau|\pi_{QSD}\rangle$"]

ob = [np.r_[observables],np.r_[robservables]]

fig, ax = plt.subplots(1,1,figsize=(8,6),dpi=100)


for i in range(2):
    print(ob[i][-1][0],ob[i][-1,6])
    ob[i][:,1:] = np.log(ob[i][:,1:])#-np.log(ob[i][:,1][0])
    #ob[i][:,6:8] = np.log(1.0/ob[i][:,6:8])#-np.log(ob[i][:,1][0])
    for j in [4,5]:
        ax.plot(ob[i][:,0],ob[i][:,1+j],'o-',label=ll[j])
    #for j in range(7):
    #    ax[0].plot(ob[i][:,0],ob[i][:,1+j],'o-',label=ll[j])

    #ax[i].set_yscale("log")
    ax.legend()
plt.show()
"""
