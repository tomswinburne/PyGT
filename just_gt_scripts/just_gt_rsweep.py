import numpy as np
import time
np.set_printoptions(linewidth=160)
from lib.ktn_io import *
from lib.gt_tools import *
from scipy.sparse import save_npz,load_npz, diags
from scipy.sparse.linalg import eigs,inv
import scipy as sp
import matplotlib.pyplot as plt



print("\n\nGT REGULARIZATION TESTS\n")

# Do we generate the KTN from data, or read in the "cache"
generate = 0

# Do we try a brute solve?
brute = True

betar = [10.0]#range(1,16)

for _beta in betar:

    Emax = None#-167.5
    beta, B, K, D, N, u, s, kt, kcon, Emin = load_save_mat(path="KTN_data/LJ38/",\
        beta=_beta,Emax=Emax,Nmax=3000,generate=generate)
    f = u - s/beta

    """
    Boolean vectors selecting A and/or B regions
    """
    initial_states, final_states = np.zeros(N,bool), np.zeros(N,bool)

    initial_states[np.loadtxt('KTN_data/LJ38/min_oct').astype(int)-1] = True
    final_states[np.loadtxt('KTN_data/LJ38/min_ico').astype(int)-1] = True

    basins = initial_states + final_states
    inter_region = ~basins

    rho_A, rho_B = np.zeros(N), np.zeros(N)

    rho_B[initial_states] = np.exp(-f[initial_states]*beta)
    rho_B = rho_B[~final_states]/rho_B[~final_states].sum()

    rho_A[final_states] = np.exp(-f[final_states]*beta)
    rho_A = rho_A[~initial_states]/rho_A[~initial_states].sum()


    #print("\n%d INITIAL STATES -> %d FINAL STATES\n" % (initial_states.sum(),final_states.sum()))
    #print("beta: ",beta,"N: ",N)
    out = output_str()
    #out(["\n%d PATH+ENV STATES\n" % ((~inter_region).sum())])
    print("\n\tbeta:",beta)
    if brute:

        K_AB = (D-K)[~final_states,:][:,~final_states]
        l_AB, v_AB = eigs(K_AB, 2, sigma=0, which='LM',maxiter=1000)
        ll_AB, w_AB = eigs(K_AB.transpose(), 1, sigma=0, which='LM',maxiter=5000)

        qsdi = np.abs(l_AB.real).argmin()
        qsdil = np.abs(ll_AB.real).argmin()
        print("\n\tRelative Spectral Gap, nu_1/nu_0, nu_0, nu_0:",\
            l_AB[1-qsdi].real/l_AB[qsdi].real,l_AB[qsdi].real,ll_AB[qsdil].real)


        tau_AB = spsolve(K_AB,rho_B).sum()
        cprob = w_AB[:,qsdil].dot(rho_B) * v_AB[:,qsdi].sum() / w_AB[:,qsdil].dot(v_AB[:,qsdi])
        #print("\n\tQSD.rho/w.v :", (w_AB[:,qsdil].dot(rho_B) / w_AB[:,qsdil].dot(v_AB[:,qsdi])).real)
        #print("\n\tRatio:",l_AB[qsdi].real * tau_AB, l_AB[qsdi].real * tau_AB/cprob.real)
        print("\n\t1/nu_0, tau_AB, P_QSD, tau_AB*nu_0/P_QSD :",\
            1.0/l_AB[qsdi].real, tau_AB, cprob.real, tau_AB*l_AB[qsdi].real/cprob.real)
        print("\n\t")

    final_print = ""
    for trmb in [100,40]:
        t = time.time()

        # first, large GT
        #inter_region[inter_region.nonzero()[0][:31]]=False

        rB, rD, rN, retry = gt_seq(N=N,rm_reg=inter_region,B=B,D=D.data,trmb=trmb)

        r_initial_states = initial_states[~inter_region]
        r_final_states = final_states[~inter_region]

        K_AB = (diags(rD)-rB.dot(diags(rD)))[~r_final_states,:][:,~r_final_states]

        l_AB, v_AB = eigs(K_AB, 2, sigma=0, which='LM',maxiter=1000)
        ll_AB, w_AB = eigs(K_AB.transpose(), 1, sigma=0, which='LM',maxiter=5000)
        qsdi = np.abs(l_AB.real).argmin()
        qsdil = np.abs(ll_AB.real).argmin()

        rho_B = np.zeros(rN)
        rho_B[r_initial_states] = np.exp(-f[~inter_region][r_initial_states]*beta)
        rho_B = rho_B[~r_final_states]/rho_B[~r_final_states].sum()

        print("\n\tRelative Spectral Gap, nu_1/nu_0, nu_0, nu_0:",\
            l_AB[1-qsdi].real/l_AB[qsdi].real,l_AB[qsdi].real,ll_AB[qsdil].real)

        tau_AB = spsolve(K_AB,rho_B).sum()
        cprob = w_AB[:,qsdil].dot(rho_B) * v_AB[:,qsdi].sum() / w_AB[:,qsdil].dot(v_AB[:,qsdi])
        print("\n\t1/nu_0, tau_AB, P_QSD, tau_AB*nu_0/P_QSD :",\
            1.0/l_AB[qsdi].real, tau_AB, cprob.real, tau_AB*l_AB[qsdi].real/cprob.real)
        print("\n\t\n\t")
        exit()






        # direct solve on retained states
        BABI, BAB = direct_solve(rB,r_initial_states,r_final_states)
        out(["\nGT[%d] justpath:" % trmb,"B(A<-B):",BABI+BAB,"B(AB):",BAB,"B(AIB):",BABI, "RESCANS: ",retry,"TIME:",time.time()-t,"max(diag(BII))):",rB.diagonal().max(),"\n"])

        basins = r_initial_states + r_final_states
        rm_reg= ~basins
        if rm_reg.sum()>0:
            rB, rD, rN, retry = gt_seq(N=rN,rm_reg=(~basins),B=rB,D=rD,trmb=1)
            r_initial_states = r_initial_states[basins]
            r_final_states = r_final_states[basins]

        oneB = np.ones(r_initial_states.sum())
        BABM = rB[r_final_states,:].transpose()[r_initial_states,:].transpose()
        BAB = BABM.dot(oneB).sum()
        BABI = 0.0

        out(["\nGT[%d] complete:" % trmb,"B(A<-B):",BAB,"RESCANS: ",retry,"TIME:",time.time()-t,"\n"])


    out.summary()

    print("\n\n")
exit()
