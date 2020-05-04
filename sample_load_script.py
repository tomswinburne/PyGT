import glob
import numpy as np
from scipy.sparse.linalg import eigs
import scipy.sparse as scsp
from scipy.linalg import eig,eigvals
"""
Load NxN matrix of rates with ZERO diagonal
"""
for f in glob.glob("LJ38_10k_K_*.npz"):
  K = scsp.load_npz(f)
  # extract smallest e-value
  w= eigvals(K.todense())
  print(w)
  #Ko = K.tocoo()
  #np.savetxt(f.split(".")[0]+".txt",np.vstack((Ko.row, Ko.col, Ko.data)).T,fmt="%d %d %20.20g")


"""
Diagonal matrix of total rate
"""
D = scsp.diags(np.ravel(K.sum(axis=0)))

