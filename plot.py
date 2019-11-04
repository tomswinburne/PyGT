import matplotlib.pyplot as plt
import numpy as np

#true_bab = 6.10047539246064e-07
data = np.loadtxt('output/pab_converge')

smoothing_window = 4

inset = False
savefig=False
if inset:
	fig, ax = plt.subplots(1,1,figsize=(5,3))
else:
	fig, ax = plt.subplots(1,1,figsize=(8,6))

raw_data = data.copy()
data = raw_data.copy()[smoothing_window-1:,:]
for _smoothing_window in range(1,smoothing_window):
  for ii in [4,5,8,9]:
    data[:,ii] = np.vstack((np.abs(data[:,ii]),np.abs(raw_data.copy()[smoothing_window-1-_smoothing_window:-_smoothing_window,ii]))).T.min(axis=1) * (1.0-2.0*(ii%2))

data[:,10] = data[:,8] + data[:,9]
for ii in [4,5,8,9,10]:
  data[:,ii] *= 1.0-data[:,11]
for i in range(8):
	data[:,i+3] *= 1.0/data[:,-1]

xa = data[:,2]*100.0

data[:,3] -= 1.0

#ax.fill_between(xa,data[:,3]+data[:,6],data[:,3]+data[:,7],label=r"Raw Confidence Bounds",alpha=0.1,color='C2',lw=1,ls='--')
#ax.fill_between(xa,data[:,3]+data[:,8]*2.0,data[:,3]+data[:,9]*2.0,label=r"Senstivity Expectation Bounds ($\pm2\sigma_\pm$)",alpha=0.1,color='C1',lw=1,ls='--')
ax.fill_between(xa,data[:,3]+data[:,8],data[:,3]+data[:,9],label=r"Senstivity Expectation Bounds ($\pm\sigma_\pm$)",alpha=0.3,color='C1',lw=1,ls='--')
ax.fill_between(xa,data[:,3]+data[:,4],data[:,3]+data[:,5],label=r"Max Single Transition Bounds",alpha=0.5,color='C0',lw=1,ls='--')



ax.plot(xa,data[:,3]+data[:,10],'C1o-',lw=2,label=r"Projected $\langle{\rm B}_\mathcal{AB}\rangle+\sigma_+-\sigma_-$")
ax.plot(xa,data[:,3],'ko-',lw=3,label=r"Estimated $\langle{\rm B}_\mathcal{AB}\rangle$")

#ax.plot(xa,xa*0.0,'k--',lw=2,label=r"True ${\rm B}_\mathcal{AB}$")
true_bab = data[:,-1].mean()
if inset:
	ax.set_ylim(-0.1,0.1)
else:
	ax.set_ylim(-1.0,9.0)

if inset:
	ax.set_xlim(float(int(xa.max()-0.5)),xa.max())
else:
	ax.set_xlim(xa.min() -0.05*(xa.max()-xa.min()),xa.max())

ax.set_yscale('symlog',ylinthresh=.01)

if inset:
	ax.set_yticks([-0.1,0.0,0.1])
	ax.set_yticklabels(["-0.1","0.0","0.1"])
else:
	ax.set_yticks([-0.9,-0.5,0.0,1.0,9.0])
	ax.set_yticklabels(["-0.9","-0.5","0.0","1.0","9.0"])


if not inset:
	ax.set_title(r"Branching Probability Estimation with LJ13 (Smoothing Window=%d)" % smoothing_window)
	ax.set_xlabel("DNEB search coverage [%]")
	ax.set_ylabel(r"Relative Branching Prob. Error")
	ax.legend(loc='upper right')


plt.tight_layout()
if savefig:
	if inset:
		plt.savefig('results/LJ13_50000_test_%d_batch_NOSS_INSET.png' % smoothing_window,dpi=400)
	else:
		plt.savefig('results/LJ13_50000_test_%d_batch_NOSS.png' % smoothing_window,dpi=400)
plt.show()
