import matplotlib.pyplot as plt
import numpy as np

name='4k_proc'
beta=10

sws = [1]

savefig = True

root_data = np.loadtxt('output/pab_converge_%s' % name)
title = "LJ38 test"

#raw_data = np.loadtxt('output/%s/dneb_ip' % name)
filename = 'output/%s/dneb_ip_gs.pdf' % (name)

title = r"LJ$_{38}$ cluster, $\beta=$%d, Initial Path: Simulated $\tt{DNEB}$" % (beta)

""" plotting (a bit messy!) """



fig, axs = plt.subplots(1,1,figsize=(6,4),sharex=True,dpi=120)
ax = axs
axs.set_title(title,fontsize=10)
axs.set_xlabel("DNEB search coverage [%]")

#for [ax,sw] in zip(axs,sws):
for sw in sws:
	#ax.set_ylabel(r" (Est. ${\rm C}^\mathcal{A}_\mathcal{B}$ - True ${\rm C}^\mathcal{A}_\mathcal{B}$) / (True ${\rm C}^\mathcal{A}_\mathcal{B}$)")

	ax.set_ylabel(r"Estimated ${\rm C}^\mathcal{A}_\mathcal{B}$ [True ${\rm C}^\mathcal{A}_\mathcal{B}$]")

	raw_data = root_data.copy()
	raw_data = raw_data[raw_data[:,3]>0.0]

	"""
	0-3: iteration nrp probe_compl ebab
	4-7: TotalMaxMin +/- TotalSparseMaxMin +/-
	8-11: ExpectMaxMin +/- ExpectMaxMaxMin +/-
	12-13: SingleMaxMin +/-
	14: Sparsity
	15: bab
	"""
	if sw>1:
		data = raw_data.copy()[sw-1:,:]
		for _sw in range(1,sw):
			for ii in range(3,15):
				data[:,ii] = np.vstack((
				np.abs(data[:,ii]),
				np.abs(raw_data.copy()[sw-1-_sw:-_sw,ii]))).T.min(axis=1)
		for ii in [5,7,9,11,13]:
			data[:,ii] *= -1.0 # for sign
	else:
		data = raw_data.copy()





	"""
	0-3: iteration nrp probe_compl ebab
	4-7: TotalMaxMin +/- TotalSparseMaxMin +/-
	8-11: ExpectMaxMin +/- ExpectMaxMaxMin +/-
	12-13: SingleMaxMin +/-
	14: Sparsity
	15: bab
	"""

	"""
	format- all divided by b_real.
	np.log(b/br) -> np.log(b) +/- db/b
	"""
	"""
	Positive dB -> (1-B)(1-e[-dB/(1-B)]) => 1+dB/B = 1/B-(1/B-1)e[-dB/(1-B)]
	Negative dB -> B(e[dB/B]-1) => 1+dB/B = e[dB/B]
	"""

	obs = [6,12,8]
	sbl=[]
	sl=[]

	sb=[]
	s=[]

	el = r"${\rm C}^\mathcal{A}_\mathcal{B}$ (Est.)"
	tl = r"${\rm C}^\mathcal{A}_\mathcal{B}$ (True)"

	#sbl.append(r"${\rm C}^\mathcal{A}_\mathcal{B}\pm\sigma^{\rm tot}_\pm$")
	#sl.append(r"${\rm C}^\mathcal{A}_\mathcal{B}+\sigma^{\rm tot}$")

	sbl.append(r"${\rm C}^\mathcal{A}_\mathcal{B}\pm\xi\sigma^{\rm tot}_\pm$")
	sl.append(r"${\rm C}^\mathcal{A}_\mathcal{B}+\xi\sigma^{\rm tot}$")

	sbl.append(r"${\rm C}^\mathcal{A}_\mathcal{B}\pm{\sigma}^{1}_\pm$")
	sl.append(r"${\rm C}^\mathcal{A}_\mathcal{B}+{\sigma}^{1}$")

	sbl.append(r"${\rm C}^\mathcal{A}_\mathcal{B}\pm{\sigma}^\xi_\pm$")
	sl.append(r"${\rm C}^\mathcal{A}_\mathcal{B}+{\sigma}^\xi$")

	#sbl.append(r"${\rm C}^\mathcal{A}_\mathcal{B}\pm{\sigma}^{1,\xi}_\pm$")
	#sl.append(r"${\rm C}^\mathcal{A}_\mathcal{B}+{\sigma}^{1,\xi}$")

	for iii in obs:
		data[:,iii] = (1.0-data[:,3].copy())*(1.-np.exp(-data[:,iii]/(1.0-data[:,3].copy())))
		data[:,iii+1] = data[:,3].copy()*(np.exp(data[:,iii+1]/data[:,3].copy())-1.0)

	bab = data[:,-1][-1].copy()
	for iii in obs:
		data[:,iii] /= data[:,-1]
		data[:,iii+1] /= data[:,-1]
	data[:,3] /= data[:,-1]
	data[:,-1] /= data[:,-1]
	xa = data[:,2] * 100.0


	t, = ax.plot(xa,data[:,-1],'k--',lw=1)#),label=

	for ci,iii in enumerate(obs):
		sb.append(ax.fill_between(xa,data[:,3]+data[:,iii],data[:,3]+data[:,iii+1],alpha=0.5,color='C'+str(ci+1),lw=1,ls='--'))

	for ci,iii in enumerate(obs):
		s.append(ax.plot(xa,data[:,3]+data[:,iii]+data[:,iii+1],'C%s-' % str(ci+1),lw=2)[0])

	e, = ax.plot(xa,data[:,3],'k-',lw=2)

	ax.set_yscale('log')
	ax.set_ylim(bab,1.0/bab)
	ax.set_xlim(xa.min(),5.0)#xa.max())

handles=[e,t]
labels = [el,tl]
for ssi in range(len(sb)):
	handles.append(s[ssi])
	handles.append(sb[ssi])
	labels.append(sl[ssi])
	labels.append(sbl[ssi])
ax.legend(handles,labels,loc='lower center',ncol=4, columnspacing=1.0, borderaxespad=0.,fontsize=10) # mode="expand",


plt.tight_layout()

fig.subplots_adjust(hspace=0.4)

if savefig:
	plt.savefig(filename,dpi=400)

plt.show()
