import matplotlib.pyplot as plt
import numpy as np

name='LJ13'
beta=10

sws = [1,4]

savefig = True

root_data = np.loadtxt('output/pab_converge_%s' % name)
title = "LJ38 test"

#raw_data = np.loadtxt('output/%s/dneb_ip' % name)
filename = 'output/%s/dneb_ip_gs.pdf' % (name)

sNEB = 1608.0
npairs = 10
Nt=1506
title = r"LJ$_{13}$ cluster, $\beta=$%d, Initial Path: Simulated $\tt{DNEB}$" % (beta)

#sNEB=200.
#npairs=50
#Nt = 1250
#title = r"LJ$_{38}$ cluster (GT reduced to %d states), $\beta=$%d, Initial Path: Simulated $\tt{DNEB}$" % (Nt,beta)

"""
raw_data = np.loadtxt('output/%s/lsad_ip' % name)
filename = 'output/%s_lsad_ip.pdf' % (name)
title = r"LJ$_{38}$ cluster (subset), $\beta=$%d, Initial Path: Minimal Saddle Point" % (beta)
"""

""" plotting (a bit messy!) """



fig, axs = plt.subplots(2,1,figsize=(6,6),sharex=True)
axs[0].set_title(title,fontsize=10)
axs[1].set_xlabel("DNEB search coverage [%]")

for [ax,sw] in zip(axs,sws):
	#ax.set_ylabel(r" (Est. ${\rm C}^\mathcal{A}_\mathcal{B}$ - True ${\rm C}^\mathcal{A}_\mathcal{B}$) / (True ${\rm C}^\mathcal{A}_\mathcal{B}$)")

	ax.set_ylabel(r"Estimated ${\rm C}^\mathcal{A}_\mathcal{B}$ [True ${\rm C}^\mathcal{A}_\mathcal{B}$]")

	raw_data = root_data.copy()
	raw_data = raw_data[raw_data[:,3]>0.0]

	"""
	0-4: ii,nrp,pc, ebab,bab
	5 : <sparsity>
	6-7 : sigma^1 +/-
	8-10 : <dc+>,<dc->,<dc> xi/(1-xi)
	11-12 : sigma_tot +/- * xi
	13-15: gsigma +,-,.
	"""
	if sw>1:
		data = raw_data.copy()[sw-1:,:]
		for _sw in range(1,sw):
		  for ii in [3,6,7,8,9,10,11,12,13,14,15]:
		    data[:,ii] = np.vstack((
			np.abs(data[:,ii]),
			np.abs(raw_data.copy()[sw-1-_sw:-_sw,ii]))
			).T.min(axis=1)
		for ii in [7,9,12,14]:
			data[:,ii] *= -1.0 # for sign
	else:
		data = raw_data.copy()

	for i in [3,6,7,8,9,10,11,12,13,14,15]:
	    data[:,i] *= 1.0/data[:,4]

	#for i in [8,9,10,11,12]:
	#    data[:,i] *= 1.0/data[:,5]


	#xa = (data[:,2]-0.0*data[:,2][0])*100.0
	#xa = np.arange(len(data[:,2])) * 20.0*100.0/1250.0/1250.0
	xa = (sNEB + np.arange(len(data[:,2])) * 2.0 * npairs)*100.0/Nt/Nt
	#rxa = (1608. + np.arange(len(raw_data[:,2])) * 20.0)*100.0/1506./1506.
#(data[:,2][1]-data[:,2][0])*100.0
	#shift=0.0
	#data[:,3] -= shift

	sl = r"${\rm C}^\mathcal{A}_\mathcal{B}$ (Est.)"

	tl = r"${\rm C}^\mathcal{A}_\mathcal{B}$ (True)"
	t, = ax.plot(xa,data[:,4]/data[:,4],'k--',lw=1)#),label=


	sobl = r"${\rm C}^\mathcal{A}_\mathcal{B}\pm\sigma^1_\pm$"
	shbl = r"${\rm C}^\mathcal{A}_\mathcal{B}\pm{\sigma}^\xi_\pm$"
	stbl = r"${\rm C}^\mathcal{A}_\mathcal{B}\pm\sigma^{\rm tot}_\pm$"
	sxtbl = r"${\rm C}^\mathcal{A}_\mathcal{B}\pm\xi\sigma^{\rm tot}_\pm$"

	stb = ax.fill_between(xa,data[:,3]+data[:,12],data[:,3]+data[:,11],alpha=0.5,color='C1',lw=1,ls='--')
	sxtb = ax.fill_between(xa,data[:,3]+data[:,11]*data[:,5],data[:,3]+data[:,12]*data[:,5],alpha=0.5,color='C5',lw=1,ls='--')
	shb = ax.fill_between(xa,data[:,3]+data[:,9],data[:,3]+data[:,8],alpha=0.5,color='C2',lw=1,ls='--')
	sob = ax.fill_between(xa,data[:,3]+data[:,7],data[:,3]+data[:,6],alpha=0.5,color='C3',lw=1,ls='--')
	#sgb = ax.fill_between(xa,data[:,3]+data[:,13],data[:,3]+data[:,14],alpha=0.5,color='C4',lw=1,ls='--')

	sol = r"${\rm C}^\mathcal{A}_\mathcal{B}+{\sigma}^1$"
	shl = r"${\rm C}^\mathcal{A}_\mathcal{B}+{\sigma}^\xi$"
	stl = r"${\rm C}^\mathcal{A}_\mathcal{B}+\sigma^{\rm tot}$"
	sxtl = r"${\rm C}^\mathcal{A}_\mathcal{B}+\xi\sigma^{\rm tot}$"
	sgl = r"${\rm C}^\mathcal{A}_\mathcal{B}+\exp(\langle\log\sigma\rangle)$"

	st, = ax.plot(xa,data[:,3]+data[:,11]+data[:,12],'C1-',lw=2)

	sxt, = ax.plot(xa,data[:,3]+(data[:,11]+data[:,12])*data[:,5],'C4-',lw=2)
	sh, = ax.plot(xa,data[:,3]+data[:,10],'C2-',lw=2)
	so, = ax.plot(xa,data[:,3]+data[:,6]+data[:,7],'C3-',lw=2)

	s, = ax.plot(xa,data[:,3],'k-',lw=2)
	#sg, = ax.plot(xa,data[:,3]+data[:,15],'C4o-',lw=2)



	#ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylim(ymin=1.0e-1,ymax=1.0e5)
	ax.set_xlim(xa.min(),xa.max())
	if sw==1:
	  sstring = "No Smoothing"
	else:
		sstring = "Smoothing Window=%d" % sw
	ax.text(0.99,0.1,sstring,fontsize=10, horizontalalignment='right',verticalalignment='top',
			transform=ax.transAxes,bbox={'facecolor':'white', 'alpha':0.5, 'pad':1, 'edgecolor':'white'})

#h, l = axs[1].get_legend_handles_labels()
#print(l)
#ord = [5,4,1,3,2,0,8,7,6]
#ord = [4,3,0,2,1,0,7,6]

axs[1].legend([s,t,sxt,sxtb,st,stb,sh,shb,so,sob],[sl,tl,sxtl,sxtbl,stl,stbl,shl,shbl,sol,sobl],
				bbox_to_anchor=(-.1, .825, 1.1, .75), loc='center',
     			ncol=5, mode="expand", borderaxespad=0.,fontsize=10)
#fontsize=10,ncol=4,bbox_to_anchor=(1., -0.25),loc='lower middle')

#axs[1].legend(loc='lower center', bbox_to_anchor=(0.5, -.4),ncol=2)

plt.tight_layout()
fig.subplots_adjust(hspace=0.4)

if savefig:
	plt.savefig(filename,dpi=400)

plt.show()
