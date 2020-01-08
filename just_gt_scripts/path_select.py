
"""
Find fastest path from state_state to end_state, then returns all states on path and 'depth' connections away
depth=1 => path_region =  all direct connections to the path
"""
path, path_region = make_fastest_path(sys.udG,sys.pi.argmax(),sys.pi.argmin(),depth=4,limit=6)
#path, path_region = make_fastest_path(sys.K,0,6,depth=3,limit=3)
print("NPathRegion=",path_region.sum())
"""
Boolean vectors selecting A and/or B regions
"""
selB = np.zeros(sys.N,bool)
selA = np.zeros(sys.N,bool)

selA[aind] = True
selB[bind] = True

selB[path[0]] = True
selA[path[-1]] = True


""" Build B -> A regions via boolean vectors"""
nbasin=2
cB = sys.ConnectingStates(path[0]) # index, rate
for ii in np.r_[cB[1]].argsort()[::-1][:nbasin]: # <=nbasin connecting states with highest rate
	selB[cB[0][ii]]=True

cA = sys.ConnectingStates(path[-1]) # index, rate
for ii in np.r_[cA[1]].argsort()[::-1][:nbasin]: # <=nbasin connecting states with highest rate
	selA[cA[0][ii]]=True

""" ensure A,B are disjoint """
for ii in np.arange(sys.N)[selB]:
	selA[ii] = False
print("DISJOINT? : ",((selA*selB).sum()==0))
sys.define_AB_regions(selA,selB)

sampler = sampler(sys)
#sampler.initial_sample()
sampler.initial_sample_path_region(np.arange(sys.N)[path_region],ncs=100)
