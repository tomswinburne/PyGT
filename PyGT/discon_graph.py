import numpy as np
from scipy.sparse import csr_matrix, csgraph
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.cm as cm
import pickle,time
from tqdm import tqdm

class Tree(object):
    def __init__(self,value=None,branches=None,depth=None):
        self.branches = branches
        self.value = value
        self.E = None
        self.weight = None
        self.minE = None
        self.depth = depth
        self.id = None

    def print_tree(self):
        for l in range(self.depth):
            print("\t")
        if not self.branches is None:
            w = 0
            for b in self.branches:
                w += b.weight
            print(self.value,"(",self.depth,",",self.weight,w,self.minE,")")
            for b in self.branches:
                b.print_tree()
        else:
            if self.E > self.value:
                print("WHAT",)
            print(self.value,"(",self.depth,",",self.weight,1,self.minE,")")
            #print self.value,self.E,"(",level,self.weight,")"

class rate_network:
    def __init__(self,file=None,efile=None,crank=5.,bins=20,E=None,nuE=None,slope=0.0, MBM=None,blank=False):
        if not blank:
            """ Load prefactor / barrier matrix """
            if nuE is None:
                if file is None:
                    print("Specify nuE or file!")
                    exit()
                nuE = np.loadtxt(file,dtype={'names': ('I', 'F', 'nu', 'E'),'formats': (np.uint64, np.uint64, np.float,np.float)})

            nst = int(nuE['I'].max())+1
            print(nst)
            print("MAXE:", nuE['E'].max(),nuE['E'].min())
            self.N = nst
            self.slope = slope
            print(self.N)

            """ Find relative state energies, E[0] == 0.0 """
            self.E = np.zeros(nst)
            filled = np.zeros(nst,bool)
            if E is None:
                if efile is None:
                    self.E[0] = 0.
                    filled[0] = True
                    sweeps=0
                    while not filled.min():
                        sweeps += 1
                        #print(sweeps,filled.sum())
                        for ist in range(nst):
                            if not filled[ist]:
                                continue
                            conn = nuE[nuE['I']==ist]['F']
                            for fst in conn:
                                self.E[fst] = self.E[ist]
                                f_list = nuE[nuE['I']==ist]
                                f_list = f_list[f_list['F']==fst]['E']
                                b_list = nuE[nuE['F']==ist]
                                b_list = b_list[b_list['I']==fst]['E']
                                self.E[fst] += nuE[nuE['I']==ist][nuE[nuE['I']==ist]['F']==fst]['E'][0]
                                self.E[fst] -= nuE[nuE['F']==ist][nuE[nuE['F']==ist]['I']==fst]['E'][0]
                                filled[fst] = True
                    print("E production took %d sweeps" % sweeps,"( minimum state energy:",self.E.min(),")")
                    self.Emin = self.E.min()
                else:
                    for line in open(efile):

                        _d = line.strip().split(" ")
                        #print(_d)
                        _id = int(_d[0])
                        self.E[_id] = float(_d[1])
                        filled[_id] = True
                    self.Emin = self.E.min()
            else:
                self.E = E
                self.Emin = self.E.min()


            """ NumPy transition state energy matrix  (+ expo cranked ~ inf norm on sum) """
            self.E -= self.Emin

            TE = np.zeros((nst,nst))
            eTE = np.zeros((nst,nst))

            for conn in nuE:
                ist = conn['I']
                fst = conn['F']
                TSE = conn['E'] + self.E[ist]
                if conn['E'] + self.E[ist] < self.E[fst] :
                    print("PP",ist,fst,conn['E'], self.E[ist],self.E[fst])
                TE[ist][fst] = TSE
                eTE[ist][fst] = np.exp(crank * TSE)

            print("HIGHEST ENERGY TS:",TE.max(),"(SHIFTED BY ",-self.Emin," eV)")
            print("LOWEST ENERGY TS:",TE[TE!=0.].min(),self.E.max())

            print("ZERO ROWS?", (TE.sum(axis=1)==0.).sum())

            self.B = TE


            self.MTSE = TE.max() + .1

            """ nx Graph """
            #G = nx.from_scipy_sparse_matrix(csr_matrix(eTE))

            """ Find minimum barrier between states """
            if MBM is None:
                time.clock()
                seTE = csr_matrix(eTE)
                self.seTE = seTE
                print("SP",time.clock())
                _MB = csgraph.shortest_path(seTE,directed=False)
                #print("CC",time.clock())
                self.MB = np.log( np.where(_MB>1.,_MB,1.) ) / crank
                #np.savetxt("MB.np",self.MB,fmt='%.4g')

                err = self.MB + np.identity(self.MB.shape[0])*10.-np.outer(self.E,np.ones(self.MB.shape[0]))
            else:
                self.MB = MBM

            self.MB = np.where(self.MB>1.e-5,self.MB,0.) #nx.read_gexf(MBM)
            self.threshes = np.linspace(self.MTSE,min(self.MB.min(),self.E.min()),bins)
            #np.append(np.r_[[self.MTSE,self.MTSE*2./3.]],)
            self.tree = None


    """ recursively build disjoint subgroups and fit into tree structure """

    def find_split(self,M,tree,_E,depth,ithresh=0):
        ds_sg = []
        tree.value = self.threshes[ithresh] #  where tree splits
        for thresh in self.threshes[ithresh:]:
            _M = np.where(M<thresh,M,0.)
            nc,cc = csgraph.connected_components(csr_matrix(_M))
            if nc>1:
                ds_sg = []
                for j in range(nc):
                    ds_sg.append([_M[cc==j,:][:,cc==j],_E[cc==j,:]])
                break
            tree.value = thresh
            ithresh += 1
        #if nc > 1:
        #    print thresh,nc
        tree.weight = M.shape[0] # number of states beneath
        tree.depth = depth + 1

        if len(ds_sg) > 1:
            """ recursive over connected subgraphs """
            tree.minE = _E[:,0].min()
            for bME in ds_sg:
                tree.branches.append(Tree())
                tree.branches[-1].branches = []
                self.find_split(bME[0],tree.branches[-1],bME[1],tree.depth,ithresh=ithresh)
        else:
            """ just a single node """
            for thresh in self.threshes:
                if thresh > _E[:,0].min():
                    tree.value = thresh
                else:
                    break
            tree.minE = _E[:,0].min()
            tree.branches = None
            #tree.weight = 1
            tree.E = _E[:,0].min()
            tree.id = int(_E[:,1][_E[:,0].argmin()])
            self.pbar.update(1)



    def build_tree(self):
        print(self.MTSE,self.E.min())

        tree = Tree(self.MTSE,[Tree()])
        tree.weight = self.N
        tree.minE = self.E.min()
        tree.depth = 0

        tree.branches[0].branches=[]
        EID = np.vstack((self.E,np.arange(len(self.E))))
        self.node_count = 0
        self.pbar = tqdm(total=self.N,mininterval=0,leave=False)
        self.find_split(self.MB,tree.branches[0],EID.T,tree.depth)
        self.tree = tree
        self.pbar.close()


    def build_lines(self,lines,tags,state_lines,tree,ctopx,ctopy):
        if not tree.branches is None:
            if tree.value < tree.minE:
                print("NOOO",tree.id,tree.value,tree.minE)
            segs = len(tree.branches)
            if segs==1:
                print(tree.id)
            # distribute equally over [-W/2,W/2] == find center of mass for each segment
            xcom = ctopx - tree.weight/2.
            xp = np.zeros(segs)
            xv = np.zeros(segs)

            for i in range(segs):
                xv[i] = tree.branches[i].value

            br_order = np.argsort(xv)
            br_order[-len(xv)//2:] = np.argsort(xv)[::2]
            br_order[:len(xv)//2] = np.argsort(xv)[1:][::2][::-1]



            xp[-1] = ctopx - .5*tree.weight
            pw = 0.

            for i in range(segs):
                b = tree.branches[br_order[i]]
                xp[i] = xp[i-1] + .5*(pw + b.weight)
                pw = b.weight
                dy = b.value - ctopy
                if b.value > ctopy:
                    print("NOOOO!!!!",b.id,b.value,b.minE,ctopy)
                else:
                    lines.append([(ctopx,ctopy),(xp[i],ctopy + self.slope*dy)])
                    lines.append([(xp[i],ctopy + self.slope*dy),(xp[i],b.value)])
                if b.branches is None:
                    self.build_lines(lines,tags,state_lines,b,xp[i],ctopy)
                else:
                    self.build_lines(lines,tags,state_lines,b,xp[i],b.value)

        else:
            if tree.minE > ctopy:
                print("!!!",tree.E,tree.value,tree.weight)
            lines.append([(ctopx,ctopy),(ctopx,tree.minE)])
            tags.append([ctopx,tree.minE,tree.id])
            state_lines.append([(ctopx,ctopy),(ctopx,tree.minE),tree.id])




    def plot(self,w=4,h=3,fn="dcg.pdf",gc=None,dump=True,linefiles=None,points=None,sels=None):


        if linefiles is None:
            lines = []
            state_lines = []
            tags = []
            self.build_lines(lines,tags,state_lines,self.tree,0.,self.tree.value)

            if dump:
                with open('dcg_lines', 'wb') as fp:
                    pickle.dump(lines, fp)
                    print("Dumped lines as dcg_lines")
                with open('dcg_tags', 'wb') as fp:
                    pickle.dump(tags, fp)
                    print("Dumped tags as dcg_tags")
                with open('dcg_state_lines', 'wb') as fp:
                    pickle.dump(state_lines, fp)
                    print("Dumped state_lines as dcg_state_lines")

        else:
            with open (linefiles[0], 'rb') as fp:
                itemlist = pickle.load(fp)
                lines = itemlist

            with open (linefiles[1], 'rb') as fp:
                itemlist = pickle.load(fp)
                tags = itemlist

            with open (linefiles[2], 'rb') as fp:
                itemlist = pickle.load(fp)
                state_lines = itemlist
        if gc is None:
            gc = (0.0,0.0,0.0,1)

        lc = mc.LineCollection(lines, linewidths=0.8, colors=[gc for l in lines])

        maxy = 0.
        for l in lines:
            if l[0][1] > maxy:
                maxy = l[0][1]

        fig,ax = plt.subplots(1,1,figsize=(w,h),dpi=120)
        ax.add_collection(lc)
        if not sels is None:
            for sel in sels:
                sl=[]
                for tsl in state_lines:
                    if tsl[2] in sel[0]:
                        sl.append([tsl[0],tsl[1]])
                ax.add_collection(mc.LineCollection(sl, linewidths=2.0, colors=sel[1],label=sel[2]))

        print(tags[0])
        print(np.r_[tags].shape)

        if not points is None:
            plt.plot(points[:,0],points[:,1],'ro')

        ax.autoscale()
        ax.margins(0.01)
        ax.set_ylim(-maxy*0.1,maxy*1.1)
        plt.ylabel(r'Relative Energy [$\epsilon$]')
        ax.set_xticks([], [])
        ax.legend(loc="upper right")
        plt.tight_layout()
        if not fn is None:
            plt.savefig(fn)
        else:
            plt.show()


        """
        f = open('tags','w')
        for t in tags:
            f.write(str(t[0])+" "+str(t[2])+" "+str(t[1])+" "+str(inv_hash_map[t[2]])+" "+str(t[1]+RM.Emin)+"\n")
        f.close()
        """
