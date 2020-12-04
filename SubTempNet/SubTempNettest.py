"""needed Packages"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy
import scipy.optimize as opt
import pickle
from IPython.display import clear_output
plt.rcParams.update({'legend.fontsize': 'x-large',
          'figure.figsize': (6, 4),
         'axes.labelsize': '26',
         'axes.titlesize':'32',
         'xtick.labelsize':'26',
         'ytick.labelsize':'26',
         'mathtext.fontset':'stix',
         'font.family':'STIXGeneral'
         })
class SubTempNet(dict):
    def __init__(self, filename, objname, directed = False):
        """Simulator Class:
        input: filename / path of data, name to store object
        Storage -> key is length of timewindow
        does: """
        self["filename"] = filename
        self["objname"] = objname+"_SubTempNet"
        self["directed"] = directed
        #if init:
        A, t_to_snapshot, T, Tcomp, ecount, ncount, active_edges = self.make_A(filename, directed)
        self["A"]=A
        self["t_to_snapshot"] = t_to_snapshot
        self["ecount"] = ecount
        self["ncount"] = ncount
        self["active edges"] = active_edges
        self["T"] = T
        self["Tcomp"] = Tcomp
        self["deg_seq"] = self.deg_seq(A)
        t, self["casual paths"] = self.unfold_accessibility(A, track = True)
        self["cA0AT"] = {} #cA0AT
        self["cA0AL"] = {} #cA0AL   
        self["PA0"]={}
        self["PAT"]={}
        self["PAT2"]={}
        self["PAT4"]={}
        self["PAT8"]={}
    def drop_A(self):
        del self["A"]
        return
    def load_A(self):
        self["A"],_,_,_,_,_,_= self.make_A(self["filename"], self["directed"])
        return
    def save(self):
        dic = self.get_dictionary()
        self.save_obj(dic, self["objname"])
    def load(self):
        dic = self.load_obj(self["objname"])
        self.update(dic)
        return
    def __enter__(self):
        self.load_A()
        return self
    def __exit__(self, type, value, traceback):
        self.drop_A()
        return      
    def run(self, *T, maxsamp = 50, minsamp = 5):
        done =[]
        for t in T:
            if t ==1:
                self["PAT"][t]=[self["ncount"]]
                self["PAT2"][t]=[self["ncount"]]
                self["PAT4"][t]=[self["ncount"]]
                self["PAT8"][t]=[self["ncount"]]
                self["PA0"][t]=[self["ncount"]]
                done.append(t)
                continue
            self["PAT"][t]=[]
            self["PAT2"][t]=[]
            self["PAT4"][t]=[]
            self["PAT8"][t]=[]
            self["PA0"][t]=[]
            samples = self.slice_time(t, START =0 , END = self["T"], maxsamp = maxsamp, minsamp = minsamp, realT= True)
            samplenum = 0
            for samplestart, sampleend in samples:
                if (samplestart and sampleend) is None: #no active timesteps in sample
                    continue
                first = self.t_to_next_active(samplestart) #time of first active snapshot in sample
                last = self.t_to_last_active(sampleend-1) #time of last active snapshot in sample
                if first > sampleend or last<samplestart: #sample does not contain any snapshots
                    continue
                start = self["t_to_snapshot"][first] #first active snapshot in sample
                end = self["t_to_snapshot"][last]+1 #last active snapshot in sample
                final = self["t_to_snapshot"][last]+1
                samplenum +=1
                #PA0 calculates accessibility of original temporal network sample
                self["PA0"][t].append(self.unfold_accessibility(self["A"][start:end]).nnz)
                
                #PAT calculates accessibility of fully aggregated network sample
                self["PAT"][t].append(self.accessibility(self.aggregate_Matrices(self["A"][start:end]), cutoff = t).nnz)
                                
                #PATk
                for k in [2,4,8]:
                    l = t//k
                    #print("t=",t,"k=",k,"l=",l)
                    #initialize sub-slice interval generator
                    slices, length= self.subslice_time(l, samplestart, sampleend)
                    AL = [scipy.sparse.identity(self["A"][0].shape[0], dtype=np.int32)]
                    for start, end in slices:
                        if (start and end) is None: #no active timesteps in slice
                            continue
                        else:
                            lastend = end
                        AL.append(self.accessibility(self.aggregate_Matrices(self["A"][start:end]), cutoff = l))
                        
                        if len(AL)>1:
                            AL=[self.unfold_accessibility(AL)]
                    if final >lastend:        
                        AL.append(self.accessibility(self.aggregate_Matrices(self["A"][lastend:final]), cutoff = l))
                    self["PAT"+str(k)][t].append(self.unfold_accessibility(AL).nnz)  
                    
                    #print status update
                    clear_output()
                    print("Done with samplelengths ",done)
                    print("Analyzing sample number",samplenum, "for samplelength ", t, "and ",k, "slices") 
            done.append(t)
    def slice_time(self, val, START = 0, END = -1,maxsamp = -1, minsamp = 5, realT = False):
        if END == -1:
            END = self["T"]
        if maxsamp == -1:
            maxsamp = 100
        T = END-START
        def intervals(steps =1):
            intlen = val
            start_t = START #start in original time dimension
            end_t = start_t + val
            while end_t <= END:
                if realT:
                    yield start_t, end_t
                    start_t += int(steps *val)
                    end_t += int(steps *val)
                    continue
                first = self.t_to_next_active(start_t) #time of first active snapshot in interval
                last = self.t_to_last_active(end_t-1) #time of last active snapshot in interval
                if first > end_t or last<start_t: #interval does not contain any snapshots
                    yield None,None
                else:
                    start = self["t_to_snapshot"][first]
                    end = self["t_to_snapshot"][last]+1
                    yield start, end
                start_t += max(int(steps *val),1)
                end_t += max(int(steps *val),1)
        if val == T:
            return intervals(steps=1)
        if T // val <= maxsamp:
            if T//val > minsamp: 
                return intervals(steps = 1)
            else:
                steps= max((T-val)/(minsamp-1)/val, 0.25)
                return intervals(steps = steps)
        else:
            steps = T/(maxsamp)/val
            return intervals(steps= steps)
    def subslice_time(self, val, START = 0, END = -1):
        if END == -1:
            END = self["T"]
        T = END-START
        def intervals():
            start_t = START #start in original time dimension
            end_t = start_t + val
            while end_t <= END:
                first = self.t_to_next_active(start_t) #time of first active snapshot in interval
                last = self.t_to_last_active(end_t-1) #time of last active snapshot in interval
                if first > end_t or last<start_t: #interval does not contain any snapshots
                    yield None,None
                else:
                    start = self["t_to_snapshot"][first]
                    end = self["t_to_snapshot"][last]+1
                    yield start, end
                start_t += val
                end_t += val
        first = self["t_to_snapshot"][self.t_to_next_active(START)]
        last = self["t_to_snapshot"][self.t_to_last_active(((T//val)*val)-1+START)]+1
        length = last - first 
        return intervals(), length     
    def t_to_next_active(self, t):
        nxt = min([i for i in list(self["t_to_snapshot"].keys()) if i>= t])
        return nxt
    def t_to_last_active(self, t):
        last = max([i for i in list(self["t_to_snapshot"].keys()) if i<= t])
        return last
    def snap_to_t(self, snap):
        snapshot_to_t= {val:key for key,val in self["t_to_snapshot"].items()}
        return snapshot_to_t[snap]
    def make_A(self, filename, directed = False):
        A = []
        edges = np.loadtxt(filename, dtype=int)
        i,j,t= np.array(list(zip(*edges)))
        tmin = min(t)
        tmax = max(t)
        indices = {} #indices [0,...ncount]
        for index ,node in enumerate(set(i) | set(j)):
            indices[node] = index        
        #what snapshots are timepoint maped on 
        t_to_snapshot = {}
        for snapshot, t in enumerate(sorted(set(t))):
            t_to_snapshot[t-tmin] = snapshot            
        #active edges for each snapshot
        edges_per_snapshot = {}
        for i,j,t in edges:       
            try:
                edges_per_snapshot[t_to_snapshot[t-tmin]].append((indices[i],indices[j]))
            except:
                edges_per_snapshot[t_to_snapshot[t-tmin]] = [(indices[i],indices[j])]
        shape = (len(indices), len(indices))        
        for t, edges in sorted(edges_per_snapshot.items()):
            row = np.array([i for i, j in edges])
            col = np.array([j for i, j in edges])
            data = np.ones(len(edges), dtype=np.int32)
            a = scipy.sparse.csr_matrix((data, (row, col)), shape=shape, dtype=np.int32)
            if not directed:
                a = a +  a.transpose()
                a = a.sign()
            A.append(a)            
        ncount = len(indices)
        active_edges = {key: A[t_to_snapshot[key]].nnz for key, val in t_to_snapshot.items()}
        ecount = sum(active_edges.values())
        T = tmax-tmin+1  
        Tcomp = len(A)
        return A, t_to_snapshot, T,Tcomp, ecount, ncount, active_edges
    def __str__(self):
        print('filename =	' + str(self['filename']))
        print('objname =	' + str(self['objname']))
        print('directed =	' + str(self['directed']))
        print('ncount = 	' + str(self['ncount']))
        print('ecount = 	' + str(self['ecount']))
        print('T = 		' + str(self['T']))
        print('Tcomp = 	' + str(self['Tcomp']))
        try:
            print("cA0AT = 	"+str(*self["cA0AT"][self["T"]]))
        except:
            pass
        return ""
    def get_dictionary(self):
        dic = {}
        dic.update(self)
        return dic
    def get_static(self):
        return self.aggregate_Matrices(self["A"])
    def static_info(self):
        name = self["objname"][:-11]
        T = self["T"]
        Tcomp = self["Tcomp"]
        W=SubTempNet.aggregate_Matrices(self["A"], weighted = True)
        A = nx.from_scipy_sparse_matrix(W,parallel_edges=True,create_using=nx.MultiDiGraph)
        selfloops = nx.selfloop_edges(A.copy())
        A.remove_edges_from(selfloops)
        deg = np.array(sorted([d for n, d in A.degree()], reverse=True))
        d = np.mean(deg)
        print(r'$\langle k \rangle$=	' + str(d))
        A = nx.from_scipy_sparse_matrix(W, create_using=nx.DiGraph)
        print(r'$\rho$=	' + str(nx.density(A)))
        print(r'$\langle C \rangle$=	' +str(nx.average_clustering(A)))
        try:
            print(r'$D$=	'+str(nx.diameter(A)))
            print(r'$\langle D \rangle$=	'+str(nx.average_shortest_path_length(A)))
        except:
            print(r'$D$=	infinit')
            print(r'$\langle D \rangle$=	infinit')
        print(r'$c_{\mathcal{A},\mathbf{A}}$=	'+str(self["cA0AL"][self["T"]]))
        return 
    def plot_PA(self, normalize=False, save = False):
        """
        alldost: True: plot all sample results, False: only plot mean
        save: save plot at "plots/objname_cA0AT"
        """        
        name = "plots/" + self["objname"][:-11]+"_cA0AT"
        s = 1
        if normalize:
            s=self["ncount"]**2
        x = list([key for key,val in self["PAT"].items()])
        PAT =  list([np.mean(y)/s for t,y in self["PAT"].items()])
        _,PAT= zip(*sorted(zip(*(x,PAT))))
        PA0 =  list([np.mean(y)/s for t,y in self["PA0"].items()])
        _,PA0= zip(*sorted(zip(*(x,PA0))))
        PAT2 = list([np.mean(y)/s for t,y in self["PAT2"].items()])
        _,PAT2= zip(*sorted(zip(*(x,PAT2))))
        PAT4 = list([np.mean(y)/s for t,y in self["PAT4"].items()])
        _,PAT4= zip(*sorted(zip(*(x,PAT4))))
        PAT8 = list([np.mean(y)/s for t,y in self["PAT8"].items()])
        x,PAT8= zip(*sorted(zip(*(x,PAT8))))
        
        fig,ax=self.init_plt("cA0AT")
        ax.set_xlim((1,max(x))) 
        plt.plot(x,PAT, label = "L=T")
        plt.plot(x,PA0, label = "L=1")
        plt.plot(x,PAT2, label = "L= T/2")
        plt.plot(x,PAT4, label = "L= T/4")
        plt.plot(x,PAT8, label = "L= T/8")  
        ax.set_ylabel("PA")  
        ax.legend()
        fig.tight_layout()
        if save:
                fig.savefig(name)
        return  ax
    def plot_cA0AT(self,  save = False):
        """
        alldost: True: plot all sample results, False: only plot mean
        save: save plot at "plots/objname_cA0AT"
        """        
        name = "plots/" + self["objname"][:-11]+"_cA0AT"
        x = list([key for key,val in self["PAT"].items()])
        PAT =  list([np.mean(y)for t,y in self["PAT"].items()])
        _,PAT= zip(*sorted(zip(*(x,PAT))))
        PA0 =  list([np.mean(y)for t,y in self["PA0"].items()])
        _,PA0= zip(*sorted(zip(*(x,PA0))))
        PAT2 = list([np.mean(y)for t,y in self["PAT2"].items()])
        _,PAT2= zip(*sorted(zip(*(x,PAT2))))
        PAT4 = list([np.mean(y)for t,y in self["PAT4"].items()])
        _,PAT4= zip(*sorted(zip(*(x,PAT4))))
        PAT8 = list([np.mean(y)for t,y in self["PAT8"].items()])
        x,PAT8= zip(*sorted(zip(*(x,PAT8))))


        fig,ax=self.init_plt("cA0AT")
        ax.set_xlim((1,max(x))) 
        ax.plot(x,list(np.array(PA0)/np.array(PAT)), label = "L= T")
        ax.plot(x,list(np.array(PA0)/np.array(PAT2)), label = "L= T/2")
        ax.plot(x,list(np.array(PA0)/np.array(PAT4)), label = "L= T/4")
        ax.plot(x,list(np.array(PA0)/np.array(PAT8)), label = "L= T/8")
        ax.legend()
        fig.tight_layout()
        if save:
                fig.savefig(name)
        return  ax
    def plot_cA0AL(self, *T, save = False):
        """
        alldost: True: plot all sample results, False: only plot mean
        save: save plot at "plots/objname_cA0AT"
        """        
        name = "plots/" + self["objname"][:-11]+"_cA0AT"
        x = list([key for key,val in self["PAT"].items()])
        PAT =  list([np.mean(y)for t,y in self["PAT"].items()])
        _,PAT= zip(*sorted(zip(*(x,PAT))))
        PA0 =  list([np.mean(y)for t,y in self["PA0"].items()])
        _,PA0= zip(*sorted(zip(*(x,PA0))))
        PAT2 = list([np.mean(y)for t,y in self["PAT2"].items()])
        _,PAT2= zip(*sorted(zip(*(x,PAT2))))
        PAT4 = list([np.mean(y)for t,y in self["PAT4"].items()])
        _,PAT4= zip(*sorted(zip(*(x,PAT4))))
        PAT8 = list([np.mean(y)for t,y in self["PAT8"].items()])
        x,PAT8= zip(*sorted(zip(*(x,PAT8))))
        
        fig, ax = self.init_plt("cA0AL")
        ax.set_xlim((1,max(x)))
        ax.set_ylim((None,None))
        for t in T:
            ax.plot([x[t],x[t]/2,x[t]/4,x[t]/8,1],[list(np.array(PA0)/np.array(PAT))[t],list(np.array(PA0)/np.array(PAT2))[t],list(np.array(PA0)/np.array(PAT4))[t],list(np.array(PA0)/np.array(PAT8))[t],1], label = "T= "+str(x[t]))
        ax.legend()
        fig.tight_layout()
        if save:
                fig.savefig(name)
        return  ax

    def plot_activity(self, save = False):
        fig, ax = plt.subplots()
        ax.plot(*zip(*self['active edges'].items()))
        ax.set_xlabel('t')
        ax.set_ylabel('active edges')
        fig.tight_layout()
        if save:
            fig.savefig("plots/"+self["objname"][:-11]+"_edgeactivity.png")
    def plot_density(self, save = False):
        fig, ax = plt.subplots()
        x,y = zip(*self['active edges'].items())
        n = self['ncount']
        ax.plot(x,[e /(n*(n-1))  for e in y], linestyle='None', marker='+')
        ax.set_xlabel('t')
        ax.set_ylabel(r'$\rho$')
        fig.tight_layout()
        if save:
            fig.savefig("plots/"+self["objname"][:-11]+"_density.png")
    def plot_densityhist(self, save = False, comp = False):
        x,y = zip(*self['active edges'].items())
        n = self['ncount']
        if (not comp):
            fig, ax = plt.subplots()
            ax.hist([e /(n*(n-1))  for e in y] + (self['T'] - self['Tcomp'])*[0], bins = 50)
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel('snapshots')
            fig.tight_layout()
            if save:
                fig.savefig("plots/"+self["objname"][:-11]+"_densityhist.png")
        else:
            fig, ax = plt.subplots()
            ax.hist([e /(n*(n-1))  for e in y], bins = 50)
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel('snapshots')
            fig.tight_layout()
            if save:
                fig.savefig("plots/"+self["objname"][:-11]+"_densityhist_comp.png")
    @staticmethod
    def init_plt(cf):
        fig, ax = plt.subplots()
        if cf =="cA0AT":
            ax.set_xscale("log")
            ax.set_yscale("linear")
            ax.set_ylabel(r'$c_{\mathcal{A},\mathbf{A}}$')
            ax.set_xlabel("T")
        if cf == "cA0AL":
            ax.set_xscale("log")
            ax.set_yscale("linear")
            ax.set_ylabel(r'$c_{\mathcal{A},\mathcal{A}_L}$')
            ax.set_xlabel("L")
        return fig,ax
    @staticmethod
    def aggregate_Matrices(Matrices, weighted = False, normalized = False):
        for m in Matrices:
            try:
                M = M + m
                if not weighted:
                    M = M.sign()
                if normalized:
                    M = M/M.max()
            except:
                M=m
        return M   
    @staticmethod    
    def accessibility(Matrix, cutoff = None):
        G = nx.DiGraph(Matrix)# directed because A need not be symmetric
        paths = nx.all_pairs_shortest_path_length(G, cutoff= cutoff)
        indices = []
        indptr = [0]
        for row in paths:
            indices.extend(row[1])
            indptr.append(len(indices))
        data = np.ones((len(indices),), dtype=np.uint8)
        A_trans = scipy.sparse.csr_matrix((data, indices, indptr), shape=Matrix.shape)
        return A_trans
    @staticmethod
    def unfold_accessibility(Matrices, track = False):
            tracker = [0,]
            P = Matrices[0].copy()
            D = scipy.sparse.identity(P.shape[0], dtype=np.int32)
            P = P + D
            if track:
                tracker.append(P.nnz)
            for i in range(1, len(Matrices)):
                P = P.sign()
                P = P + P*Matrices[i]
                if track:
                    tracker.append(P.nnz)
            P = P.sign()
            if track:
                return P, tracker
            return P
    @staticmethod
    def save_obj(obj, name ):
        with open('obj/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)   
    @staticmethod
    def load_obj(name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    @staticmethod
    def interpolate_list(lst, interpolationvalue = 1):
        """interpolates all values < interpolationvalue"""
        last = 0
        nexd = 0
        y=[]
        for ind, val in enumerate(lst):
            if val<interpolationvalue:
                y.append(val) 
                last = ind
            elif nexd > ind:
                if last== 0:
                    y.append(lst[nexd])
                else:
                    y.append((lst[last]+lst[nexd])/2)
            else:
                for i in range(ind+1,len(lst)+1):
                    if i == len(lst):
                        y.append(y[-1])
                        break
                    elif lst[i]<interpolationvalue:
                        continue
                    else:
                        nexd = i
                        if last== 0:
                            y.append(lst[nexd])
                        else:
                            y.append((lst[last]+lst[nexd])/2)
                        break
        return y
    @staticmethod
    def intersect_Matrices(Matrices, cumulative = False):
        for m in Matrices:
            try:
                M = M_1 + m
                M_1 = M
                M = M/M.max()
                M = M.floor()
                M = M.astype('int')
            except:
                M = m
                M_1 = M
        return M
    @staticmethod
    def TC(n,p):
        h = np.log(1-(np.log(n)/n))/np.log(1-p)
        l = np.log(1-(1/n)) /np.log(1-p)
        return (h*l)**0.5
    
    @staticmethod
    def deg_seq(A):
        W=SubTempNet.aggregate_Matrices(A, weighted = True)
        AW = nx.from_scipy_sparse_matrix(W,parallel_edges=True,create_using=nx.MultiDiGraph)
        selfloops = nx.selfloop_edges(AW.copy())
        AW.remove_edges_from(selfloops)
        deg = np.array(sorted([d for n, d in AW.degree()], reverse=True))
        return deg
    @staticmethod
    def snapshotlist_to_temporal_edgelist(snapshot_list, filename, separator='\t'):
        """writes snapshot list with networkx networks into temporal edgelist file"""
        with open(filename, 'w') as f:
            t = 0
            for snapshot in snapshot_list:
                t +=1
                for v, w in snapshot.edges():
                    f.write(str(v) + separator + str(w) + separator + str(t)+'\n')