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
    def run_cA0AT(self, *T, maxsamp = 50, minsamp = 5):
        CF ="cA0AT"
        done =[]
        for t in T:
            try:
                self[CF][t]               
            except:
                self[CF][t] = []
            if t == 1:
                self[CF][1]=[1]
                done.append(1)
                continue
            intervals= self.slice_time(t, START =0 , END = self["T"], maxsamp = maxsamp, minsamp = minsamp)
            samplenum = 0
            for start, end in intervals:
                if (start and end) is None: 
                    self[CF][t].append(1) #no active timesteps in interval, only selfloops active
                samplenum += 1
                clear_output()
                print("Done with samplelengths ",done)
                print("Analyzing sample number",samplenum, "for samplelength ", t) 
                try:
                    if len(self[CF][t]) >= samplenum:    #check if this samplenum is already analysed    
                        continue
                except:
                    pass
                else:
                    interval = self["A"][start:end]
                    PA0 = self.unfold_accessibility(interval)
                    AT = self.aggregate_Matrices(interval)
                    PAT = self.accessibility(AT, cutoff = t)
                    self[CF][t].append(PA0.nnz / PAT.nnz)
            done.append(t)
    def run_cA0AL(self, T, *L, maxsamp = 50, minsamp = 5):
        CF ="cA0AL"
        try:
            self[CF][T]
        except:
            self[CF][T] = {}
        samples = self.slice_time(T, START =0 , END = self["T"], maxsamp = maxsamp, minsamp = minsamp, realT= True)
        samplenum = 0
        for samplestart, sampleend in samples:
            if (samplestart and sampleend) is None: #no active timesteps in sample
                continue
            first = self.t_to_next_active(samplestart) #time of first active snapshot in sample
            last = self.t_to_last_active(sampleend-1) #time of last active snapshot in sample
            if first > sampleend or last<samplestart: #sample does not contain any snapshots
                continue
            start = self["t_to_snapshot"][first]
            end = self["t_to_snapshot"][last]+1
            samplenum +=1
            PA0, c_paths = self.unfold_accessibility(self["A"][start:end], track = True)
            for l in L:
                #check if list already exists, else initialize
                try:
                    self[CF][T][l]               
                except:
                    self[CF][T][l] = []
                clear_output()
                print("Done with", samplenum-1, "samples")
                print("Currently analyzing L-Sub with intervallength",l,"for sample number",samplenum,"of sample length",T) 
                #check if this samplenum is already analysed   
                try:
                    assert len(self[CF][T][l]) >= samplenum     
                    continue
                except:
                    pass
                if l == 1:#if L is one, casual fidelity ia also one -> dont analyze
                    self[CF][T][l].append(1)
                    continue
                #initialize sub-slice interval generator
                intervals, length= self.subslice_time(l, START =samplestart , END = sampleend)
                #analyze
                AL = []#scipy.sparse.identity(self["ncount"], dtype=np.int32) #[(I+A_[0,L])^L*(I+A_[L,2L])^L,...
                for start, end in intervals:
                    if (start and end) is None: #no active timesteps in interval, only selfloops active
                        continue
                    else:
                        interval = self["A"][start:end]
                        at = self.aggregate_Matrices(interval)
                        AL.append(self.accessibility(at, cutoff = l))
                        if len(AL)>1:
                            AL=[self.unfold_accessibility(AL)]
                if len(AL)==0:
                    continue
                PAL= self.unfold_accessibility(AL).nnz #(I+A_[0,L])^L * (I+A_[L,2L])^L *...*(I+A_[(k-1)*L, K*L])^L
                #save result:
                self[CF][T][l].append(c_paths[length]/PAL)
    def slice_time(self, val, START = 0, END = -1,maxsamp = -1, minsamp = 5, realT = False):
        if END == -1:
            END = self["T"]
        if maxsamp == -1:
            maxsamp = self["T"]
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
    def ER_fit(self, plot = True, save = False, efficient = False):        
        dic = {}
        
        #SIMULATION
        x = [key for key in self["cA0AT"].keys()]
        y = np.asarray([np.mean(val) for val in self["cA0AT"].values()])
        x,y  = zip(*sorted(zip(*(list(x),list(y))), reverse=True))
        x=np.array(x)
        y=np.array(y)
        t =self["T"]
        n = self["ncount"]
        ecount= self["ecount"]
        per = (ecount /t)/(n*(n-1))
        vals, idxs = min((val, idx) for (idx, val) in enumerate(y))
        dic['p']=per
        dic["CM sim"]=vals
        dic["CT sim"]=x[idxs]
        
        #THEORETICAL FIT:
        T = self.TC(n,per)
        mü = np.log(T)
        sig = self.sig(n)
        if efficient:
            A = self.AC(n,(ecount /self["Tcomp"])/(n*(n-1)))
        else:
            A = self.AC(n,per)  
        xtheo = x.copy()
        ytheo = self.lognormal(xtheo,A,mü,sig)
        r2=self.R2(y,ytheo)
        dic["MC theo"]=1-A
        dic["TC theo"]=T
        dic["R2 theo"]=r2
        dic["EMC theo"]=abs(vals-(1-A))
        dic["ETC theo"]=abs(np.log(x[idxs])- np.log(T))/np.log(x[idxs])
        
        #OPTIMAL FIT
        popt,pcov = opt.curve_fit(self.lognormal,x,y,p0=[A,mü,sig],absolute_sigma=True)
        xfit = x.copy()
        yfit = self.lognormal(xfit,*popt)
        r2=self.R2(y,yfit)
        valf, idxf = min((val, idx) for (idx, val) in enumerate(yfit))
        dic["MC opt"]=valf
        dic["TC opt"]=xfit[idxf]
        dic["R2 opt"]=r2
        dic["EMC opt"]=abs(vals-valf)
        dic["ETC opt"]=abs(np.log(x[idxs])- np.log(xfit[idxf]))/np.log(x[idxs])
        
        #PLOT
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_xlabel('T')
        ax.set_ylabel(r'$c_{\mathcal{A},\mathbf{A}}$')
        ax.plot(x,y,'b-',label='sim')
        ax.plot(xtheo,ytheo,'g-',label='theo')
        ax.plot(xfit,yfit,'r-',label='opt')                     
        ax.legend(fontsize=26)
        fig.tight_layout()
        if save:
            plt.savefig("plots/"+self["objname"][:-11] + "_ERfit"+".png")
            
        return dic,ax
    def EDS_fit(self, save = False, sample = [None,None,1]):
        name = self["objname"][:-11]+"_EDSfit"
        dic = {}
        x = np.asarray([key for key,val in self["cA0AT"].items()])
        y = np.asarray([np.mean(val) for key,val in self["cA0AT"].items()])
        x,y  = zip(*sorted(zip(*(x,y)), reverse=True))
        t = self["T"]
        deg = self["deg_seq"] / t
        d= np.mean(deg)
        T = np.ceil(np.e/d)
        n = self["ncount"]
        dic["mean deg"]=d
        dic["TC^"] = T

        #build new temporal network
        A = []
        for j in range(t):
            G = nx.expected_degree_graph(deg)
            G = nx.Graph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
            A.append(G)
        #ensure that network is full length
        if len(A[0].edges()) ==0:
            u,v = np.random.choice(range(n),2,replace = False )
            A[0].add_edge(u,v)
        if len(A[-1].edges()) ==0:
            u,v = np.random.choice(range(n),2,replace = False )
            A[-1].add_edge(u,v)
        #save edgelist
        self.snapshotlist_to_temporal_edgelist(A, "data/" + name +".edges")
        #create new object
        temp = SubTempNet("data/"+name+".edges", name ,directed= self["directed"])
        print("STN created")
        
        #run
        T= sorted(list(set(list(x)[sample[0]:sample[1]:sample[2]])), reverse = True)
        for t in T:
            temp.run_cA0AT(t,maxsamp = 30, minsamp = 5)        
        print("all runned")
        
        #Plot:
        xfit = np.asarray([key for key,val in temp["cA0AT"].items() if key in T])
        yfit = np.asarray([np.mean(val) for key,val in temp["cA0AT"].items() if key in T])
        xfit,yfit  = zip(*sorted(zip(*(xfit,yfit)), reverse=True))
        fig,ax=self.init_plt("cA0AT")
        ax.set_xlim((1,max(x))) 
        ax.plot(list(xfit)+[1],list(yfit)+[1],'g',label='fit')
        ax.plot(x,y,'b',label='sim')
        ax.legend(fontsize=26)
        fig.tight_layout()
        if save:
            plt.savefig("plots/"+self["objname"][:-11] + "_EDSfit"+".png")
        
        #Goodnes of Fit
        vals, idxs = min((val, idx) for (idx, val) in enumerate(y))
        valf, idxf = min((val, idx) for (idx, val) in enumerate(yfit))
        y = np.asarray(list(y)[sample[0]:sample[1]:sample[2]])
        r2=self.R2(y,yfit)
        dic["MC sim"]=vals
        dic["TC sim"]=x[idxs]
        dic["MC fit"]=valf
        dic["TC fit"]=xfit[idxf]
        dic["R2"]=r2
        dic["EMC"]=abs(vals-valf)
        dic["ETC"]=abs(np.log(x[idxs])- np.log(xfit[idxf]))/np.log(x[idxs])
        return dic, ax
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
    def plot_cA0AT(self, shade = False, save = False, alldots = True):
        """
        alldost: True: plot all sample results, False: only plot mean
        save: save plot at "plots/objname_cA0AT"
        """
        CF = self["cA0AT"]
        name = "plots/" + self["objname"][:-11]+"_cA0AT"
        y = [val for key,val in CF.items()]
        x = [key for key, values in CF.items()]
        yq1=[np.quantile(val,0.25) for val in y]
        yq3=[np.quantile(val,0.75) for val in y]
        ymin=[min(val) for val in y]
        ymax=[max(val) for val in y]
        _,ymin  = zip(*sorted(zip(*(x,ymin))))
        _,ymax  = zip(*sorted(zip(*(x,ymax))))
        _,yq1  = zip(*sorted(zip(*(x,yq1))))
        _,yq3  = zip(*sorted(zip(*(x,yq3))))
        y = [np.mean(val) for val in y]
        x,y  = zip(*sorted(zip(*(x,y))))
        
        fig,ax=self.init_plt("cA0AT")
        ax.set_xlim((1,max(x))) 
        ax.plot(x,y, label = "mean")
        if alldots:
            for key,values in CF.items():
                ax.scatter(len(values)*[key],values)
            ax.legend(fontsize=26)
        if shade:
            ax.fill_between(x, ymin, ymax,facecolor='grey',alpha=0.15, label = "min-max")
            ax.fill_between(x, yq1, yq3,facecolor='grey',alpha=0.1, label = "Q1-Q3")
            ax.legend(fontsize=26)
        fig.tight_layout()
        if save:
                fig.savefig(name)
        return  ax
    def plot_cA0AL(self, *Tlist, intlen = True, alldots = True,  change = False, save = False, normalized = False,cA0AT = False, label = True):
        name = "plots/" + self["objname"][:-11]+"_cA0AL"  + (not intlen)*"_intnum" + change*"_change"+ normalized*"_norm" +cA0AT*"_cA0AT"
        fig, ax = self.init_plt("cA0AL")
        for T in Tlist:
            CF = self["cA0AL"][T]
            y = [np.mean(val) for key,val in CF.items()]
            #x
            if intlen:
                x = [key for key, values in CF.items()]
            else:
                x =[T//key for key,val in CF.items()]
                ax.set_xlabel("number of intervals")
            x,y  = zip(*sorted(zip(*(x,y))))
            if normalized:
                ymin = min(y)
                y = [(i-ymin)/(1-ymin) for i in y]
                ax.set_ylabel(r"$\tilde{c}_{\mathcal{A},\mathcal{A}_L}$")
            if change:
                y = [0] + [y[i]-y[i-1] for i in range(1,len(y))]
                if normalized:
                    ax.set_ylabel(r"$ \Delta \tilde{c}_{\mathcal{A},\mathcal{A}_L}$")
                else:
                    ax.set_ylabel(r'$ \Delta c_{\mathcal{A},\mathcal{A}_L}$')
            ax.plot(x,y,label = "T="+str(T))
            if (alldots and len(Tlist)==1):
                if intlen:
                    x=[key for key, values in CF.items()]
                else:
                    x=[T//key for key,val in CF.items()]
                ylist = zip(*[val for key,val in CF.items()])  
                for y in ylist:
                    ax.scatter(x,y)
        if cA0AT:
            CF = self["cA0AT"]
            y = [np.mean(val) for key,val in CF.items()]
            x = [key for key, values in CF.items()]
            ax.set_xlabel(r'L, T')
            x,y  = zip(*sorted(zip(*(x,y))))
            if normalized:
                ax.set_ylabel(r"$\tilde{c}_{\mathcal{A},\mathcal{A}_L}, \ c_{\mathcal{A},\mathbf{A}}$")
            else:
                ax.set_ylabel(r'$c_{\mathcal{A},\mathcal{A}_L}, c_{\mathcal{A},\mathbf{A}}$')
            ax.plot(x,y, color = "b", label = r'mean $c_{\mathcal{A},\mathbf{A}}$')
        fig.tight_layout()
        if label:
                ax.legend(fontsize=18)
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
    def sig(n):
        sigma = np.log((np.log(1-(np.log(n)/n))/np.log(1-(1/n)))**0.5)
        return sigma
    @staticmethod
    def R2(y,yfit):
        ssr = np.sum((y - yfit) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ssr / sst)
        return r2
    @staticmethod
    def lognormal(x,a,mü,sigma):
        return 1- a*np.exp(-(np.log(x)-mü)**2/(2*sigma**2))   
    @staticmethod
    def AC(n,per):
        p = 1-(1-per)**SubTempNet.TC(n, per)
        d = n*p
        PA0  = sum(SubTempNet.pathnum(n,per))
        lcc = ((scipy.special.lambertw(-np.exp(-d)*d)+d)/d).real
        PAT = n + (lcc*n)**2
        return 1-(PA0/PAT)
    @staticmethod
    def pathnum(n,per):
        T = int(SubTempNet.TC(n, per))
        p0 = n
        p1 = n*(n-1)*(1-(1-per)**T)
        p2 = n*(n-1)*sum([(1-per)**(t-1)*per*(n-2)*(1-(1-per)**(T-t)) for t in range(1,T+1)])
        p3 = n*(n-1)*sum([(1-per)**(t1-1)*per*(n-2)*sum([(1-per)**(t2-1)*per*(n-3)*(1-(1-per)**(T-t1-t2)) for t2 in range(1,T-t1+1)]) for t1 in range(T+1)])
        return [p0,p1,p2,p3]
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