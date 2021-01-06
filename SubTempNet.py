"""needed Packages"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import networkx as nx
import pathpy as pp
import scipy
import scipy.optimize as opt
import pickle
from IPython.display import clear_output
plt.rcParams.update({'legend.fontsize': 'x-large',
         'figure.figsize': (6, 4),
         'axes.labelsize': '17',
         'axes.titlesize':'17',
         'legend.fontsize': 17,
         'legend.fontsize': 17,
         'xtick.labelsize':'17',
         'ytick.labelsize':'17',
         'mathtext.fontset':'stix',
         'font.family':'STIXGeneral'
         })
class SubTempNet(dict):
    def __init__(self, filename, objname, directed = False, init = True):
        """ 
        filename:= path of data
        objname:= name to store object
        """
        self["filename"] = filename
        self["objname"] = objname+"_SubTempNet"
        self["directed"] = directed
        if init:
            A, T, ecount, ncount = self.make_A(filename, directed)
            self["A"]=A
            self["ecount"] = ecount
            self["ncount"] = ncount
            self["T"] = T
            self["deg_seq"] = self.deg_seq(A, directed = directed)
            
            #Statistics
            self["PA0"]={}
            self["PAT"]={}
            self["PAT2"]={}
            self["PAT4"]={}
            self["PAT8"]={}
            self["PAT_LCC"]={}
        else:
            self.load()
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
    def run(self, *T, maxsamp = 50, minsamp = 5):
        done =[]
        reached_max = self["T"]+1 
        for t in T:
            if t ==1:
                self["PAT"][t]=[self["ncount"]]
                self["PAT2"][t]=[self["ncount"]]
                self["PAT4"][t]=[self["ncount"]]
                self["PAT8"][t]=[self["ncount"]]
                self["PA0"][t]=[self["ncount"]]
                self["PAT_LCC"][t]=[1]
                done.append(t)
                continue
            if reached_max < t:
                self["PAT"][t]=[self["ncount"]**2]
                self["PAT2"][t]=[self["ncount"]**2]
                self["PAT4"][t]=[self["ncount"]**2]
                self["PAT8"][t]=[self["ncount"]**2]
                self["PA0"][t]=[self["ncount"]**2]
                self["PAT_LCC"][t]=[self["ncount"]]
                continue
            self["PAT"][t]=[]
            self["PAT_LCC"][t]=[]
            self["PA0"][t]=[]
            for k in [2,4,8]:
                    if k <= t: #temporal network is longer than aggregation window
                        self["PAT"+str(k)][t]=[]
            samplenum = 0
            samples = self.sample_TN(t, maxsamp = maxsamp, minsamp = minsamp)
            for samplestart, sampleend in samples:
                samplenum +=1
                
                #PA0 calculates accessibility of original temporal network sample
                PA0 = self.unfold_accessibility(self["A"][samplestart:sampleend])
                self["PA0"][t].append(PA0.nnz)
                
                #PAT calculates accessibility of fully aggregated network sample
                AT= self.aggregate_Matrices(self["A"][samplestart:sampleend])
                PAT = self.accessibility(AT, cutoff = t)
                self["PAT"][t].append(PAT.nnz)
                self["PAT_LCC"][t].append(self.LCC_size(AT))
                                
                #PATk calculates accessibility of subaggregated network sample
                for k in [2,4,8]:
                    if t < k: #temporal network is shorter than aggregation window
                        continue
                    slicelengh = t//k ##????
                    slices= self.slice_TN(slicelengh, samplestart, sampleend)
                    AL = []
                    for slicestart, sliceend in slices:
                        AL.append(self.accessibility(self.aggregate_Matrices(self["A"][slicestart:sliceend]), cutoff = slicelengh))
                        if len(AL)>1:
                            AL=[self.unfold_accessibility(AL)]
                            
                    PATk = self.unfold_accessibility(AL)
                    self["PAT"+str(k)][t].append(PATk.nnz) 
                    
                    #print status update
                    clear_output()
                    print("Done with samplelengths ",done)
                    print("Analyzing sample number",samplenum, "for samplelength ", t, "and ",k, "slices") 
            done.append(t)
            if np.mean(self["PA0"][t]) == self["ncount"]**2:
                reached_max = t
    def sample_TN(self, samplelength, maxsamp = -1, minsamp = 5):
        def intervals(steps =1):
            intlen = samplelength
            step = max(int(steps *samplelength),1)
            start = 0
            end = samplelength
            while end <= self["T"]:
                yield start, end
                start += step
                end += step
        if samplelength == self["T"]:
            return [(0,self["T"])]
        if self["T"] // samplelength <= maxsamp:
            if self["T"]//samplelength > minsamp: 
                return intervals(steps = 1)
            else:
                steps= max((self["T"]-samplelength)/(minsamp-1)/samplelength, 0.25)
                return intervals(steps = steps)
        else:
            steps = self["T"]/(maxsamp)/samplelength
            return intervals(steps= steps)
    def slice_TN(self, slicelengh, samplestart, sampleend):
        def intervals():
            step = slicelengh
            start = samplestart 
            end = start + step
            while end <= sampleend:
                yield start, end
                start += step
                end += step
            if start < sampleend:
                yield start,sampleend
        return intervals()     
    def make_A(self, filename, directed = False):
        #create snapshotlist
        A = []
        edges = np.loadtxt(filename, dtype=int)
        i,j,t= np.array(list(zip(*edges)))
        tmin = min(t)
        tmax = max(t)
        T = tmax-tmin+1  
        indices = {} #indices [0,...ncount]
        for index ,node in enumerate(set(i) | set(j)):
            indices[node] = index   
        ncount = len(indices)
        ecount = len(edges)
        shape = (len(indices), len(indices))        

        #active edges for each snapshot
        edges_per_snapshot = {t: [] for t in range(T)}
        for i,j,t in edges:   
            edges_per_snapshot[t-tmin].append((indices[i],indices[j]))
        
        for t in range(T):
            edges = edges_per_snapshot[t]
            row = np.array([i for i, j in edges])
            col = np.array([j for i, j in edges])
            data = np.ones(len(edges), dtype=np.int32)
            a = scipy.sparse.csr_matrix((data, (row, col)), shape=shape, dtype=np.int32)
            if not directed:
                a = a +  a.transpose()
                a = a.sign()
            A.append(a)            
        return A, T, ecount, ncount
    def __str__(self):
        print('filename =	' + str(self['filename']))
        print('objname =	' + str(self['objname']))
        print('directed =	' + str(self['directed']))
        print('ncount = 	' + str(self['ncount']))
        print('ecount = 	' + str(self['ecount']))
        print('T = 		' + str(self['T']))
        return ""
    def get_dictionary(self):
        dic = {}
        dic.update(self)
        return dic
    def get_static(self):
        return self.aggregate_Matrices(self["A"])
    def plot_PA(self, normalize=True, sub = True, save = False, vline = False):
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_ylabel(r'$\rho$')  
        ax.set_xlabel(r'$T$')
        linestyle = "--*"
        colrange = [1,2,3,4,5]
        colo = plt.cm.get_cmap('viridis', len(colrange)).colors
        
        if normalize:
            s=self["ncount"]**2
        else:
            s=1
        
        x = list([key for key,val in self["PAT"].items()])
        PAT =  list([np.mean(y)/s for t,y in self["PAT"].items()])
        x,PAT= zip(*sorted(zip(*(x,PAT))))
        plt.plot(x,PAT, linestyle, color = colo[0], label = r'$I=1$')
        
        if sub:
            x = list([key for key,val in self["PAT2"].items()])
            PAT2 = list([np.mean(y)/s for t,y in self["PAT2"].items()])
            x,PAT2= zip(*sorted(zip(*(x,PAT2))))
            plt.plot(x,PAT2, linestyle, color = colo[1], label = r'$I=2$')

            x = list([key for key,val in self["PAT4"].items()])
            PAT4 = list([np.mean(y)/s for t,y in self["PAT4"].items()])
            x,PAT4= zip(*sorted(zip(*(x,PAT4))))
            plt.plot(x,PAT4, linestyle, color = colo[2], label = r'$I=4$')

            x = list([key for key,val in self["PAT8"].items()])
            PAT8 = list([np.mean(y)/s for t,y in self["PAT8"].items()])
            x,PAT8= zip(*sorted(zip(*(x,PAT8))))
            plt.plot(x,PAT8, linestyle, color = colo[3], label = r'$I=8$')  
        
        x = list([key for key,val in self["PA0"].items()])
        PA0 =  list([np.mean(y)/s for t,y in self["PA0"].items()])
        x,PA0= zip(*sorted(zip(*(x,PA0))))
        plt.plot(x,PA0, linestyle, color = colo[4], label = r'$I=T$')
        
        if vline:
            for (x,col,label) in vline:
                ax.vlines(x = x, ymin=0, ymax = 1, colors = col,   label = label)
        plt.legend(handlelength = 0.8, handletextpad=0.2)
        ax.tick_params(which = 'major', axis='both', width=1, length = 10, labelsize=17, direction='in')
        ax.tick_params(which = 'minor', axis='both', width=1, length = 5, labelsize=17, direction='in')
        ax.set_ylim(0, ax.set_ylim()[1])
        ax.set_xlim(1,ax.set_xlim()[1])
        fig.tight_layout()
        
        #save plot
        if save:
                fig.savefig("fig/"+save, dpi=600)
        return  ax
    def plot_LCC(self, normalize=True, vline = False, log = False, save = False, ACC = True):
        fig, ax = plt.subplots()
        colrange = [1,2,3,4,5]
        colo = plt.cm.get_cmap('viridis', len(colrange)).colors
        color = "red"
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(r'$G^2$')
        ax.set_xlabel(r'$T$')
        linestyle = "--*"
        
        if normalize:
            s=self["ncount"]**2
        else:
            s=1
        x = list([key for key,val in self["PAT"].items()])
        PAT_LCC =  list([(np.mean(LCC)**2)/s for t,LCC in self["PAT_LCC"].items()])
        x,PAT_LCC= zip(*sorted(zip(*(x,PAT_LCC))))
        ax.plot(x,PAT_LCC, "^", color = color, label = r'$G^2$')
        
        if ACC:
            color = colo[0]
            ax2= ax.twinx()
            if normalize:
                s=self["ncount"]**2
            x = list([key for key,val in self["PAT"].items()])
            PAT =  list([((np.mean(y))/s)**1 for t,y in self["PAT"].items()])
            #PAT =  list([((np.mean(y))/s)**1 for t,y in self["PAT"].items()])
            x,PAT= zip(*sorted(zip(*(x,PAT))))
            ax2.plot(x,PAT, linestyle, color = color,label = r'$\rho$')
            ax2.set_ylabel(ylabel = r'$\rho$')
            if log:
                ax2.set_yscale("log")
                ax2.set_ylim(1, 13000)#ax2.set_ylim()[1])
            else:
                ax2.set_ylim(0, ax2.set_ylim()[1])
            ax2.tick_params(which = 'major', axis='both', width=1, length = 10, labelsize=17, direction='in')
            ax2.tick_params(which = 'minor', axis='both', width=1, length = 5, labelsize=17, direction='in')
            ax2.set_xticks([10,100,1000])
            ax.plot([], [], linestyle, label = r'$\rho$', color = color)
        ax.legend()
        
        ax.tick_params(which = 'major', axis='both', width=1, length = 10, labelsize=17, direction='in')
        ax.tick_params(which = 'minor', axis='both', width=1, length = 5, labelsize=17, direction='in')
        ax.set_xticks([10,100,1000])
        if log:
            ax.set_ylim(1, 13000)#ax.set_ylim()[1])
        else:
            ax.set_ylim(0,ax.set_ylim()[1])
        ax.set_xlim(1,ax.set_xlim()[1])
        if vline:
            for (x,col,label) in vline:
                ax.vlines(x = x, ymin=0, ymax = ax.set_ylim()[1], colors = col,   label = label)
        fig.tight_layout()
        
        #save plot
        if save:
                fig.savefig("fig/"+save, dpi=600)
        return  ax
    def plot_cA0AT(self, sub = True, vline = False, rho = False, legend = False, save = False): 
        fig, ax = plt.subplots()
        if vline:
            for (x,col,label) in vline:
                ax.vlines(x = x, ymin=0, ymax = 1, colors = col,   label = label)
        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_ylabel(r'$c$')
        ax.set_xlabel(r'$T$')
        linestyle = "--*"
        colrange = [1,2,3,4,5]
        colo = plt.cm.get_cmap('viridis', len(colrange)).colors
        
        PA0 =  {t:np.mean(y) for t,y in self["PA0"].items()}

        x = list([key for key,val in self["PAT"].items()])
        PAT =  list([PA0[t]/np.mean(self["PAT"][t]) for t in x])
        x,PAT= zip(*sorted(zip(*(x,PAT))))
        ax.plot(x,PAT,linestyle, color = colo[0], label = r'$I=1$')
        
        if sub:
            x = list([key for key,val in self["PAT2"].items()])
            PAT2 = list([PA0[t]/np.mean(self["PAT2"][t]) for t in x])
            x,PAT2= zip(*sorted(zip(*(x,PAT2))))
            ax.plot(x,PAT2, linestyle, color = colo[1], label = r'$I=2$')

            x = list([key for key,val in self["PAT4"].items()])
            PAT4 = list([PA0[t]/np.mean(self["PAT4"][t]) for t in x])
            x,PAT4= zip(*sorted(zip(*(x,PAT4))))
            ax.plot(x,PAT4,linestyle, color = colo[2], label = r'$I=4$')

            x = list([key for key,val in self["PAT8"].items()])
            PAT8 = list([PA0[t]/np.mean(self["PAT8"][t]) for t in x])
            x,PAT8= zip(*sorted(zip(*(x,PAT8))))
            ax.plot(x,PAT8,linestyle, color = colo[3], label = r'$I=8$')
            
            x = list([key for key,val in self["PA0"].items()])
            PA0 = list([PA0[t]/np.mean(self["PA0"][t]) for t in x])
            x,PA0= zip(*sorted(zip(*(x,PA0))))
            ax.plot(x,PA0,linestyle, color = colo[4], label = r'$I=T$')
        if rho:
            try:
                axin = ax.inset_axes(rho) 
            except:
                axin = ax.inset_axes([0.13, 0.16, 0.39, 0.45]) 
            axin.set(xscale ="log",
                     yscale = "linear")
            axin.set_ylabel(r'$\rho$',labelpad=0) 
            axin.set_xlabel(r'$T$',labelpad=-2)
            axin.tick_params(which = 'major', axis='both', width=1, length = 10, labelsize=17, direction='in')
            axin.tick_params(which = 'minor', axis='both', width=1, length = 5, labelsize=17, direction='in')
            colrange = [1,2,3,4,5]
            colo = plt.cm.get_cmap('viridis', len(colrange)).colors

            if True:
                s=self["ncount"]**2
            else:
                s=1

            x = list([key for key,val in self["PAT"].items()])
            PAT =  list([np.mean(y)/s for t,y in self["PAT"].items()])
            x,PAT= zip(*sorted(zip(*(x,PAT))))
            axin.plot(x,PAT, linestyle, color = colo[0], label = r'$1$')
            
            if sub:
                x = list([key for key,val in self["PAT2"].items()])
                PAT2 = list([np.mean(y)/s for t,y in self["PAT2"].items()])
                x,PAT2= zip(*sorted(zip(*(x,PAT2))))
                axin.plot(x,PAT2, linestyle, color = colo[1], label = r'$2$')

                x = list([key for key,val in self["PAT4"].items()])
                PAT4 = list([np.mean(y)/s for t,y in self["PAT4"].items()])
                x,PAT4= zip(*sorted(zip(*(x,PAT4))))
                axin.plot(x,PAT4, linestyle, color = colo[2], label = r'$4$')

                x = list([key for key,val in self["PAT8"].items()])
                PAT8 = list([np.mean(y)/s for t,y in self["PAT8"].items()])
                x,PAT8= zip(*sorted(zip(*(x,PAT8))))
                axin.plot(x,PAT8, linestyle, color = colo[3], label = r'$8$')  

            x = list([key for key,val in self["PA0"].items()])
            PA0 =  list([np.mean(y)/s for t,y in self["PA0"].items()])
            x,PA0= zip(*sorted(zip(*(x,PA0))))
            axin.plot(x,PA0, linestyle, color = colo[4], label = r'$T$')
            axin.set_ylim(0, axin.set_ylim()[1])
            axin.set_xlim(5, axin.set_xlim()[1])
            #axin.set_xticks([10,100,1000])
            #axin.set_yticks([0,0.2,0.4]) #SBM
            axin.set_yticks([0,0.5,1])
            if vline:
                for (x,col,label) in vline:
                    axin.vlines(x = x, ymin=0, ymax = 1, colors = col,   label = label)
            
        if legend:
            #ax.legend(bbox_to_anchor=(0, -0.5, 1, 0), loc="lower left", mode="expand", ncol=5, title_fontsize = 17, title= None, handlelength = 0.8, handletextpad=0.2)
            #plt.legend(bbox_to_anchor=(1,1), loc="upper left", title_fontsize = 17, title= r'$I=$', handlelength = 0.8, handletextpad=0.2)
            try:
                plt.legend(handlelength = 0.8, handletextpad=0.2, loc= legend)
            except:
                plt.legend(handlelength = 0.8, handletextpad=0.2)
        ax.tick_params(which = 'major', axis='both', width=1, length = 10, labelsize=17, direction='in')
        ax.tick_params(which = 'minor', axis='both', width=1, length = 5, labelsize=17, direction='in')
        #ax.set_xticks([10,100,1000])
        ax.set_ylim(0, ax.set_ylim()[1])
        ax.set_xlim(1, ax.set_xlim()[1])
        fig.tight_layout()
        
        #save plot
        if save:
                fig.savefig("fig/"+save, dpi=600)
        return
    def plot_min(self, save = False):  
        fig, ax = plt.subplots()
        ax.set_ylabel(r'$min(c)$')
        ax.set_xlabel(r'$I$')
        linestyle = "--*"
        M = []
        L = [1,2,4,8]
        PA0 =  {t:np.mean(y) for t,y in self["PA0"].items()}

        x = list([key for key,val in self["PAT"].items()])
        PAT =  list([PA0[t]/np.mean(self["PAT"][t]) for t in x])
        x,PAT= zip(*sorted(zip(*(x,PAT))))
        M.append(min(PAT))

        x = list([key for key,val in self["PAT2"].items()])
        PAT2 = list([PA0[t]/np.mean(self["PAT2"][t]) for t in x])
        x,PAT2= zip(*sorted(zip(*(x,PAT2))))
        M.append(min(PAT2))
        
        x = list([key for key,val in self["PAT4"].items()])
        PAT4 = list([PA0[t]/np.mean(self["PAT4"][t]) for t in x])
        x,PAT4= zip(*sorted(zip(*(x,PAT4))))
        M.append(min(PAT4))
        
        x = list([key for key,val in self["PAT8"].items()])
        PAT8 = list([PA0[t]/np.mean(self["PAT8"][t]) for t in x])
        x,PAT8= zip(*sorted(zip(*(x,PAT8))))
        M.append(min(PAT8))
        
        ax.plot(L,M,linestyle)
        ax.tick_params(which = 'major', axis='both', width=1, length = 10, labelsize=17, direction='in')
        ax.tick_params(which = 'minor', axis='both', width=1, length = 5, labelsize=17, direction='in')
        #ax.set_ylim(0, ax.set_ylim()[1])
        fig.tight_layout()
        
        #save plot
        if save:
                fig.savefig("fig/"+save, dpi=600)
        return
    def plot_cA0AL(self, *T, save = False):  
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_ylabel(r'$c_{\mathcal{A},\mathcal{A}_L}$')
        ax.set_xlabel("L")
        linestyle = "--*"

        x = sorted(list([key for key,val in self["PAT8"].items()]))
        
        for t in T:
            t= x[t]
            ax.plot([t,t/2,t/4,t/8, 1],[np.mean(self["PA0"][t])/np.mean(self["PAT"][t]),
                                                    np.mean(self["PA0"][t])/np.mean(self["PAT2"][t]),
                                                    np.mean(self["PA0"][t])/np.mean(self["PAT4"][t]),
                                                    np.mean(self["PA0"][t])/np.mean(self["PAT8"][t]),
                                                    1],
                    linestyle, label = "T= "+str(t))
        
        ax.legend()
        ax.tick_params(which = 'major', axis='both', width=1, length = 10, labelsize=16, direction='in')
        ax.tick_params(which = 'minor', axis='both', width=1, length = 5, labelsize=16, direction='in')
        ax.set_ylim(0, ax.set_ylim()[1])
        #ax.set_xlim(1,10**3.8)
        fig.tight_layout()
        
        #save plot
        if save:
                fig.savefig("fig/"+save, dpi=600)
        return
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
    def LCC_size(M, verbose = False):
        LCC = max(nx.connected_components(nx.from_scipy_sparse_matrix(M)), key=len)
        return len(LCC)
        
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
    def TC1(n,p):
        l = np.log(1-(1/n)) /np.log(1-p)
        return l
    @staticmethod
    def TC2(n,p):
        h = np.log(1-(np.log(n)/n))/np.log(1-p)
        return h
    @staticmethod
    def deg_seq(A, directed = False):
        W=SubTempNet.aggregate_Matrices(A, weighted = True)
        if directed:
            AW = nx.from_scipy_sparse_matrix(W,parallel_edges=True,create_using=nx.MultiDiGraph)
        else:
            AW = nx.from_scipy_sparse_matrix(W,parallel_edges=True,create_using=nx.MultiGraph)
        selfloops = nx.selfloop_edges(AW.copy())
        AW.remove_edges_from(selfloops)
        deg = np.array(sorted([d for n, d in AW.degree()], reverse=True))
        return deg
    @staticmethod
    def write_snapshotlist_to_edgelist(snapshotlist, filename, separator='\t'):
        """writes snapshot list with networkx networks into temporal edgelist file"""
        with open(filename, 'w') as f:
            t = 0
            for snapshot in snapshotlist:
                t +=1
                for v, w in snapshot.edges():
                    f.write(str(v) + separator + str(w) + separator + str(t)+'\n')
    @staticmethod
    def ChungLu(deg):
        G = nx.expected_degree_graph(deg, selfloops = False)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G