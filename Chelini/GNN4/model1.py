import torch
import torch.nn as nn
import numpy as np
import math 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Sequential


class pre_processing(torch.nn.Module):
    def __init__(self, 
                 in_feat:int, 
                 hid:int,
                 emb:dict, 
                 device):
        super(pre_processing, self).__init__()
        self.hid = hid
        self.in_feat = in_feat
        self.device = device
        self.emb = nn.ModuleDict({})
        out = 0
        for key in emb.keys():
            self.emb[key] = nn.Embedding(num_embeddings = emb[key][0],
                                         embedding_dim = emb[key][1], 
                                         device=device)
            out += emb[key][1]
        
        # sommo le tre variabili non categoriche dell'input
        out += 3
        
        self.linear = nn.Sequential(nn.Linear(in_features = out, 
                                              out_features = 128),
                                    nn.ReLU(), 
                                    nn.Linear(in_features = 128,
                                              out_features = 128),
                                    nn.ReLU(),
                                    nn.Linear(in_features = 128,
                                              out_features = out-1)
        )
        
        
    def forward(self, x):
        out = []
        for i,key in enumerate(self.emb.keys()):
            out.append(self.emb[key](x[:,i].int().to(self.device)))
        out = torch.cat(out,-1)    
        out = torch.cat((out, x[:,-3:]),-1)
        out = self.linear(out.float())
        
        out = torch.cat((out, x[:,-1:]),-1)
        return out.float()

    
class Kernel(torch.nn.Module):
    def __init__(self, 
                 d: int,
                 hid: int, 
                 past: int,
                 future: int
                ):
        super(Kernel, self).__init__()
        
        # Siccome non posso generare una matrice simmetrica definita positiva 
        # costruisco una matrice "d x hid" che poi trasformo in simmetrica definita positiva
        self.d = d
        self.hid = hid
        self.past = past
        self.future = future
        
        # Parametri da imparare
        self.smoothing = torch.nn.Parameter(torch.randn(1))
        self.W = torch.nn.Parameter(torch.randn(self.d, self.hid))
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x, y, A):
        x_p = x[:,:self.past,:]
        x_f = x[:,self.past:,:]
        B, P, O =x_p.shape
        B, P1, O =x_f.shape
        
        # now the matrix is positive definite
        theta = torch.matmul(self.W, self.W.T)
        theta = (theta+theta.T)/2
        
        diff = x_p.view(B,1,P,O)-x_f.view(B,P1,1,O)
        w = torch.matmul(diff, theta)
        w = torch.matmul(w,torch.transpose(diff, -2,-1))
        w = -0.5*w/(self.sig(self.smoothing)*0.01)
        
        # B, F, P, P ---> B, F, P
        # indico i pesi di ogni osservazione del passato per quanto riguarda la i-esima osservazione nel futuro
        alpha = torch.diagonal(w,dim1=-1, dim2=-2)
        # devo fare la trasposta perchè i vettori sono per riga e non per colonna
        A_tmp = A[self.past:,:self.past]#.transpose(-2,-1)
        alpha = alpha.masked_fill(A_tmp == 0, float('-inf'))
        alpha = F.softmax(alpha, -1)
        yh = torch.matmul(alpha,y)
        return yh
    
    
class GAT(torch.nn.Module):
    def __init__(self, 
                in_feat:int, 
                hid:int,  
                in_head:int, 
                out_head:int,
                past: int, 
                future: int,
                emb:dict,
                device,
                num_layer1:int = None,
                num_layer2:int = None,
                hid_out_features1:int = None,
                hid_out_features2:int = None):
        
        super(GAT, self).__init__()
        self.hid = hid
        self.in_head = in_head
        self.out_head = out_head
        self.in_feat = in_feat         # numero di features di ogni nodo prima del primo GAT 
        self.device = device
        

        self.Threshold = torch.tensor(0.5, device = device)
        self.connection = torch.tensor(1, device = device)
        self.past = past 
        self.future = future
        # B = batch size
        # I = features in the input
        # O = out preprocessing
        # O'= out gat
        # F = Future
        # P = Past
        # H = hid

        
        ########## PREPROCESSING PART #############
        # devo convertire le variabili categoriche
        # B, T, I ---> B, T, O
        # T = tokens, quindi T in (P,F)
        
        out = 3
        for key in emb.keys():
            out += emb[key][1]
        self.out_preprocess = out
        
        self.pre_processing = pre_processing(in_feat = in_feat, 
                                            hid = hid,
                                            emb = emb, 
                                             device = device)
        
        N = past + future 
        d = N//3
        self.M = torch.nn.Parameter(torch.randn(N, d)) # Matrice di adiacenza
        self.mask = torch.tril(torch.ones(N, N)).to(device)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        
        
        ########## FIRST GAT PART #############
        # B * P, O ---> B * T, O-1
        if num_layer1 == 0:
            self.gnn1 = GCNConv(in_channels = self.out_preprocess, 
                                out_channels = self.out_preprocess-1)     
        else:
            layers = [(GCNConv(in_channels = self.out_preprocess, 
                               out_channels = hid_out_features1), 'x, edge_index -> x')]
            
            for _ in range(max(0,num_layer1-2)):
                layers.append(nn.ReLU(inplace=True))
                layers.append((GCNConv(in_channels = hid_out_features1, 
                                        out_channels = hid_out_features1), 'x, edge_index -> x'))
                
            layers.append(nn.ReLU(inplace=True))
            layers.append((GCNConv(in_channels = hid_out_features1, 
                                   out_channels = self.out_preprocess-1), 'x, edge_index -> x'))    
            
            self.gnn1 = Sequential('x, edge_index', layers)

        ########## KERNELL PART #############
        # B, P, O-1 ---> B, P, H
        
        self.kernel = Kernel(d = self.out_preprocess-1, 
                            hid = self.hid, 
                            past = self.past, 
                            future = self.future)
        #self.W = torch.nn.Parameter(torch.randn(self.out_preprocess-1,self.hid))
        
        
        # weight in R^(B, P, F)
        # interpreto la diffusione dell'informazione tramite colonna
        # la colonna indica quanto i nodi del passato influiscano sul un determinato nodo del futuro
        # tipo matrice di Markov
        
        ########## SECOND GAT PART #############
        
        if num_layer2 == 0:
            self.gnn2 = GCNConv(in_channels = self.out_preprocess, 
                                out_channels = 1)
        else:
            layers = [(GCNConv(in_channels = self.out_preprocess, 
                               out_channels = hid_out_features2), 'x, edge_index -> x')]
            
            for _ in range(max(0,num_layer2-2)):
                layers.append(nn.ReLU(inplace=True))
                layers.append((GCNConv(in_channels = hid_out_features2, 
                                        out_channels = hid_out_features2), 'x, edge_index -> x'))
                
            layers.append(nn.ReLU(inplace=True))
            layers.append((GCNConv(in_channels = hid_out_features2, 
                                   out_channels = 1), 'x, edge_index -> x'))    
            
            self.gnn2 = Sequential('x, edge_index', layers)
                        
    def forward(self, data):
        
        # estraggo i due sottografi      
        # il primo è riferito al passato 
        # il secondo è riferito al futuro
        sb = len(data.batch.unique())
        A = self.relu(torch.matmul(self.M,self.M.T))
        A = self.softmax(A.masked_fill(self.mask == 0, float('-inf')))

        index_sg1 = np.array([np.arange(i*(self.past+self.future), i*(self.past+self.future)+self.past) 
                            for i in range(sb)])
        index_sg2 = np.array([np.arange(i*(self.past+self.future)+self.past, (i+1)*(self.past+self.future)) 
                            for i in range(sb)])
        
        ####### estraggo le sotto connessioni
        sg1 = data.subgraph(torch.from_numpy(index_sg1).reshape(-1).to(self.device))
        sg2 = data.subgraph(torch.from_numpy(index_sg2).reshape(-1).to(self.device))
        
        x = self.pre_processing(sg1.x.to(self.device)).float()
        x2 = self.pre_processing(sg2.x.to(self.device)).float().reshape(-1,self.future, self.out_preprocess)

        ########## FIRST GAT PART #############
        x1 = self.gnn1(x, sg1.edge_index.to(self.device)).reshape(-1,self.past, self.out_preprocess-1)
        
        ########## KERNELL PART #############
        x_kernel = torch.cat((x1,x2[:,:,:-1]),-2)
        y_kernel = x.reshape(-1,self.past, self.out_preprocess)[:,:,-1:]

        yh = self.kernel(x_kernel,y_kernel, A.to(self.device))

        # B,P
        x2 = torch.cat((x2[:,:,:-1],yh),-1)
        ########## SECOND GAT PART #############
        
        out = self.gnn2(x2.reshape(-1,self.out_preprocess), sg2.edge_index.to(self.device)).reshape(-1,self.future)
        x = torch.cat((x.reshape(-1,self.past, self.out_preprocess),x2),-2)
        x = torch.cdist(x,x)
        return out.float(), x, A
    
    def get_density(self):
        A = self.relu(torch.matmul(self.M,self.M.T))
        A = self.softmax(A.masked_fill(self.mask == 0, float('-inf')))
        area = (self.past+self.future)**2-(self.past+self.future)
        density = torch.sum(A)/area
        return density.item()