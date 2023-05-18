import torch
import torch.nn as nn

class pre_processing(torch.nn.Module):
    def __init__(self, 
                 in_feat:int, 
                 emb:dict,
                 device):
        super(pre_processing, self).__init__()
        self.in_feat = in_feat
        self.emb = nn.ModuleDict({})
        out = 0
        for key in emb.keys():
            self.emb[key] = nn.Embedding(num_embeddings = emb[key][0],
                                         embedding_dim = emb[key][1], 
                                         device=device)
            out += emb[key][1]
        
        # sommo le tre variabili non categoriche dell'input
        out += 3
        
        self.norm = nn.BatchNorm1d(265)
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
            out.append(self.emb[key](x[:, :,i].int().to(x.device.type)))
        out = torch.cat(out,-1)    
        out = self.norm(out.float())
        out = torch.cat((out, x[:,:,-3:]),-1)
        out = self.linear(out.float())
        out = torch.cat((out, x[:,:,-1:]),-1)
        
        return out.float()

    

class my_gcnn(torch.nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 relu: bool = True):
        super(my_gcnn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, 
                             out_channels, 
                             bias = False)
        self.relu = nn.ReLU()
        if relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        

    def forward(self,
                x0: tuple) -> torch.tensor:
        x, A = x0
        # Inserisco una threshold
        # se i valori sono più piccoli di un tot significa che non ci è connessione
        D = torch.sign(self.relu(A - 1e-15))
        D = torch.diag((1+torch.sum(D , 1))**(-0.5))
        node_start = A.shape[0]
        node_end = A.shape[1]
        
        if node_start == node_end:
            A_tilde = A + torch.diag(torch.ones(node_start)).to(A.device.type)
            D1 = D2 = D
        else:
            I = torch.diag(torch.ones(node_end))[node_end-node_start:]
            A_tilde = A+I.to(A.device.type)
            D1 = D**2
            D2 = torch.diag(torch.ones(node_end)).to(A.device.type)
            
        x = self.lin(x)
        out = torch.matmul(D1, A_tilde)
        out = torch.matmul(out,D2)
        out = self.act(torch.matmul(out,x))
        
        return (out.float(), A)

    
class GAT(torch.nn.Module):
    def __init__(self, 
                in_feat:int, 
                past: int, 
                future: int,
                emb:dict,
                device,
                nfeat_out_gnn:int = 16,
                num_layer1:int = None,
                num_layer2:int = None,
                hid_out_features1:int = None,
                hid_out_features2:int = None):
        
        super(GAT, self).__init__()
        
        self.device = device
        self.out_gnn = nfeat_out_gnn
        self.hid_out_features1 = hid_out_features1
        self.hid_out_features2 = hid_out_features2
        self.num_layer1 = num_layer1
        self.num_layer2 = num_layer2
        
        self.in_feat = in_feat         # numero di features di ogni nodo prima del primo GAT        
        self.past = past 
        self.future = future

        
        ########## PREPROCESSING PART #############
        out = 3
        for key in emb.keys():
            out += emb[key][1]
        self.out_preprocess = out
        self.pre_processing = pre_processing(in_feat = in_feat, 
                                            emb = emb, 
                                            device = device)
        
        N = past + future
        self.M = torch.nn.Parameter(torch.randn(N, N)) # Matrice di adiacenza
        self.mask = torch.tril(torch.ones(N, N))-torch.diag(torch.ones(N))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        
        
        ########## FIRST GAT PART #############
        # B * P, O ---> B * T, O-1
        if num_layer1 in [0,1]:
            self.gnn1 = my_gcnn(in_channels = self.out_preprocess, 
                                out_channels = self.out_gnn, 
                                relu = False)     
        else:
            layers = [my_gcnn(in_channels = self.out_preprocess, 
                                      out_channels = hid_out_features1)]
            
            for _ in range(max(0,num_layer1-2)):
                layers.append(my_gcnn(in_channels = hid_out_features1, 
                                      out_channels = hid_out_features1))
                
            layers.append(my_gcnn(in_channels = hid_out_features1, 
                                  out_channels = self.out_gnn,
                                  relu= False))    
            
            self.gnn1 = nn.Sequential(*layers)
            
        ########## SECOND GAT PART #############
        
        if num_layer2 in [0, 1]:
            self.gnn2 = my_gcnn(in_channels = self.out_preprocess-1, 
                                out_channels = self.out_gnn, 
                                relu = False)
        else:
            layers = [my_gcnn(in_channels = self.out_preprocess-1, 
                                      out_channels = hid_out_features2)]
            
            for _ in range(max(0,num_layer2-2)):
                layers.append(my_gcnn(in_channels = hid_out_features2, 
                                      out_channels = hid_out_features2))
                
            layers.append(my_gcnn(in_channels = hid_out_features2, 
                                  out_channels = self.out_gnn, 
                                  relu = False))    
            
            self.gnn2 = nn.Sequential(*layers)
        
        self.prop_gnn = my_gcnn(in_channels = self.out_gnn, 
                                out_channels = self.out_gnn, 
                                relu = False)
        self.propagation = nn.Sequential(nn.Flatten(-2),
                                         nn.BatchNorm1d(self.out_gnn*self.future),
                                         nn.Linear(in_features = self.out_gnn*self.future, 
                                                   out_features = 128),
                                         nn.ReLU(), 
                                         nn.Linear(in_features = 128,
                                                   out_features = 128),
                                         nn.ReLU(),
                                         nn.Linear(in_features = 128,
                                                   out_features = self.future))
               
    def forward(self, data):

        A = self.relu(torch.matmul(self.M,self.M.T))
        A = self.softmax(A.masked_fill(self.mask.to(A.device.type) == 0, float('-inf')))
        A = A.masked_fill(self.mask.to(A.device.type) == 0, 0)
        
        # pre-processing dei dati
        x = self.pre_processing(data)
        
        # creation of the 2 subgraph
        sg1 = x[:,:self.past,:]
        sg2 = x[:,self.past:,:]

        ########## GNN processing ######################
        
        x1, _ = self.gnn1((sg1, A[:self.past, :self.past]))
        x2, _ = self.gnn2((sg2[:,:,:-1], A[self.past:, self.past:]))
        x = torch.cat((x1,x2),-2)
        
        ########## PROPAGATION PART ####################
        D = torch.sign(self.relu(A - 1e-15))
        D = torch.diag((1+torch.sum(D , 1))**(-0.5))
        
        future, _ = self.prop_gnn((x, A[self.past:,:]))
        future = self.propagation(future)
        
        ########## PREPARE THE DIRICHLET ENERGY ##########
        x = torch.matmul(D,x)
        x = torch.cdist(x,x)        
        
        return future.float(), x, A