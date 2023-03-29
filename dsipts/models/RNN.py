
from torch import  nn
import torch
from .base import Base
from .utils import QuantileLossMO,Permute, get_device



class RNN(Base):

    def __init__(self, past_steps,future_steps,past_channels,future_channels,embs,cat_emb_dim,hidden_RNN,num_layers_RNN,kind,kernel_size_encoder,sum_emb,out_channels,quantiles=[],optim_config=None,scheduler_config=None):
        super(RNN, self).__init__()
        self.save_hyperparameters(logger=False)
        #self.device = get_device()
        self.past_steps = past_steps
        self.future_steps = future_steps

        self.num_layers_RNN = num_layers_RNN
        self.hidden_RNN = hidden_RNN
        self.past_channels = past_channels 
        self.future_channels = future_channels 
        self.embs = nn.ModuleList()
        self.sum_emb = sum_emb
        self.kind = kind
        
        assert (len(quantiles) ==0) or (len(quantiles)==3)
        if len(quantiles)>0:
            self.use_quantiles = True
            self.mul = 3
        else:
            self.use_quantiles = False
            self.mul = 1
        
        emb_channels = 0
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config

        for k in embs:
            self.embs.append(nn.Embedding(k+1,cat_emb_dim))
            emb_channels+=cat_emb_dim
            
            
        if sum_emb and (emb_channels>0):
            emb_channels = cat_emb_dim
            print('Using sum')
        else:
            print('Using stacked')
    
        self.initial_linear_encoder =  nn.Sequential(nn.Linear(past_channels,4),nn.PReLU(),nn.Linear(4,8),nn.PReLU(),nn.Linear(8,hidden_RNN//8))
        self.initial_linear_decoder =  nn.Sequential(nn.Linear(future_channels,4),nn.PReLU(),nn.Linear(4,8),nn.PReLU(),nn.Linear(8,hidden_RNN//8))
        self.conv_encoder = nn.Sequential(Permute(), nn.Conv1d(emb_channels+hidden_RNN//8, hidden_RNN//8, kernel_size_encoder, stride=1,padding='same'),Permute(),nn.Dropout(0.3))
        
        if future_channels+emb_channels>0:
            ## occhio che vuol dire che non ho passato , per ora ci metto una pezza e uso hidden dell'encoder
            self.conv_decoder =  nn.Sequential(nn.Linear(future_channels+emb_channels,hidden_RNN//4),  nn.PReLU(),nn.Dropout(0.2),nn.Linear(hidden_RNN//4, hidden_RNN//8),nn.Dropout(0.3))
        else:
            self.conv_decoder =  nn.Sequential(Permute(),nn.Linear(past_steps,past_steps*2),  nn.PReLU(),nn.Dropout(0.2),nn.Linear(past_steps*2, future_steps),nn.Dropout(0.3),nn.Conv1d(hidden_RNN, hidden_RNN//8, 3, stride=1,padding='same'),   Permute())
        if self.kind=='lstm':
            self.Encoder = nn.LSTM(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
            self.Decoder = nn.LSTM(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
        elif self.kind=='gru':
            self.Encoder = nn.GRU(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
            self.Decoder = nn.GRU(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
        else:
            print('Speciky kind= lstm or gru please')
        self.final_linear = nn.ModuleList()
        for _ in range(out_channels*self.mul):
            self.final_linear.append(nn.Sequential(nn.Linear(hidden_RNN,hidden_RNN//2), 
                                            nn.PReLU(),nn.Dropout(0.2),nn.Linear(hidden_RNN//2,hidden_RNN//4),
                                            nn.PReLU(),nn.Dropout(0.2),nn.Linear(hidden_RNN//4,hidden_RNN//8),
                                            nn.PReLU(),nn.Dropout(0.2),nn.Linear(hidden_RNN//8,1)))

  
        if  self.use_quantiles:
            self.loss = QuantileLossMO(quantiles)
        else:
            self.loss = nn.L1Loss()
        #self.device = get_device()
    def forward(self, batch):
        x =  batch['x_num_past'].to(self.device)

        if 'x_cat_future' in batch.keys():
            cat_future = batch['x_cat_future'].to(self.device)
        if 'x_cat_past' in batch.keys():
            cat_past = batch['x_cat_past'].to(self.device)
        if 'x_num_future' in batch.keys():
            x_future = batch['x_num_past'].to(self.device)
        else:
            x_future = None
            
        tmp = [self.initial_linear_encoder(x)]
        
        for i in range(len(self.embs)):
            if self.sum_emb:
                if i>0:
                    tmp_emb+=self.embs[i](cat_past[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_past[:,:,i])
            else:
                tmp.append(self.embs[i](cat_past[:,:,i]))
        if self.sum_emb and (len(self.embs)>0):
            tmp.append(tmp_emb)
        tot = torch.cat(tmp,2)
            
        out, hidden = self.Encoder(self.conv_encoder(tot))      

        tmp = []
        for i in range(len(self.embs)):
            if self.sum_emb:
                if i>0:
                    tmp_emb+=self.embs[i](cat_future[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_future[:,:,i])
            else:
                tmp.append(self.embs[i](cat_future[:,:,i]))   
        if self.sum_emb and (len(self.embs)):
            tmp.append(tmp_emb)
            
        if x_future is not None:
            tmp.append(x_future)
        if len(tmp)>0:
            tot = torch.cat(tmp,2)
            out, _ = self.Decoder(self.conv_decoder(tot),hidden)  
        else:
            out, _ = self.Decoder(self.conv_decoder(out),hidden)  
        res = []

      
        for j in range(len(self.final_linear)):
            res.append(self.final_linear[j](out))
            
        res = torch.cat(res,2)
        ##BxLxC
        B,L,_ = res.shape
        return res.reshape(B,L,-1,self.mul)
    