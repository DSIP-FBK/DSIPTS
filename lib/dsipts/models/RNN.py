
from torch import optim, nn
import torch
import pickle
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from .base import QuantileLoss, Base
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class Permute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.permute(input,(0,2,1))

class RNN(Base):

    
    def __init__(self, seq_len,pred_len,channels_past,channels_future,embs,embedding_final,hidden_LSTM,num_layers,kernel_size_encoder,sum_embs,out_channels,quantiles=[],optim_config=None,scheduler_config=None):
        super(RNN, self).__init__()
        self.save_hyperparameters(logger=False)
        #self.device = get_device()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.num_layers = num_layers
        self.hidden_LSTM = hidden_LSTM
        self.channels_past = channels_past 
        self.channels_future = channels_future 
        self.embs = nn.ModuleList()
        self.sum_embs = sum_embs
        assert (len(quantiles) ==0) or (len(quantiles)==3)
        if len(quantiles)>0:
            self.use_quantiles = True
        else:
            self.use_quantiles = False
        
        emb_channels = 0
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        if (out_channels>1) and self.use_quantiles:
            print('THis is not implemented, please adjust the code for multiputput quantile')
        for k in embs:
            self.embs.append(nn.Embedding(k+1,embedding_final))
            emb_channels+=embedding_final
            
            
        if sum_embs and (emb_channels>0):
            emb_channels = embedding_final
            print('Using sum')
        else:
            print('Using stacked')
    
        self.initial_linear_encoder =  nn.Sequential(nn.Linear(channels_past,4),nn.PReLU(),nn.Linear(4,8),nn.PReLU(),nn.Linear(8,hidden_LSTM//8))
        self.initial_linear_decoder =  nn.Sequential(nn.Linear(channels_future,4),nn.PReLU(),nn.Linear(4,8),nn.PReLU(),nn.Linear(8,hidden_LSTM//8))
        self.conv_encoder = nn.Sequential(Permute(), nn.Conv1d(emb_channels+hidden_LSTM//8, hidden_LSTM//8, kernel_size_encoder, stride=1,padding='same'),Permute(),nn.Dropout(0.3))
        
        if channels_future+emb_channels>0:
            ## occhio che vuol dire che non ho passato , per ora ci metto una pezza e uso hidden dell'encoder
            self.conv_decoder =  nn.Sequential(nn.Linear(channels_future+emb_channels,hidden_LSTM//4),  nn.PReLU(),nn.Dropout(0.2),nn.Linear(hidden_LSTM//4, hidden_LSTM//8),nn.Dropout(0.3))
        else:
            self.conv_decoder =  nn.Sequential(Permute(),nn.Linear(seq_len,seq_len*2),  nn.PReLU(),nn.Dropout(0.2),nn.Linear(seq_len*2, pred_len),nn.Dropout(0.3),nn.Conv1d(hidden_LSTM, hidden_LSTM//8, 3, stride=1,padding='same'),   Permute())

        self.Encoder = nn.LSTM(input_size= hidden_LSTM//8,hidden_size=hidden_LSTM,num_layers = num_layers,batch_first=True)
        self.Decoder = nn.LSTM(input_size= hidden_LSTM//8,hidden_size=hidden_LSTM,num_layers = num_layers,batch_first=True)
        self.final_linear = nn.ModuleList()
        for _ in range(3 if self.use_quantiles else out_channels):
            self.final_linear.append(nn.Sequential(nn.Linear(hidden_LSTM,hidden_LSTM//2), 
                                            nn.PReLU(),nn.Dropout(0.2),nn.Linear(hidden_LSTM//2,hidden_LSTM//4),
                                            nn.PReLU(),nn.Dropout(0.2),nn.Linear(hidden_LSTM//4,hidden_LSTM//8),
                                            nn.PReLU(),nn.Dropout(0.2),nn.Linear(hidden_LSTM//8,1)))

  
        if  self.use_quantiles:
            self.loss = QuantileLoss(quantiles)
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
            if self.sum_embs:
                if i>0:
                    tmp_emb+=self.embs[i](cat_past[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_past[:,:,i])
            else:
                tmp.append(self.embs[i](cat_past[:,:,i]))
        if self.sum_embs and (len(self.embs)>0):
            tmp.append(tmp_emb)
        tot = torch.cat(tmp,2)
            
        out, hidden = self.Encoder(self.conv_encoder(tot))      

        tmp = []
        for i in range(len(self.embs)):
            if self.sum_embs:
                if i>0:
                    tmp_emb+=self.embs[i](cat_future[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_future[:,:,i])
            else:
                tmp.append(self.embs[i](cat_future[:,:,i]))   
        if self.sum_embs and (len(self.embs)):
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

        return torch.cat(res,2)
    