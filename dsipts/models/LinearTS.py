
from torch import nn
import torch
import pytorch_lightning as pl
from .base import  Base
from .utils import QuantileLossMO, get_device

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean



class LinearTS(Base):

    
    def __init__(self, seq_len,pred_len,channels_past,channels_future,embs,embedding_final,kernel_size_encoder,sum_embs,out_channels,hidden_size,kind='linar',quantiles=[],optim_config=None,scheduler_config=None):
        super(LinearTS, self).__init__()
        self.save_hyperparameters(logger=False)
        #self.device = get_device()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kind = kind
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


        for k in embs:
            self.embs.append(nn.Embedding(k+1,embedding_final))
            emb_channels+=embedding_final
            
            
        if sum_embs and (emb_channels>0):
            emb_channels = embedding_final
            print('Using sum')
        else:
            print('Using stacked')
    
        ## ne faccio uno per ogni canale
        self.linear =  nn.ModuleList()
        if  self.use_quantiles:
            self.loss = QuantileLossMO(quantiles)
            self.mul = 3
        else:
            self.loss = nn.L1Loss()
            self.mul = 1 
            
        if kind=='dlinear':
            self.decompsition = series_decomp(kernel_size_encoder)    
            self.Linear_Trend = nn.ModuleList()
            for _ in range(out_channels):
                self.Linear_Trend.append(nn.Linear(seq_len,pred_len))
            
        
        for _ in range(out_channels):
            self.linear.append(nn.Sequential(nn.Linear(embedding_final*(seq_len+pred_len)+seq_len*channels_past+channels_future*pred_len,hidden_size),
                                                nn.PReLU(),
                                                nn.Dropout(0.2),    
                                                nn.Linear(hidden_size,hidden_size//2), 
                                                nn.PReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(hidden_size//2,hidden_size//4),
                                                nn.PReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(hidden_size//4,hidden_size//8),
                                                nn.PReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(hidden_size//8,self.pred_len*self.mul)))
                               

    def forward(self, batch):
        
        x =  batch['x_num_past'].to(self.device)
        if self.kind=='nlinear':
            idx_target = batch['idx_target'][0]
            x_start = x[:,-1,idx_target]
            ##BxC
            x[:,:,idx_target]-=x_start
        
        if self.kind=='dlinear':
            idx_target = batch['idx_target'][0]
            x_start = x[:,:,idx_target]

            seasonal_init, trend_init = self.decompsition(x_start.permute(0,2,1))
            seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
            
            x[:,:,idx_target] = seasonal_init

            tmp = []
            for j in range(len(self.Linear_Trend)):
               
                tmp.append(self.Linear_Trend[j](trend_init[:,:,j]))

            trend = torch.stack(tmp,2)
            
            
        if 'x_cat_future' in batch.keys():
            cat_future = batch['x_cat_future'].to(self.device)
        if 'x_cat_past' in batch.keys():
            cat_past = batch['x_cat_past'].to(self.device)
        if 'x_num_future' in batch.keys():
            x_future = batch['x_num_past'].to(self.device)
        else:
            x_future = None
            
        tmp = [x]
        
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
        ##BxLxC
        tot_past = torch.cat(tmp,2).flatten(1)
        


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
            tot_future = torch.cat(tmp,2).flatten(1)
            tot = torch.cat([tot_past,tot_future],1)
            
        else:
            tot = tot_past

        res = []
        B = tot.shape[0]
        for j in range(len(self.linear)):
            res.append(self.linear[j](tot).reshape(B,-1,self.mul))
        ## BxLxCxMUL
        res = torch.stack(res,2)

        if self.kind=='nlinear':
            #BxC
            res+=x_start.unsqueeze(1).unsqueeze(3)
        

        if self.kind=='dlinear':
            res = res+trend.unsqueeze(3)
        
        return res
    