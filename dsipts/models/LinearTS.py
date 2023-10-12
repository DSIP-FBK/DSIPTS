
from torch import nn
import torch
from .base import  Base
from .utils import QuantileLossMO, get_activation
from typing import List, Union
from ..data_structure.utils import beauty_string

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

    
    def __init__(self, 
                 past_steps:int,
                 future_steps:int,
                 past_channels:int,
                 future_channels:int,
                 embs:List[int],
                 cat_emb_dim:int,
                 kernel_size:int,
                 sum_emb:bool,
                 out_channels:int,
                 hidden_size:int,
                 dropout_rate:float=0.1,
                 activation:str='torch.nn.ReLU',
                 kind:str='linear',
                 use_bn:bool=False,
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[int]=[],
                 n_classes:int=0,
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None,
                 simple:bool=False,
                 **kwargs)->None:
        """Linear model from https://github.com/cure-lab/LTSF-Linear/blob/main/run_longExp.py

        Args:
            past_steps (int): number of past datapoints used 
            future_steps (int): number of future lag to predict
            past_channels (int):  number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            embs (List[int]): list of the initial dimension of the categorical variables
            cat_emb_dim (int): final dimension of each categorical variable
            kernel_size (int): kernel dimension for initial moving average
            sum_emb (bool): if true the contribution of each embedding will be summed-up otherwise stacked
            out_channels (int): number of output channels
            hidden_size (int): hidden size of the lienar block
            dropout_rate (float, optional): dropout rate in Dropout layers. Default 0.1
            activation (str, optional): activation fuction function pytorch. Default torch.nn.ReLU
            kind (str, optional): one among linear, dlinear (de-trending), nlinear (differential). Defaults to 'linear'.
            use_bn (bool, optional): if true BN layers will be added and dropouts will be removed. Default False
            quantiles (List[int], optional):  we can use quantile loss il len(quantiles) = 0 (usually 0.1,0.5, 0.9) or L1loss in case len(quantiles)==0. Defaults to [].
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            n_classes (int): number of classes (0 in regression)
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
            simple (bool, optional): if simple, the model used is the same that the one illustrated in the paper, otherwise it is a more complicated one with the same idea
        """
        super().__init__(**kwargs)

        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
            use_bn = False
            
        if isinstance(activation, str):
            activation = get_activation(activation)
        else:
            beauty_string('There is a bug in pytorch lightening, the constructior is called twice','info',self.verbose)
        
        self.save_hyperparameters(logger=False)
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.kind = kind
        self.past_channels = past_channels 
        self.future_channels = future_channels 
        self.embs = nn.ModuleList()
        self.sum_emb = sum_emb
        self.persistence_weight = persistence_weight 
        self.loss_type = loss_type
        self.simple = simple
        if n_classes==0:
            self.is_classification = False
            if len(quantiles)>0:
                self.use_quantiles = True
                self.mul = len(quantiles)
                self.loss = QuantileLossMO(quantiles)
            else:
                self.use_quantiles = False
                self.mul = 1
                if self.loss_type == 'mse':
                    self.loss = nn.MSELoss()
                else:
                    self.loss = nn.L1Loss()
        else:
            self.is_classification = True
            self.use_quantiles = False
            self.mul = n_classes
            self.loss = torch.nn.CrossEntropyLoss()
            #assert out_channels==1, "Classification require only one channel"
        
        emb_channels = 0
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config


        for k in embs:
            self.embs.append(nn.Embedding(k+1,cat_emb_dim))
            emb_channels+=cat_emb_dim
            
            
        if sum_emb and (emb_channels>0):
            emb_channels = cat_emb_dim
            beauty_string('Using sum','info',self.verbose)
        else:
            beauty_string('Using stacked',"info",self.verbose)
    
        ## ne faccio uno per ogni canale
        self.linear =  nn.ModuleList()

            
        if kind=='dlinear':
            self.decompsition = series_decomp(kernel_size)    
            self.Linear_Trend = nn.ModuleList()
            for _ in range(out_channels):
                self.Linear_Trend.append(nn.Linear(past_steps,future_steps))
            
        
        for _ in range(out_channels):
            if simple:
                self.linear.append(nn.Linear(past_steps,self.future_steps*self.mul))
                                               
            else:
                self.linear.append(nn.Sequential(nn.Linear(emb_channels*(past_steps+future_steps)+past_steps*past_channels+future_channels*future_steps,hidden_size),
                                                    activation(),
                                                    nn.BatchNorm1d(hidden_size) if use_bn else nn.Dropout(dropout_rate) ,    
                                                    nn.Linear(hidden_size,hidden_size//2), 
                                                    activation(),
                                                    nn.BatchNorm1d(hidden_size//2) if use_bn else nn.Dropout(dropout_rate) ,    
                                                    nn.Linear(hidden_size//2,hidden_size//4),
                                                    activation(),
                                                    nn.BatchNorm1d(hidden_size//4) if use_bn else nn.Dropout(dropout_rate) ,    
                                                    nn.Linear(hidden_size//4,hidden_size//8),
                                                    activation(),
                                                    nn.BatchNorm1d(hidden_size//8) if use_bn else nn.Dropout(dropout_rate) ,    
                                                    nn.Linear(hidden_size//8,self.future_steps*self.mul)))
                               
    def forward(self, batch):
      
        x =  batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        if self.kind=='nlinear':
            
            x_start = x[:,-1,idx_target].unsqueeze(1)
            ##BxC
            x[:,:,idx_target]-=x_start
        
        if self.kind=='alinear':
            x[:,:,idx_target]=0
        
        if self.kind=='dlinear':
            x_start = x[:,:,idx_target]
            seasonal_init, trend_init = self.decompsition(x_start)
            seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
            x[:,:,idx_target] = seasonal_init.permute(0,2,1)
            tmp = []
            for j in range(len(self.Linear_Trend)):
               
                tmp.append(self.Linear_Trend[j](trend_init[:,j,:]))

            trend = torch.stack(tmp,2)
            
        if self.simple is False:
            if 'x_cat_future' in batch.keys():
                cat_future = batch['x_cat_future'].to(self.device)
            if 'x_cat_past' in batch.keys():
                cat_past = batch['x_cat_past'].to(self.device)
            if 'x_num_future' in batch.keys():
                x_future = batch['x_num_future'].to(self.device)
            else:
                x_future = None
                
            tmp = [x]
    
            tmp_emb = None
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
            ##BxLxC
            tot_past = torch.cat(tmp,2).flatten(1)
        


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

                tot_future = torch.cat(tmp,2).flatten(1)
                tot = torch.cat([tot_past,tot_future],1)
                
            else:
                tot = tot_past
            tot = tot.unsqueeze(2).repeat(1,1,len(self.linear)).permute(0,2,1)
        else:
            tot = seasonal_init
        res = []
        B = tot.shape[0]
    
        for j in range(len(self.linear)):
            res.append(self.linear[j](tot[:,j,:]).reshape(B,-1,self.mul))
        ## BxLxCxMUL
        res = torch.stack(res,2)

        if self.kind=='nlinear':
            #res BxLxCx3
            #start BxCx1
            res+=x_start.unsqueeze(1)
        

        if self.kind=='dlinear':
            res = res+trend.unsqueeze(3)
        
            
        return res
    