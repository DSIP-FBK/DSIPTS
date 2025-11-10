
## Copyright 2022 DLinear Authors (https://github.com/cure-lab/LTSF-Linear/tree/main?tab=Apache-2.0-1-ov-file#readme)
## Code modified for align the notation and the batch generation
## extended to all present in informer, autoformer folder

from torch import nn
import torch

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base
from .utils import QuantileLossMO, get_activation
from typing import List, Union
from ..data_structure.utils import beauty_string
from .utils import  get_scope
from .utils import Embedding_cat_variables
    


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
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    description+='\n THE SIMPLE IMPLEMENTATION DOES NOT USE CATEGORICAL NOR FUTURE VARIABLES'
    
    def __init__(self, 
               
                 kernel_size:int,
                 hidden_size:int,
                 dropout_rate:float=0.1,
                 activation:str='torch.nn.ReLU',
                 kind:str='linear',
                 use_bn:bool=False,
                 simple:bool=False,
                 **kwargs)->None:
        """Initialize the model with specified parameters. Linear model from https://github.com/cure-lab/LTSF-Linear/blob/main/run_longExp.py
        
        Args:
            kernel_size (int): Kernel dimension for the initial moving average.
            hidden_size (int): Hidden size of the linear block.
            dropout_rate (float, optional): Dropout rate in Dropout layers. Default is 0.1.
            activation (str, optional): Activation function in PyTorch. Default is 'torch.nn.ReLU'.
            kind (str, optional): Type of model, can be 'linear', 'dlinear' (de-trending), or 'nlinear' (differential). Defaults to 'linear'.
            use_bn (bool, optional): If True, Batch Normalization layers will be added and Dropouts will be removed. Default is False.
            simple (bool, optional): If True, the model used is the same as illustrated in the paper; otherwise, a more complex model with the same idea is used. Default is False.
            **kwargs: Additional keyword arguments for the parent class.
        
        Raises:
            ValueError: If an invalid activation function is provided.
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
      
        self.kind = kind
       

        self.simple = simple
       
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels



        ## ne faccio uno per ogni canale
        self.linear =  nn.ModuleList()

            
        if kind=='dlinear':
            self.decompsition = series_decomp(kernel_size)    
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.out_channels):
                self.Linear_Trend.append(nn.Linear(self.past_steps,self.future_steps))
            
        
        for _ in range(self.out_channels):
            if simple:
                self.linear.append(nn.Linear(self.past_steps,self.future_steps*self.mul))
                                               
            else:
                self.linear.append(nn.Sequential(nn.Linear(emb_past_out_channel*self.past_steps+emb_fut_out_channel*self.future_steps+self.past_steps*self.past_channels+self.future_channels*self.future_steps,hidden_size),
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
    def can_be_compiled(self):
        return True                            
    def forward(self, batch):
      
        x =  batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        
        BS = x.shape[0]
        if 'x_cat_future' in batch.keys():
            emb_fut = self.emb_fut(BS,batch['x_cat_future'].to(self.device))
        else:
            emb_fut = self.emb_fut(BS,None)
        if 'x_cat_past' in batch.keys():
            emb_past = self.emb_past(BS,batch['x_cat_past'].to(self.device))
        else:
            emb_past = self.emb_past(BS,None)
        
        if self.kind=='nlinear':
            
            x_start = x[:,-1,idx_target].unsqueeze(1)
            x[:,:,idx_target]-=x_start
        
        if self.kind=='alinear':
            x[:,:,idx_target] = 0
        
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
            if 'x_num_future' in batch.keys():
                x_future = batch['x_num_future'].to(self.device)
            else:
                x_future = None
                
            tmp = [x,emb_past]
            tot_past = torch.cat(tmp,2).flatten(1)
        


            tmp = [emb_fut]
                          
            if x_future is not None:
                tmp.append(x_future)
           
            tot_future = torch.cat(tmp,2).flatten(1)
            tot = torch.cat([tot_past,tot_future],1)
                
            tot = tot.unsqueeze(2).repeat(1,1,len(self.linear)).permute(0,2,1)
        else:
            tot = x.permute(0,2,1)
        res = []

        for j in range(len(self.linear)):
            res.append(self.linear[j](tot[:,j,:]).reshape(BS,-1,self.mul))
        ## BxLxCxMUL
        res = torch.stack(res,2)

        if self.kind=='nlinear':
            #res BxLxCx3
            #start BxCx1
            res+=x_start.unsqueeze(1)
        

        if self.kind=='dlinear':
            res = res+trend.unsqueeze(3)
        
            
        return res
    