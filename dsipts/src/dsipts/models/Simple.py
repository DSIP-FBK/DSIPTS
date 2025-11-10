
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
    



class Simple(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    description+='\n THE SIMPLE IMPLEMENTATION DOES NOT USE CATEGORICAL NOR FUTURE VARIABLES'
    
    def __init__(self, 
               
                 hidden_size:int,
                 dropout_rate:float=0.1,
                 activation:str='torch.nn.ReLU',


                 **kwargs)->None:
       
        super().__init__(**kwargs)

        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
            use_bn = False
            
        if isinstance(activation, str):
            activation = get_activation(activation)
        else:
            beauty_string('There is a bug in pytorch lightening, the constructior is called twice','info',self.verbose)
        
        self.save_hyperparameters(logger=False)
      
        
       
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels





        self.linear = (nn.Sequential(nn.Linear(emb_past_out_channel*self.past_steps+emb_fut_out_channel*self.future_steps+self.past_steps*self.past_channels+self.future_channels*self.future_steps,hidden_size),
                                                    activation(),nn.Dropout(dropout_rate),
                                                    nn.Linear(hidden_size,self.out_channels*self.future_steps*self.mul)))
    def can_be_compiled(self):
        return True  
                             
    def forward(self, batch):
      
        x =  batch['x_num_past'].to(self.device)
        
        BS = x.shape[0]
        if 'x_cat_future' in batch.keys():
            emb_fut = self.emb_fut(BS,batch['x_cat_future'].to(self.device))
        else:
            emb_fut = self.emb_fut(BS,None)
        if 'x_cat_past' in batch.keys():
            emb_past = self.emb_past(BS,batch['x_cat_past'].to(self.device))
        else:
            emb_past = self.emb_past(BS,None)
        
        
            

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

                    
        res = self.linear(tot)
        res = res.reshape(BS,self.future_steps,-1,self.mul)
       
        ##

        return res
    