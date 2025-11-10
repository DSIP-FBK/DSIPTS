## Copyright https://github.com/romilbert/samformer/tree/main?tab=MIT-1-ov-file#readme
## Modified for notation alignmenet and batch structure
## extended to what inside samformer folder

import torch
import torch.nn as nn
import numpy as np
from .samformer.utils import scaled_dot_product_attention, RevIN



try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base
from .utils import QuantileLossMO,Permute, get_activation

from typing import List, Union
from ..data_structure.utils import beauty_string
from .utils import  get_scope
from .utils import Embedding_cat_variables




class Samformer(Base):
    handle_multivariate = True
    handle_future_covariates = False # or at least it seems...
    handle_categorical_variables = False #solo nel encoder
    handle_quantile_loss = False # NOT EFFICIENTLY ADDED, TODO fix this
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
               
                 # specific params
                 hidden_size:int,
                 use_revin: bool,
                 activation: str='',
                 
                 **kwargs)->None:
        """Initialize the model with specified parameters. Samformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention.
        https://arxiv.org/pdf/2402.10198
        
        Args:
            hidden_size (int): The size of the hidden layer.
            use_revin (bool): Flag indicating whether to use RevIN.
            activation (str, optional): The activation function to use. Defaults to ''.
            **kwargs: Additional keyword arguments passed to the parent class.
        
        Raises:
            ValueError: If the activation function is not recognized.
        

        """
        
        super().__init__(**kwargs)
        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
            use_bn = False
        if isinstance(activation,str):
            activation = get_activation(activation)
        self.save_hyperparameters(logger=False)


        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels
        
        
        self.revin = RevIN(num_features=self.past_channels+emb_past_out_channel)
        self.compute_keys = nn.Linear(self.past_steps, hidden_size)
        self.compute_queries = nn.Linear(self.past_steps, hidden_size)
        self.compute_values = nn.Linear(self.past_steps, self.past_steps)
        self.linear_forecaster = nn.Linear(self.past_steps, self.future_steps)
        self.use_revin = use_revin

        dim = emb_past_out_channel+self.past_channels+emb_fut_out_channel+self.future_channels
        self.final_layer = nn.Sequential(activation(),
                                         nn.Linear(dim, dim*2),
                                         activation(),
                                         nn.Linear(dim*2,self.out_channels*self.mul  ))

    def can_be_compiled(self):
        return True  
  
    def forward(self, batch:dict)-> float:

        x = batch['x_num_past'].to(self.device)
        BS = x.shape[0]
        if 'x_cat_future' in batch.keys():
            emb_fut = self.emb_fut(BS,batch['x_cat_future'].to(self.device))
        else:
            emb_fut = self.emb_fut(BS,None)
        if 'x_cat_past' in batch.keys():
            emb_past = self.emb_past(BS,batch['x_cat_past'].to(self.device))
        else:
            emb_past = self.emb_past(BS,None)
            
        tmp_future = [emb_fut]
        if 'x_num_future' in batch.keys():
            x_future = batch['x_num_future'].to(self.device)
            tmp_future.append(x_future)
            
        tot = [x,emb_past]
        x = torch.cat(tot,axis=2)

        
 

        if self.use_revin:
            x_norm = self.revin(x, mode='norm').transpose(1, 2) # (n, D, L)
        else:
            x_norm = x.transpose(1, 2)
        # Channel-Wise Attention

        queries = self.compute_queries(x_norm) # (n, D, hid_dim)
        keys = self.compute_keys(x_norm) # (n, D, hid_dim)
        values = self.compute_values(x_norm) # (n, D, L)
        if hasattr(nn.functional, 'scaled_dot_product_attention'):
            att_score = nn.functional.scaled_dot_product_attention(queries, keys, values) # (n, D, L)
        else:
            att_score = scaled_dot_product_attention(queries, keys, values) # (n, D, L)
        out = x_norm + att_score # (n, D, L)
        # Linear Forecasting
        out = self.linear_forecaster(out) # (n, D, H)
        # RevIN Denormalization
        if self.use_revin:
            out = self.revin(out.transpose(1, 2), mode='denorm').transpose(1, 2) # (n, D, H)


        tmp_future.append(out.permute(0,2,1))
        tmp_future = torch.cat(tmp_future,2)
        output = self.final_layer(tmp_future)

        return output.reshape(BS,self.future_steps,self.out_channels,self.mul)

