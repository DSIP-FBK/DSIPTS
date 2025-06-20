## Copyright https://github.com/romilbert/samformer/tree/main?tab=MIT-1-ov-file#readme
## Modified for notation alignmenet and batch structure
## extended to what inside samformer folder

import torch
import torch.nn as nn
import numpy as np
from .samformer.utils import scaled_dot_product_attention, RevIN


from .base import  Base
from .utils import QuantileLossMO,Permute, get_activation

from typing import List, Union
from ..data_structure.utils import beauty_string
from .utils import  get_scope




class Samformer(Base):
    handle_multivariate = True
    handle_future_covariates = False # or at least it seems...
    handle_categorical_variables = False #solo nel encoder
    handle_quantile_loss = False # NOT EFFICIENTLY ADDED, TODO fix this
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 out_channels: int,
                 past_steps: int,
                 future_steps: int, 
                 past_channels: int,
                 future_channels: int,
                 embs: List[int],

                 # specific params
                 hidden_size:int,
                 use_revin: bool,
                 rho: float=0.5,

                 
                 
                 dropout_rate: float=0.1,
                 activation: str='',
                 
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[float]=[],
                 optim:Union[str,None]=None,
                 optim_config:Union[dict,None]=None,
                 scheduler_config:Union[dict,None]=None,
                 **kwargs)->None:
        """Samformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention.
        https://arxiv.org/pdf/2402.10198

   
        Args:
            out_channels (int): number of variables to be predicted
            past_steps (int): Lookback window length
            future_steps (int): Horizon window length
            past_channels (int): number of past variables
            future_channels (int): number of future auxiliary variables 
            embs (List[int]): list of embeddings
            hidden_size (int): first embedding size of the model ('r' in the paper)
            d_model (int): second embedding size (r^{tilda} in the model). Should be smaller than hidden_size
            n_head (int): number of heads
            n_layer_decoder (int): number layers
            dropout_rate (float): 
            class_strategy (str): strategy (see paper) projection/average/cls_token
            
            activation (str, optional): activation function to be used 'nn.GELU'.
            persistence_weight (float, optional): Defaults to 0.0.
            loss_type (str, optional): Defaults to 'l1'.
            quantiles (List[float], optional): Defaults to []. NOT USED
            optim (Union[str,None], optional): Defaults to None.
            optim_config (Union[dict,None], optional): Defaults to None.
            scheduler_config (Union[dict,None], optional): Defaults to None.
        """
        
        super().__init__(**kwargs)
        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
            use_bn = False
        if isinstance(activation,str):
            activation = get_activation(activation)
        self.save_hyperparameters(logger=False)

        # self.dropout = dropout_rate
        self.persistence_weight = persistence_weight 
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.loss_type = loss_type
        self.future_steps = future_steps
                
        if len(quantiles)==0:
            self.mul = 1
            self.use_quantiles = False
            if self.loss_type == 'mse':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()
        else:
            assert len(quantiles)==3, beauty_string('ONLY 3 quantiles premitted','info',True)
            self.mul = len(quantiles)
            self.use_quantiles = True
            self.loss = QuantileLossMO(quantiles)



         


        self.out_channels = out_channels

        self.revin = RevIN(num_features=past_channels)
        self.compute_keys = nn.Linear(past_steps, hidden_size)
        self.compute_queries = nn.Linear(past_steps, hidden_size)
        self.compute_values = nn.Linear(past_steps, past_steps)
        self.linear_forecaster = nn.Linear(past_steps, future_steps)
        self.use_revin = use_revin

 

    def forward(self, batch:dict)-> float:

        x = batch['x_num_past'].to(self.device)
        BS = x.shape[0]

        if self.use_revin:
            x_norm = self.revin(x.transpose(1, 2), mode='norm').transpose(1, 2) # (n, D, L)
        else:
            x_norm = x
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
            
        
        return out.reshape(BS,self.future_steps,self.out_channels,self.mul)

