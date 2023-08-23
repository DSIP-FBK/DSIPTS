
from torch import  nn
import torch
from .base import Base
from .utils import QuantileLossMO,Permute, get_device,L1Loss, get_activation
from typing import List,Union
import numpy as np
import logging

import torch.nn.functional as F
from einops import rearrange, repeat

from .crossformer.cross_encoder import Encoder
from .crossformer.cross_decoder import Decoder
from .crossformer.cross_embed import DSW_embedding

from math import ceil
  
  
#    self, past_channels, past_steps, future_steps, seg_len, win_size = 4,
#                factor=10, d_model=512, hidden_size = 1024, n_head=8, n_layer_encoder=3, 
#                dropout=0.0, baseline = False,
  
class CrossFormer(Base):

    
    def __init__(self, 
                 past_steps:int,
                 future_steps:int,
                 past_channels:int,
                 future_channels:int,
                 d_model:int,
                 embs:List[int],
                 hidden_size:int,
                 n_head:int,
                 seg_len:int,
                 n_layer_encoder:int,
                 win_size:int,
                 out_channels:int,
                 factor:int=5,
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[int]=[],
                 dropout_rate:float=0.1,
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None)->None:
      
   
        super(CrossFormer, self).__init__()
        self.save_hyperparameters(logger=False)
        self.use_quantiles = False
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.loss_type = loss_type
        self.persistence_weight = persistence_weight 
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

        self.future_steps = future_steps
        

        # The padding operation to handle invisible sgemnet length
        self.pad_past_steps = ceil(1.0 *past_steps / seg_len) * seg_len
        self.pad_future_steps = ceil(1.0 * future_steps / seg_len) * seg_len
        self.past_steps_add = self.pad_past_steps - past_steps

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, past_channels, (self.pad_past_steps // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(n_layer_encoder, win_size, d_model, n_head, hidden_size, block_depth = 1, \
                                    dropout = dropout_rate,in_seg_num = (self.pad_past_steps // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, past_channels, (self.pad_future_steps // seg_len), d_model))
        self.decoder = Decoder(seg_len, n_layer_encoder + 1, d_model, n_head, hidden_size, dropout_rate, \
                                    out_seg_num = (self.pad_future_steps // seg_len), factor = factor)
        
    def forward(self, batch):
        idx_target = batch['idx_target'][0]
        x_seq = batch['x_num_past']#[:,:,idx_target]
        
      
 
        batch_size = x_seq.shape[0]
        if (self.past_steps_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.past_steps_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)


        return predict_y[:, :self.future_steps,idx_target].unsqueeze(3)