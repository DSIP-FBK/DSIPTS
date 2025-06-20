## Copyright https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py
## Modified for notation alignmenet and batch structure
## extended to what inside timexer folder

import torch
import torch.nn as nn
import numpy as np


from .base import  Base
from .utils import QuantileLossMO,Permute, get_activation
from .itransformer.SelfAttention_Family import FullAttention, AttentionLayer
from .itransformer.Embed import DataEmbedding_inverted
from .timexer.Layers import FlattenHead,EnEmbedding, EncoderLayer, Encoder

from typing import List, Union
from ..data_structure.utils import beauty_string
from .utils import  get_scope




class TimeXER(Base):
    handle_multivariate = True
    handle_future_covariates = True # or at least it seems...
    handle_categorical_variables = True #solo nel encoder
    handle_quantile_loss = True # NOT EFFICIENTLY ADDED, TODO fix this
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 out_channels: int,
                 past_steps: int,
                 future_steps: int, 
                 past_channels: int,
                 future_channels: int,
                 embs: List[int],

                 # specific params
                 patch_len:int,
                 d_model: int,

                 n_head: int,
                 d_ff:int=512,
                 dropout_rate: float=0.1,
                 
                 
           
                 n_layer_decoder: int=1,

                 activation: str='',
                 
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[float]=[],
                 optim:Union[str,None]=None,
                 optim_config:Union[dict,None]=None,
                 scheduler_config:Union[dict,None]=None,
                 **kwargs)->None:
        """https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py

   
        Args:
            UPDATE THOSE
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
        self.optim = optim
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



        self.patch_len = patch_len
        self.patch_num = int(past_steps // patch_len)
        ##my update
        self.embs = nn.ModuleList()

        
        d_model = d_model*self.mul
        
        for k in embs:
            self.embs.append(nn.Embedding(k+1,d_model))
         
        


        self.out_channels = out_channels
        self.seq_len = past_steps
        self.pred_len = future_steps
        self.output_attention = False## not need output attention
        
        ###
        self.en_embedding = EnEmbedding(past_channels, d_model, patch_len, dropout_rate)

        self.ex_embedding = DataEmbedding_inverted(past_steps, d_model, embed_type='what?', freq='what?', dropout=dropout_rate)  ##embed, freq not used inside


        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor = 0.1, attention_dropout=dropout_rate, ##NB factor is not used
                                      output_attention=False),
                        d_model, n_head),
                    AttentionLayer(
                        FullAttention(False, 0.1, attention_dropout=dropout_rate,
                                      output_attention=False),
                        d_model, n_head),
                    d_model,
                    d_ff,
                    dropout=dropout_rate,
                    activation=activation(),
                )
                for l in range(n_layer_decoder)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(past_channels, self.head_nf, future_steps*self.mul,  head_dropout=dropout_rate)
        
        

        




    def forward(self, batch:dict)-> float:

        x_enc = batch['x_num_past'].to(self.device)
        
        if 'x_cat_past' in batch.keys():
            x_mark_enc =  batch['x_cat_past'].to(self.device)
            tmp = []
            for i in range(len(self.embs)):
                tmp.append(self.embs[i](x_mark_enc[:,:,i]))
            x_mark_enc = torch.cat(tmp,2)
            
        else:
            x_mark_enc = None  
            
        #if 'x_num_future' in batch.keys():
        #    x_dec = batch['x_num_future'].to(self.device)
        #else:
        #    x_dec = None    
            
        #if 'x_cat_future' in batch.keys():
        #    x_mark_dec =  batch['x_cat_future'].to(self.device)
        #    tmp = []
        #    for i in range(len(self.embs)):
        #        tmp.append(self.embs[i](x_mark_dec[:,:,i]))
        #    x_mark_dec = torch.cat(tmp,2)
            
        #else:
        #    x_mark_dec = None  
        




        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        
        


        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        

        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        BS = x_enc.shape[0]


        idx_target = batch['idx_target'][0]
        return dec_out[:, :,idx_target].reshape(BS,self.future_steps,self.out_channels,self.mul)
        

