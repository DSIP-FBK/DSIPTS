## Copyright https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py
## Modified for notation alignmenet and batch structure
## extended to what inside timexer folder

import torch
import torch.nn as nn
import numpy as np



try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base
from .utils import QuantileLossMO,Permute, get_activation
from .itransformer.SelfAttention_Family import FullAttention, AttentionLayer
from .itransformer.Embed import DataEmbedding_inverted
from .timexer.Layers import FlattenHead,EnEmbedding, EncoderLayer, Encoder

from typing import List, Union
from ..data_structure.utils import beauty_string
from .utils import  get_scope
from .utils import Embedding_cat_variables




class TimeXER(Base):
    handle_multivariate = True
    handle_future_covariates = True # or at least it seems...
    handle_categorical_variables = True #solo nel encoder
    handle_quantile_loss = True # NOT EFFICIENTLY ADDED, TODO fix this
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 patch_len:int,
                 d_model: int,
                 n_head: int,
                 d_ff:int=512,
                 dropout_rate: float=0.1,
                 n_layer_decoder: int=1,
                 activation: str='',
                 **kwargs)->None:
        """Initialize the model with specified parameters. https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py
        
        Args:
            patch_len (int): Length of the patches.
            d_model (int): Dimension of the model.
            n_head (int): Number of attention heads.
            d_ff (int, optional): Dimension of the feedforward network. Defaults to 512.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.1.
            n_layer_decoder (int, optional): Number of layers in the decoder. Defaults to 1.
            activation (str, optional): Activation function to use. Defaults to ''.
            **kwargs: Additional keyword arguments passed to the superclass.
        
        Raises:
            ValueError: If an invalid activation function is provided.
        
   
        """
        super().__init__(**kwargs)
        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
            use_bn = False
        if isinstance(activation,str):
            activation = get_activation(activation)
        self.save_hyperparameters(logger=False)


                


        self.patch_len = patch_len
        self.patch_num = int(self.past_steps // patch_len)
        d_model = d_model*self.mul
        
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels

        self.output_attention = False## not need output attention
        
        ###
        self.en_embedding = EnEmbedding(self.past_channels, d_model, patch_len, dropout_rate)

        self.ex_embedding = DataEmbedding_inverted(self.past_steps, d_model, embed_type='what?', freq='what?', dropout=dropout_rate)  ##embed, freq not used inside


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
        self.head = FlattenHead(self.past_channels, self.head_nf, self.future_steps*self.mul,  head_dropout=dropout_rate)
        
        
        self.future_reshape = nn.Linear(self.future_steps,self.future_steps*self.mul)
        self.final_linear = nn.Sequential(activation(),
            nn.Linear(self.past_channels+self.future_channels+emb_fut_out_channel,(self.past_channels+self.future_channels+emb_fut_out_channel)//2),
            activation(),
            nn.Linear((self.past_channels+self.future_channels+emb_fut_out_channel)//2,self.out_channels)  
        )

        

    def can_be_compiled(self):
        return True  
  


    def forward(self, batch:dict)-> float:


        x_enc = batch['x_num_past'].to(self.device)

        BS = x_enc.shape[0]
        if 'x_cat_future' in batch.keys():
            emb_fut = self.emb_fut(BS,batch['x_cat_future'].to(self.device))
        else:
            emb_fut = self.emb_fut(BS,None)
        tmp_future = [emb_fut]
        if 'x_cat_past' in batch.keys():
            emb_past = self.emb_past(BS,batch['x_cat_past'].to(self.device))
        else:
            emb_past = self.emb_past(BS,None)

        if 'x_num_future' in batch.keys():
            x_future = batch['x_num_future'].to(self.device)
            tmp_future.append(x_future)
        if len(tmp_future)>0:
            tmp_future = torch.cat(tmp_future,2)
        else:
            tmp_future = None





        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, emb_past)

        enc_out = self.encoder(en_embed, ex_embed)
        
        


        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        

        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        #dec_out = dec_out.permute(0, 2, 1)
        if tmp_future is not None:
            tmp_future = self.future_reshape(tmp_future.permute(0, 2, 1))
            dec_out = torch.cat([tmp_future,dec_out],1)
        dec_out = self.final_linear(dec_out.permute(0, 2, 1))
        return dec_out.reshape(BS,self.future_steps,self.out_channels,self.mul)
        
        #idx_target = batch['idx_target'][0]
        #return dec_out[:, :,idx_target].reshape(BS,self.future_steps,self.out_channels,self.mul)
        

