## Copyright https://github.com/thuml/iTransformer?tab=MIT-1-ov-file#readme
## Modified for notation alignmenet and batch structure
## extended to what inside itransformer folder

import torch
import torch.nn as nn
import numpy as np
from .timesnet.Layers import TimesBlock
from ..data_structure.utils import beauty_string
from .utils import  get_scope,get_activation,Embedding_cat_variables

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base



class TimesNet(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 # specific params
                 e_layers:int,
                 d_model: int,
                 top_k: int,
                 d_ff: int,
                 num_kernels: int,
                 activation: str='',
                 **kwargs)->None:
             
        
        
        
        super().__init__(**kwargs)
        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
            use_bn = False
        if isinstance(activation,str):
            activation = get_activation(activation)
        self.save_hyperparameters(logger=False)
        self.e_layers = e_layers
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        #self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        #emb_fut_out_channel = self.emb_fut.output_channels
        channels = self.mul*(emb_past_out_channel+self.past_channels)
        self.model = nn.ModuleList([TimesBlock(self.past_steps,self.future_steps,top_k,channels,d_ff,num_kernels) for _ in range(e_layers)])
        self.layer_norm = nn.LayerNorm(channels)

        self.predict_linear = nn.Linear(self.past_steps, self.future_steps + self.past_steps)
  
        self.projection = nn.Linear(channels, self.out_channels*self.mul, bias=True)
  
    def can_be_compiled(self):
        return False#True  

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        #means = x_enc.mean(1, keepdim=True).detach()
        #x_enc = x_enc.sub(means)
        #stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #x_enc = x_enc.div(stdev)

        # embedding
        enc_out = torch.cat([x_enc, x_mark_enc],axis=2)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        # TimesNet

        for i in range(self.e_layers):
            if i ==0:
                enc_out = self.layer_norm(self.model[i](enc_out.repeat(1,1,self.mul)))
            else:
                enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        #dec_out = dec_out.mul((stdev[:, 0, :].unsqueeze(1).repeat(1, self.future_steps + self.past_steps, 1)))
        #dec_out = dec_out.add((means[:, 0, :].unsqueeze(1).repeat(1, self.future_steps + self.past_steps, 1)))
        return dec_out

    def forward(self, batch:dict)-> float:

        x_enc = batch['x_num_past'].to(self.device)
        BS = x_enc.shape[0]
        if 'x_cat_past' in batch.keys():
            emb_past = self.emb_past(BS,batch['x_cat_past'].to(self.device))
        else:
            emb_past = self.emb_past(BS,None)

        dec_out = self.forecast(x_enc, emb_past, None, None)

        return dec_out[:, -self.future_steps:,:].reshape(BS,self.future_steps,self.out_channels,self.mul)
