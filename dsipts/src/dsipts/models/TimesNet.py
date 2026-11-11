## Copyright https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py
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
    if pl.__version__>='2.0.0':
        from .base_v2 import Base
        OLD_PL = False
    else:
        from .base import Base
        OLD_PL = True



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
                 **kwargs)->None:
             
        
        
        
        super().__init__(**kwargs)
 
        self.save_hyperparameters(logger=False)
        self.e_layers = e_layers
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        ## future covariates (numerical + categorical embedding) are placed on the future part of the extended
        ## sequence (past_steps+future_steps) produced by predict_linear, zero on the past part, so they go through
        ## the 2D TimesBlocks together with the past. They do not need to exist in the past and can have any
        ## number of channels. Without future covariates the original layers are kept (old checkpoints still load).
        self.use_future = (self.future_channels>0) or (len(self.embs_fut or [])>0)
        fut_dim = 0
        if self.use_future:
            self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
            fut_dim = self.emb_fut.output_channels + self.future_channels

        self.prepare = nn.Linear(emb_past_out_channel+self.past_channels+fut_dim, d_model)

        self.model = nn.ModuleList([TimesBlock(self.past_steps,self.future_steps,top_k,d_model,d_ff,num_kernels) for _ in range(e_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

        self.predict_linear = nn.Linear(self.past_steps, self.future_steps + self.past_steps)
  
        self.projection = nn.Linear(d_model, self.out_channels*self.mul, bias=True)
  
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
        if x_mark_dec is not None:
            ## B F Cf -> B (T+F) Cf, zeros on the past steps
            pad = torch.zeros(x_mark_dec.shape[0], self.past_steps, x_mark_dec.shape[2], device=x_mark_dec.device, dtype=x_mark_dec.dtype)
            enc_out = torch.cat([enc_out, torch.cat([pad, x_mark_dec], 1)], 2)
        # TimesNet
        enc_out = self.prepare(enc_out)
        for i in range(self.e_layers):
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

        future = None
        if self.use_future:
            if 'x_cat_future' in batch.keys():
                future = [self.emb_fut(BS,batch['x_cat_future'].to(self.device))]
            else:
                future = [self.emb_fut(BS,None)]
            if 'x_num_future' in batch.keys():
                future.append(batch['x_num_future'].to(self.device))
            future = torch.cat(future, 2)

        dec_out = self.forecast(x_enc, emb_past, None, future)

        return dec_out[:, -self.future_steps:,:].reshape(BS,self.future_steps,self.out_channels,self.mul)
