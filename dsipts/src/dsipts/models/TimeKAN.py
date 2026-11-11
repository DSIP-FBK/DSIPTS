## Copyright https://github.com/huangst21/TimeKAN/blob/main/models/TimeKAN.py
## Modified for notation alignmenet and batch structure
## extended to what inside itransformer folder

import torch
import torch.nn as nn
import numpy as np
from .timekan.Layers import FrequencyDecomp,FrequencyMixing,series_decomp,Normalize
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



class TimeKAN(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 # specific params
                 down_sampling_window:int,
                 e_layers:int,
                 moving_avg:int,
                 down_sampling_layers: int,
                 d_model: int,
                 begin_order: int,
                 use_norm:bool,
                 **kwargs)->None:
             
        
        
        
        super().__init__(**kwargs)
       
        self.save_hyperparameters(logger=False)
        self.e_layers = e_layers
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        ## future covariates (numerical and/or categorical) are fused late, per future step, with the
        ## backbone output: they do not need to exist in the past and can have any number of channels.
        ## Without future covariates the original head is kept (old checkpoints still load).
        self.use_future = (self.future_channels>0) or (len(self.embs_fut or [])>0)
        if self.use_future:
            self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
            emb_fut_out_channel = self.emb_fut.output_channels
        
        
        
        self.res_blocks = nn.ModuleList([FrequencyDecomp( self.past_steps,down_sampling_window,down_sampling_layers) for _ in range(e_layers)])
        self.add_blocks = nn.ModuleList([FrequencyMixing(d_model,self.past_steps,begin_order,down_sampling_window,down_sampling_layers) for _ in range(e_layers)])

        self.preprocess = series_decomp(moving_avg)
        self.enc_in = self.past_channels + emb_past_out_channel
        self.project = nn.Linear(self.enc_in,d_model)
        self.layer = e_layers
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.enc_in, affine=True, non_norm=use_norm)
                for i in range(down_sampling_layers + 1)
            ]
        )
        self.predict_layer =nn. Linear(
                        self.past_steps,
                        self.future_steps,
                    )
        if self.use_future:
            dim = d_model + emb_fut_out_channel + self.future_channels
            self.final_layer = nn.Sequential(nn.Linear(dim, dim),
                                             nn.GELU(),
                                             nn.Linear(dim, self.mul*self.out_channels))
        else:
            self.final_layer = nn.Linear(d_model, self.mul*self.out_channels)
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
    def can_be_compiled(self):
        return True#True  


    def __multi_level_process_inputs(self, x_enc):
        down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        x_enc = x_enc_sampling_list
        return x_enc


    def forward(self, batch:dict)-> float:

        x_enc = batch['x_num_past'].to(self.device)
        BS = x_enc.shape[0]
        if 'x_cat_past' in batch.keys():
            emb_past = self.emb_past(BS,batch['x_cat_past'].to(self.device))
        else:
            emb_past = self.emb_past(BS,None)

        x_past = torch.cat([x_enc,emb_past],2)

        x_enc = self.__multi_level_process_inputs(x_past)

        x_list = []
        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(self.project(x.reshape(B, T, N)))



        for i in range(self.layer):
            x_list = self.res_blocks[i](x_list)
            x_list = self.add_blocks[i](x_list)

        dec_out = x_list[0]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute( 0, 2, 1)  ## B, future_steps, d_model

        if self.use_future:
            tmp_future = [dec_out]
            if 'x_cat_future' in batch.keys():
                tmp_future.append(self.emb_fut(BS,batch['x_cat_future'].to(self.device)))
            else:
                tmp_future.append(self.emb_fut(BS,None))
            if 'x_num_future' in batch.keys():
                tmp_future.append(batch['x_num_future'].to(self.device))
            dec_out = torch.cat(tmp_future,2)

        dec_out = self.final_layer(dec_out)

        return dec_out.reshape(BS,self.future_steps,self.out_channels,self.mul)
