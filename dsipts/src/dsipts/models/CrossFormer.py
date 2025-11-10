## Copyright 2023 Yunhao Zhang and Junchi Yan (https://github.com/Thinklab-SJTU/Crossformer?tab=Apache-2.0-1-ov-file#readme)
## Code modified for align the notation and the batch generation
## extended to all present in crossformer folder



from torch import  nn
import torch

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base
from typing import List,Union
from einops import  repeat
from ..data_structure.utils import beauty_string
from .utils import  get_scope
from .crossformer.cross_encoder import Encoder
from .crossformer.cross_decoder import Decoder
from .crossformer.cross_embed import DSW_embedding
from .utils import Embedding_cat_variables
from math import ceil
from .utils import Embedding_cat_variables
from .utils import get_activation

  

  
class CrossFormer(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True

    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 
                 d_model:int,
                 hidden_size:int,
                 n_head:int,
                 seg_len:int,
                 n_layer_encoder:int,
                 win_size:int,
                 factor:int=10,
                 dropout_rate:float=0.1,
                 activation:str='torch.nn.ReLU',

                 **kwargs)->None:
        """  CrossFormer (https://openreview.net/forum?id=vSVLM2j9eie)

        
        Args:
            d_model (int): The dimensionality of the model.
            hidden_size (int): The size of the hidden layers.
            n_head (int): The number of attention heads.
            seg_len (int): The length of the segments.
            n_layer_encoder (int): The number of layers in the encoder.
            win_size (int): The size of the window for attention.
            factor (int, optional): see .crossformer.attn.TwoStageAttentionLayer. Defaults to 10.
            dropout_rate (float, optional): The dropout rate. Defaults to 0.1.
            activation (str, optional): The activation function to use. Defaults to 'torch.nn.ReLU'.
            **kwargs: Additional keyword arguments for the parent class.
        
        Returns:
            None: This method does not return a value.
        
        Raises:
            ValueError: If the activation function is not recognized.
            

        """
      
        if isinstance(activation, str):
            activation = get_activation(activation)
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        

        # The padding operation to handle invisible sgemnet length
        self.pad_past_steps = ceil(1.0 *self.past_steps / seg_len) * seg_len
        self.pad_future_steps = ceil(1.0 * self.future_steps / seg_len) * seg_len
        self.past_steps_add = self.pad_past_steps - self.past_steps

        # Embedding
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.past_channels+emb_past_out_channel, (self.pad_past_steps // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        ## Custom embeddings ##these are not used in crossformer


        self.encoder = Encoder(n_layer_encoder, win_size, d_model, n_head, hidden_size, block_depth = 1, \
                                    dropout = dropout_rate,in_seg_num = (self.pad_past_steps // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.past_channels+emb_past_out_channel, (self.pad_future_steps // seg_len), d_model))
        self.decoder = Decoder(seg_len, n_layer_encoder + 1, d_model, n_head, hidden_size, dropout_rate, \
                                    out_seg_num = (self.pad_future_steps // seg_len), factor = factor)
     
        dim = self.past_channels+emb_past_out_channel+emb_fut_out_channel+self.future_channels
        self.final_layer = nn.Sequential(activation(),
                                         nn.Linear(dim, dim//2),
                                         activation(),
                                         nn.Linear(dim//2, self.mul*self.out_channels ))
        
                                         
        
    def can_be_compiled(self):
        return True
        
    def forward(self, batch):

        x_seq = batch['x_num_past'].to(self.device)#[:,:,idx_target]
        BS = x_seq.shape[0]
        
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

        x_seq = torch.cat([x_seq,emb_past],2) 
        batch_size = x_seq.shape[0]
        if (self.past_steps_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.past_steps_add, -1), x_seq), dim = 1)
        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)        
        enc_out = self.encoder(x_seq)
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        res = predict_y[:, :self.future_steps,:]
        tmp_future.append(res)
        res = self.final_layer(torch.cat(tmp_future,2))
        return res.reshape(BS, -1, self.out_channels,self.mul)