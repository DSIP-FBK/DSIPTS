## Copyright https://github.com/thuml/iTransformer?tab=MIT-1-ov-file#readme
## Modified for notation alignmenet and batch structure
## extended to what inside itransformer folder

import torch
import torch.nn as nn
import numpy as np
from .itransformer.Transformer_EncDec import Encoder, EncoderLayer
from .itransformer.SelfAttention_Family import FullAttention, AttentionLayer
from .itransformer.Embed import DataEmbedding_inverted
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



class ITransformer(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 # specific params
                 hidden_size:int,
                 d_model: int,
                 n_head: int,
                 n_layer_decoder: int,
                 use_norm: bool,
                 class_strategy: str = 'projection', #projection/average/cls_token    
                 dropout_rate: float=0.1,
                 activation: str='',
                 **kwargs)->None:
        """Initialize the ITransformer model for time series forecasting.
        
        This class implements the Inverted Transformer architecture as described in the paper 
        "ITRANSFORMER: INVERTED TRANSFORMERS ARE EFFECTIVE FOR TIME SERIES FORECASTING" 
        (https://arxiv.org/pdf/2310.06625).
        
        Args:
            hidden_size (int): The first embedding size of the model ('r' in the paper).
            d_model (int): The second embedding size (r^{tilda} in the model). Should be smaller than hidden_size.
            n_head (int): The number of attention heads.
            n_layer_decoder (int): The number of layers in the decoder.
            use_norm (bool): Flag to indicate whether to use normalization.
            class_strategy (str, optional): The strategy for classification, can be 'projection', 'average', or 'cls_token'. Defaults to 'projection'.
            dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.1.
            activation (str, optional): The activation function to be used. Defaults to ''.
            **kwargs: Additional keyword arguments.
        
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


  
        self.output_attention = False## not need output attention
        self.use_norm = use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.past_steps, d_model, embed_type='what?', freq='what?', dropout=dropout_rate)  ##embed, freq not used inside
        ## future covariates (numerical + categorical embedding) become extra variate tokens: each future channel
        ## is embedded along the future time axis (future_steps -> d_model). They do not need to exist in the past
        ## and can have any number of channels; their tokens are dropped after the encoder like the past covariates.
        ## Without future covariates no extra module is created (old checkpoints still load).
        self.use_future = (self.future_channels>0) or (len(self.embs_fut or [])>0)
        if self.use_future:
            self.fut_embedding = DataEmbedding_inverted(self.future_steps, d_model, embed_type='what?', freq='what?', dropout=dropout_rate)
        self.class_strategy = class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=0.1, attention_dropout=dropout_rate, ##factor is not used in the Full attention
                                      output_attention=self.output_attention), d_model, n_head), ## not need output attention
                    d_model,
                    hidden_size,
                    dropout = dropout_rate,
                    activation = activation()
                ) for l in range(n_layer_decoder)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, self.future_steps*self.mul, bias=True)

    def can_be_compiled(self):
        return True  

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        if x_mark_dec is not None:
            ## B F Cf -> B Cf E, appended after the past tokens (the first N tokens stay the past variates)
            enc_out = torch.cat([enc_out, self.fut_embedding(x_mark_dec, None)], 1)
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N (S*mul) -> B S N mul
        ## NB: the projector output is (S, mul) per token: reshaping (B, S*mul, N) straight to (B, S, N, mul)
        ## would mix the quantiles of one variate with the outputs of another when mul>1 (identical for mul=1)
        B = enc_out.shape[0]
        dec_out = self.projector(enc_out)[:, :N, :] # filter the covariates
        dec_out = dec_out.reshape(B, N, self.future_steps, self.mul).permute(0, 2, 1, 3)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * stdev[:, 0, :].reshape(B, 1, N, 1)
            dec_out = dec_out + means[:, 0, :].reshape(B, 1, N, 1)

        return dec_out

    def forward(self, batch:dict)-> float:

        x_enc = batch['x_num_past'].to(self.device)
        BS = x_enc.shape[0]
        future = None
        if self.use_future:
            if 'x_cat_future' in batch.keys():
                future = [self.emb_fut(BS,batch['x_cat_future'].to(self.device))]
            else:
                future = [self.emb_fut(BS,None)]
            if 'x_num_future' in batch.keys():
                future.append(batch['x_num_future'].to(self.device))
            future = torch.cat(future, 2)
        if 'x_cat_past' in batch.keys():
            emb_past = self.emb_past(BS,batch['x_cat_past'].to(self.device))
        else:
            emb_past = self.emb_past(BS,None)
 



        dec_out = self.forecast(x_enc, emb_past, None, future)  ## B S N mul
        if ('idx_target' not in batch.keys()) or (len(batch['idx_target'][0])==0):
            raise ValueError('ITransformer predicts the targets from their own past tokens: the targets must be among the past variables')
        idx_target = batch['idx_target'][0]
        return dec_out[:, :, idx_target, :].reshape(BS,self.future_steps,self.out_channels,self.mul)
        
        #return dec_out[:, -self.pred_len:, :]  # [B, L, D]
