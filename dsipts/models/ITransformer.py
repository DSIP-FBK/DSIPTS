import torch
import torch.nn as nn
import numpy as np
from .itransformer.Transformer_EncDec import Encoder, EncoderLayer
from .itransformer.SelfAttention_Family import FullAttention, AttentionLayer
from .itransformer.Embed import DataEmbedding_inverted
from .base import  Base
from .utils import QuantileLossMO,Permute, get_activation

from typing import List, Union
from ..data_structure.utils import beauty_string
from .utils import  get_scope

class ITransformer(Base):
    handle_multivariate = True
    handle_future_covariates = False # or at least it seems...
    handle_categorical_variables = True #solo nel encoder
    handle_quantile_loss = False #EASY to add
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
                 d_model: int,
                 n_head: int,
                 n_layer_decoder: int,
                 use_norm: bool,
                 dropout_rate: float,
                 activation: str='',
                 class_strategy: str = 'projection', #projection/average/cls_token
                 
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[float]=[],
                 optim:Union[str,None]=None,
                 optim_config:Union[dict,None]=None,
                 scheduler_config:Union[dict,None]=None,
                 **kwargs)->None:
        """ITRANSFORMER: INVERTED TRANSFORMERS ARE EFFECTIVE FOR TIME SERIES FORECASTING
        https://arxiv.org/pdf/2310.06625

   
        Args:
            out_channels (int): number of variables to be predicted
            past_steps (int): Lookback window length
            future_steps (int): Horizon window length
            past_channels (int): number of past variables
            future_channels (int): number of future auxiliary variables 
            embs (List[int]): 
            hidden_size (int): first embedding size of the model ('r' in the paper)
            d_model (int): second embedding size (r^{tilda} in the model). Should be smaller than hidden_size
            n_add_enc (int): number of OTHERS heads for the encoder part in the NN. 1 is always used by default.
            n_add_dec (int): number of OTHERS heads for the decoder part in the NN. 1 is always used by default.
            dropout_rate (float): 
            activation (str, optional): activation function to be used in the Residual Block. E.g., 'nn.GELU'. Defaults to ''.
            persistence_weight (float, optional): Defaults to 0.0.
            loss_type (str, optional): Defaults to 'l1'.
            quantiles (List[float], optional): Defaults to [].
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


        ##my update
        self.embs = nn.ModuleList()

        for k in embs:
            self.embs.append(nn.Embedding(k+1,d_model))
         


        self.out_channels = out_channels
        self.seq_len = past_steps
        self.pred_len = future_steps
        self.output_attention = False## not need output attention
        self.use_norm = use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, d_model, embed_type='what?', freq='what?', dropout=dropout_rate)  ##embed, freq not used inside
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
        self.projector = nn.Linear(d_model, future_steps, bias=True)

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
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))


        return dec_out

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
        #    x_dec =  batch['x_num_future'].to(self.device)
        ##not used known variables
        if 'x_cat_future' in batch.keys():
            x_mark_dec =  batch['x_cat_future'].to(self.device)
        else:
            x_mark_dec = None



        ##row 124 Transformer/experiments/exp_long_term_forecasting.py ma in realta' NON USATO!
        x_dec = torch.zeros(x_enc.shape[0],self.pred_len,self.out_channels).float()
        x_dec = torch.cat([batch['y'], x_dec], dim=1).float().to(self.device)

        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        idx_target = batch['idx_target'][0]
        return dec_out[:, :,idx_target].unsqueeze(3)
        
        #return dec_out[:, -self.pred_len:, :]  # [B, L, D]
