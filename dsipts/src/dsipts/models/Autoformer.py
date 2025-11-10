## Copyright 2022 DLinear Authors (https://github.com/cure-lab/LTSF-Linear/tree/main?tab=Apache-2.0-1-ov-file#readme)
## Code modified for align the notation and the batch generation
## extended to all present in informer, autoformer folder


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
from ..data_structure.utils import beauty_string
from .utils import  get_activation,get_scope,QuantileLossMO
from .autoformer.layers import AutoCorrelation, AutoCorrelationLayer, Encoder, Decoder,\
    EncoderLayer, DecoderLayer, my_Layernorm, series_decomp,PositionalEmbedding
from .utils import Embedding_cat_variables


  
class Autoformer(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss= True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 label_len: int, 
                 d_model:int,
                 dropout_rate:float,
                 kernel_size:int,
                 activation:str='torch.nn.ReLU',
                 factor: float=0.5,
                 n_head:int=1,
                 n_layer_encoder:int=2,
                 n_layer_decoder:int=2,
                 hidden_size:int=1048,
                 **kwargs
                )->None:
        """Autoformer from https://github.com/cure-lab/LTSF-Linear

        Args:
            label_len (int): see the original implementation, seems like a warmup dimension (the decoder part will produce also some past predictions that are filter out at the end)
            d_model (int): embedding dimension of the attention layer
            dropout_rate (float): dropout raye
            kernel_size (int): kernel size
            activation (str, optional): _description_. Defaults to 'torch.nn.ReLU'.
            factor (int, optional): parameter of `.autoformer.layers.AutoCorrelation` for find the top k. Defaults to 0.5.
            n_head (int, optional): number of heads. Defaults to 1.
            n_layer_encoder (int, optional): number of  encoder layers. Defaults to 2.
            n_layer_decoder (int, optional): number of decoder layers. Defaults to 2.
            hidden_size (int, optional): output dimension of the transformer layer. Defaults to 1048.
        """
        super().__init__(**kwargs)
        beauty_string(self.description,'info',True)

        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
        if isinstance(activation,str):
            activation = get_activation(activation)
        else:
            beauty_string('There is a bug in pytorch lightening, the constructior is called twice ','info',self.verbose)
        
   
        
        self.save_hyperparameters(logger=False)
      


        
        self.seq_len = self.past_steps
        self.label_len = label_len
        self.pred_len = self.future_steps

        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps+label_len,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels


        # Decomp
        self.decomp = series_decomp(kernel_size)



        self.linear_encoder = nn.Sequential(nn.Linear(self.past_channels+emb_past_out_channel,self.past_channels*2),
                                            activation(),
                                            nn.Dropout(dropout_rate),
                                            nn.Linear(self.past_channels*2,d_model*2),
                                            activation(),
                                            nn.Dropout(dropout_rate),
                                            nn.Linear(d_model*2,d_model))
        
        self.linear_decoder = nn.Sequential(nn.Linear(self.future_channels+emb_fut_out_channel,self.future_channels*2),
                                            activation(),
                                            nn.Dropout(dropout_rate),
                                            nn.Linear(self.future_channels*2,d_model*2),
                                            activation() ,nn.Dropout(dropout_rate),
                                            nn.Linear(d_model*2,d_model))
       
        #self.final_layer =  nn.Linear(self.past_channels,self.out_channels)
       
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout_rate,
                                        output_attention=False),
                        d_model, n_head),
                    d_model,
                    hidden_size,
                    moving_avg=kernel_size,
                    dropout=dropout_rate,
                    activation=activation
                ) for _ in range(n_layer_encoder)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout_rate,
                                        output_attention=False),
                        d_model, n_head),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout_rate,
                                        output_attention=False),
                        d_model, n_head),
                    d_model,
                    self.out_channels,
                    hidden_size,
                    moving_avg=kernel_size,
                    dropout=dropout_rate,
                    activation=activation,
                )
                for _ in range(n_layer_decoder)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, self.out_channels*self.mul, bias=True)
        )
        self.projection = nn.Linear(self.past_channels,self.out_channels*self.mul )
    def can_be_compiled(self):
        return True
    def forward(self, batch):
        


        idx_target_future = batch['idx_target_future'][0]
        x = batch['x_num_past'].to(self.device)
        BS = x.shape[0]
        if 'x_cat_future' in batch.keys():
            emb_fut = self.emb_fut(BS,batch['x_cat_future'].to(self.device))
        else:
            emb_fut = self.emb_fut(BS,None)
        if 'x_cat_past' in batch.keys():
            emb_past = self.emb_past(BS,batch['x_cat_past'].to(self.device))
        else:
            emb_past = self.emb_past(BS,None)
        
  
        if 'x_num_future' in batch.keys():
            x_future = batch['x_num_future'].to(self.device)
            x_future[:,-self.pred_len:,idx_target_future] = 0
        
        

        
        mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)

        zeros = torch.zeros([x_future.shape[0], self.pred_len, x.shape[2]], device=x.device)
        seasonal_init, trend_init = self.decomp(x)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.linear_encoder(torch.cat([x,emb_past],2))
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.linear_decoder(torch.cat([x_future,emb_fut],2))
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final

        trend_part = self.projection(trend_part)
        dec_out = trend_part + seasonal_part

    
        BS = dec_out.shape[0]
        
        return dec_out[:, -self.pred_len:, :].reshape(BS,self.pred_len,-1,self.mul)  # [B, L, D,MUL]
         
        
