
## Copyright 2020    Informer (hhttps://github.com/zhouhaoyi/Informer2020/tree/main/models)
## Code modified for align the notation and the batch generation
## extended to all present in informer, autoformer folder
from torch import  nn
import torch
from .base import Base
from typing import List,Union

from .informer.encoder import Encoder, EncoderLayer, ConvLayer
from .informer.decoder import Decoder, DecoderLayer
from .informer.attn import FullAttention, ProbAttention, AttentionLayer
from .informer.embed import DataEmbedding
from ..data_structure.utils import beauty_string
from .utils import  get_scope,QuantileLossMO
  
    
  
class Informer(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    
    def __init__(self, 
                 past_steps:int,
                 future_steps:int,
                 past_channels:int,
                 future_channels:int,
                 d_model:int,
                 embs:List[int],
                 hidden_size:int,
                 n_layer_encoder:int,
                 n_layer_decoder:int,
                 out_channels:int,
                 mix:bool=True,
                 activation:str='torch.nn.ReLU',
                 remove_last = False,
                 attn: str='prob',
                 distil:bool=True,
                 factor:int=5,
                 n_head:int=1,
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[int]=[],
                 dropout_rate:float=0.1,
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None,
                 **kwargs)->None:
        """Informer

        Args:
            past_steps (int): number of past datapoints used , not used here
            future_steps (int): number of future lag to predict
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            d_model (int):  dimension of the attention model
            embs (List): list of the initial dimension of the categorical variables
            hidden_size (int): hidden size of the linear block
            n_layer_encoder (int):  layers to use in the encoder
            n_layer_decoder (int):  layers to use in the decoder
            out_channels (int):  number of output channels
            mix (bool, optional): se mix attention in generative decoder. Defaults to True.
            activation (str, optional): relu or gelu. Defaults to 'relu'.
            remove_last (boolean,optional): if true the model try to predic the difference respect the last observation.
            attn (str, optional): attention used in encoder, options:[prob, full]. Defaults to 'prob'.
            distil (bool, optional): whether to use distilling in encoder, using this argument means not using distilling. Defaults to True.
            factor (int, optional): probsparse attn factor. Defaults to 5.
            n_head (int, optional):  heads equal in the encoder and encoder. Defaults to 1.
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            quantiles (List[int], optional): NOT USED YET
            dropout_rate (float, optional):  dropout rate in Dropout layers. Defaults to 0.1.
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
        """
   
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        beauty_string("BE SURE TO SETUP split_params:  shift:  ${model_configs.future_steps} BECAUSE IT IS REQUIRED",'info',True)
        self.future_steps = future_steps
        self.use_quantiles = False
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.persistence_weight = persistence_weight 
        self.loss_type = loss_type
        self.remove_last = remove_last
        
        if len(quantiles)>0:
            assert len(quantiles)==3, beauty_string('ONLY 3 quantiles premitted','info',True)
            self.use_quantiles = True
            self.mul = len(quantiles)
            self.loss = QuantileLossMO(quantiles)
        else:
            self.use_quantiles = False
            self.mul = 1
            if self.loss_type == 'mse':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()
        
        self.enc_embedding = DataEmbedding(past_channels, d_model, embs, dropout_rate)
        self.dec_embedding = DataEmbedding(future_channels, d_model, embs, dropout_rate)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout_rate, output_attention=False), 
                                d_model, n_head, mix=False),
                    d_model,
                    hidden_size,
                    dropout=dropout_rate,
                    activation=activation
                ) for _ in range(n_layer_encoder)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(n_layer_encoder-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout_rate, output_attention=False), 
                                d_model, n_head, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout_rate, output_attention=False), 
                                d_model, n_head, mix=False),
                    d_model,
                    hidden_size,
                    dropout=dropout_rate,
                    activation=activation,
                )
                for _ in range(n_layer_decoder)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, out_channels*self.mul, bias=True)
        
                
        
  
        
    def forward(self,batch): 
        #x_enc, x_mark_enc, x_dec, x_mark_dec,enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        x_enc = batch['x_num_past'].to(self.device)
        idx_target_future = batch['idx_target_future'][0]

        if 'x_cat_past' in batch.keys():
            x_mark_enc = batch['x_cat_past'].to(self.device)
        else:
            x_mark_enc = None
        enc_self_mask = None
        
        x_dec = batch['x_num_future'].to(self.device)
        x_dec[:,-self.future_steps:,idx_target_future] = 0
        
        
        if 'x_cat_future' in batch.keys():
            x_mark_dec = batch['x_cat_future'].to(self.device)
        else:
            x_mark_dec = None
        dec_self_mask = None
        dec_enc_mask = None
        
        
        if self.remove_last:
            idx_target = batch['idx_target'][0]
            x_start = x_enc[:,-1,idx_target].unsqueeze(1)
            x_enc[:,:,idx_target]-=x_start   
        
    
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        
        #import pdb
        #pdb.set_trace()
        res = dec_out[:,-self.future_steps:,:].unsqueeze(3)
        if self.remove_last:
            res+=x_start.unsqueeze(1)
        BS = res.shape[0]
        return  res.reshape(BS,self.future_steps,-1,self.mul)
       
       
       
          