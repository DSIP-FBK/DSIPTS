
## Copyright 2020    Informer (hhttps://github.com/zhouhaoyi/Informer2020/tree/main/models)
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

from .informer.encoder import Encoder, EncoderLayer, ConvLayer
from .informer.decoder import Decoder, DecoderLayer
from .informer.attn import FullAttention, ProbAttention, AttentionLayer
from .informer.embed import DataEmbedding
from ..data_structure.utils import beauty_string
#from .utils import Embedding_cat_variables not used here, custom cat embedding
from .utils import  get_scope

    
  
class Informer(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    
    def __init__(self, 
                 d_model:int,
                 hidden_size:int,
                 n_layer_encoder:int,
                 n_layer_decoder:int,
                 mix:bool=True,
                 activation:str='torch.nn.ReLU',
                 remove_last = False,
                 attn: str='prob',
                 distil:bool=True,
                 factor:int=5,
                 n_head:int=1,
                 dropout_rate:float=0.1,
                
                 **kwargs)->None:
        """Initialize the model with specified parameters. hhttps://github.com/zhouhaoyi/Informer2020/tree/main/models
        
        Args:
            d_model (int): The dimensionality of the model.
            hidden_size (int): The size of the hidden layers.
            n_layer_encoder (int): The number of layers in the encoder.
            n_layer_decoder (int): The number of layers in the decoder.
            mix (bool, optional): Whether to use mixed attention. Defaults to True.
            activation (str, optional): The activation function to use. Defaults to 'torch.nn.ReLU'.
            remove_last (bool, optional): Whether to remove the last layer. Defaults to False.
            attn (str, optional): The type of attention mechanism to use. Defaults to 'prob'.
            distil (bool, optional): Whether to use distillation. Defaults to True.
            factor (int, optional): The factor for attention. Defaults to 5.
            n_head (int, optional): The number of attention heads. Defaults to 1.
            dropout_rate (float, optional): The dropout rate. Defaults to 0.1.
            **kwargs: Additional keyword arguments.
        
        Raises:
            ValueError: If any of the parameters are invalid.
        
        Notes:
            Ensure to set up split_params: shift: ${model_configs.future_steps} as it is required!!
        """
   
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        beauty_string("BE SURE TO SETUP split_params:  shift:  ${model_configs.future_steps} BECAUSE IT IS REQUIRED",'info',True)
       
        self.remove_last = remove_last
        
        
        self.enc_embedding = DataEmbedding(self.past_channels, d_model, self.embs_past, dropout_rate)
        self.dec_embedding = DataEmbedding(self.future_channels, d_model, self.embs_fut, dropout_rate)
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

        self.projection = nn.Linear(d_model, self.out_channels*self.mul, bias=True)
        
                
        
    def can_be_compiled(self):
        return True  
        
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
       
       
       
          