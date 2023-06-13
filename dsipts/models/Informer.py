
from torch import  nn
import torch
from .base import Base
from .utils import QuantileLossMO,Permute, get_device,L1Loss, get_activation
from typing import List,Union
import numpy as np
import logging
from .informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from .informer.decoder import Decoder, DecoderLayer
from .informer.attn import FullAttention, ProbAttention, AttentionLayer
from .informer.embed import DataEmbedding
  
  
    
  
class Informer(Base):

    
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
                 activation:str='relu',
                 attn: str='prob',
                 output_attention:bool=False,
                 distil:bool=True,
                 factor:int=5,
                 num_heads:int=1,
                 quantiles:List[int]=[],
                 loss_type: str='standard',
                 dropout_rate:float=0.1,
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None)->None:
        """Informer

        Args:
            past_steps (int): number of past datapoints used , not used here
            future_steps (int): number of future lag to predict
            d_model (int): _description_
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
            attn (str, optional): attention used in encoder, options:[prob, full]. Defaults to 'prob'.
            output_attention (bool, optional): visualize attention, keep it False please . Defaults to False. TODO: FIX THIS
            distil (bool, optional): whether to use distilling in encoder, using this argument means not using distilling. Defaults to True.
            factor (int, optional): probsparse attn factor. Defaults to 5.
            num_heads (int, optional):  heads equal in the encoder and encoder. Defaults to 1.
            quantiles (List[int], optional): NOT USED YET
            loss_type (str, optional): this model uses custom losses
            dropout_rate (float, optional):  dropout rate in Dropout layers. Defaults to 0.1.
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
        """
   
        super(Informer, self).__init__()
        self.save_hyperparameters(logger=False)

        self.future_steps = future_steps
        self.use_quantiles = False
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.output_attention = output_attention
        
        self.enc_embedding = DataEmbedding(past_channels, d_model, embs, dropout_rate)
        self.dec_embedding = DataEmbedding(future_channels, d_model, embs, dropout_rate)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout_rate, output_attention=output_attention), 
                                d_model, num_heads, mix=False),
                    d_model,
                    hidden_size,
                    dropout=dropout_rate,
                    activation=activation
                ) for l in range(n_layer_encoder)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(n_layer_encoder-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout_rate, output_attention=False), 
                                d_model, num_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout_rate, output_attention=False), 
                                d_model, num_heads, mix=False),
                    d_model,
                    hidden_size,
                    dropout=dropout_rate,
                    activation=activation,
                )
                for l in range(n_layer_decoder)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, out_channels, bias=True)
        
                
        self.loss_type = loss_type
        
        
        self.loss = L1Loss()
        
    def forward(self,batch): 
        #x_enc, x_mark_enc, x_dec, x_mark_dec,enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        x_enc = batch['x_num_past']
        x_mark_enc = batch['x_cat_past']
        enc_self_mask = None
        
        x_dec = batch['x_num_future']
        #idx_target = batch['idx_target']
        ##BS x L x channels
        #import pdb
       # pdb.set_trace()
        x_dec[:,-self.future_steps:,:] = 0 #padding in teoria quelle future sono tutte 0, TODO:  add idx_target future
        x_mark_dec = batch['x_cat_future']
        dec_self_mask = None
        dec_enc_mask = None
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        
        #import pdb
        #pdb.set_trace()
        if self.output_attention:
            return dec_out[:,-self.future_steps:,:], attns
        else:
            return dec_out[:,-self.future_steps:,:].unsqueeze(3) # [B, L, C,1]
       
       
       
       
       
       

        
    def compute_loss(self,batch):
        """
        custom loss calculation
        
        :meta private:
        """
        y_hat = self(batch)
        
        mse_loss = self.loss(y_hat, batch['y'])
        x =  batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        x_start = x[:,-1,idx_target].unsqueeze(1)
        y_persistence = x_start.repeat(1,self.future_steps,1)
        
        #import pdb
        #pdb.set_trace()
        if self.loss_type == 'linear_penalization':
            persistence_loss = -nn.L1Loss()(y_persistence,y_hat[:,:,:,0])
            loss = self.persistence_weight*persistence_loss + (1-self.persistence_weight)*mse_loss
        elif self.loss_type == 'inverse_penalization':
            idx = 1 if self.use_quantiles else 0
            weights = self.persistence_weight/(torch.abs(y_hat[:,:,:,0]-y_persistence) +0.1)

            loss = torch.mean(torch.abs(y_hat[:,:,:,0]- batch['y'])*weights)
        else:
            loss = mse_loss
        
        return loss
    def training_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
        return self.compute_loss(batch)
    
    def validation_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
        return  self.compute_loss(batch)  
    
    