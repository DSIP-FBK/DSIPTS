
from torch import  nn
import torch
from .base import Base
from typing import List,Union
from einops import  repeat

from .crossformer.cross_encoder import Encoder
from .crossformer.cross_decoder import Decoder
from .crossformer.cross_embed import DSW_embedding

from math import ceil
  
  
#    self, past_channels, past_steps, future_steps, seg_len, win_size = 4,
#                factor=10, d_model=512, hidden_size = 1024, n_head=8, n_layer_encoder=3, 
#                dropout=0.0, baseline = False,
  
class CrossFormer(Base):

    
    def __init__(self, 
                 past_steps:int,
                 future_steps:int,
                 past_channels:int,
                 future_channels:int,
                 d_model:int,
                 embs:List[int],
                 hidden_size:int,
                 n_head:int,
                 seg_len:int,
                 n_layer_encoder:int,
                 win_size:int,
                 out_channels:int,
                 factor:int=5,
                 remove_last = False,
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[int]=[],
                 dropout_rate:float=0.1,
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None)->None:
        """CroosFormer (https://openreview.net/forum?id=vSVLM2j9eie)

        Args:
            past_steps (int): number of past datapoints used , not used here
            future_steps (int): number of future lag to predict
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            d_model (int):  dimension of the attention model
            embs (List): list of the initial dimension of the categorical variables
            hidden_size (int): hidden size of the linear block
            n_head (int): number of heads
            seg_len (int): segment length (L_seg) see the paper for more details
            n_layer_encoder (int):  layers to use in the encoder
            win_size (int): window size for segment merg
            factor (int): num of routers in Cross-Dimension Stage of TSA (c) see the paper
            remove_last (boolean,optional): if true the model try to predic the difference respect the last observation.
            out_channels (int):  number of output channels
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            quantiles (List[int], optional): NOT USED YET
            dropout_rate (float, optional):  dropout rate in Dropout layers. Defaults to 0.1.
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
        """
      
   
        super(CrossFormer, self).__init__()
        self.save_hyperparameters(logger=False)
        self.use_quantiles = False
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.loss_type = loss_type
        self.persistence_weight = persistence_weight 
        self.remove_last = remove_last

        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

        self.future_steps = future_steps
        

        # The padding operation to handle invisible sgemnet length
        self.pad_past_steps = ceil(1.0 *past_steps / seg_len) * seg_len
        self.pad_future_steps = ceil(1.0 * future_steps / seg_len) * seg_len
        self.past_steps_add = self.pad_past_steps - past_steps

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, past_channels, (self.pad_past_steps // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(n_layer_encoder, win_size, d_model, n_head, hidden_size, block_depth = 1, \
                                    dropout = dropout_rate,in_seg_num = (self.pad_past_steps // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, past_channels, (self.pad_future_steps // seg_len), d_model))
        self.decoder = Decoder(seg_len, n_layer_encoder + 1, d_model, n_head, hidden_size, dropout_rate, \
                                    out_seg_num = (self.pad_future_steps // seg_len), factor = factor)
        
    def forward(self, batch):

        idx_target = batch['idx_target'][0]
        x_seq = batch['x_num_past'].to(self.device)#[:,:,idx_target]
        
        if self.remove_last:
            x_start = x_seq[:,-1,:].unsqueeze(1)
            x_seq[:,:,:]-=x_start   
 
        batch_size = x_seq.shape[0]
        if (self.past_steps_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.past_steps_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        res = predict_y[:, :self.future_steps,:].unsqueeze(3)
        if self.remove_last:
            res+=x_start.unsqueeze(1)

        return res[:, :,idx_target,:]