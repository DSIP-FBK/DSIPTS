
from torch import  nn
import torch
from .base import Base
from typing import List,Union
from ..data_structure.utils import beauty_string
from .utils import  get_activation
from .patchtst.layers import series_decomp, PatchTST_backbone


  
class PatchTST(Base):

    
    def __init__(self, 
                 past_steps:int,
                 future_steps:int,
                 patch_len: int,
                 past_channels:int,
                 future_channels:int,
                 out_channels:int,
                 d_model:int,
                 embs:List[int],
                 kernel_size:int,
                 decomposition:bool=True,
                 activation:str='torch.nn.ReLU',
                 n_head:int=1,
                 n_layer:int=2,
                 stride:int=8,
                 remove_last:bool = False,
                 hidden_size:int=1048,
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[int]=[],
                 dropout_rate:float=0.1,
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None,
                 **kwargs)->None:
        """

        Args:
            past_steps (int): number of past datapoints used , not used here
            future_steps (int): number of future lag to predict
            patch_len (int): patch_len
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            out_channels (int):  number of output channels
            d_model (int):  dimension of the attention model
            embs (List): list of the initial dimension of the categorical variables
            embed_type (int): type of embedding
            kernel_size (int): kernel_size
            activation (str, optional): activation fuction function pytorch. Default torch.nn.ReLU
            n_head (int, optional): number of heads
            n_layer (int, optional): number of encoding layers
            
            remove_last (boolean,optional): if true the model try to predic the difference respect the last observation.
            
            out_channels (int):  number of output channels
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            quantiles (List[int], optional): NOT USED YET
            dropout_rate (float, optional):  dropout rate in Dropout layers. Defaults to 0.1.
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
        """
        super().__init__(**kwargs)

        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
        if isinstance(activation, str):
            activation = get_activation(activation)
        else:
            beauty_string('There is a bug in pytorch lightening, the constructior is called twice ','info',self.verbose)
        
   
        self.save_hyperparameters(logger=False)
        self.use_quantiles = False
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.loss_type = loss_type
        self.persistence_weight = persistence_weight 
        self.remove_last = remove_last
        self.future_steps = future_steps  ##this is mandatory
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

    

        self.embs = nn.ModuleList()
        emb_channels = 0
        for k in embs:
            self.embs.append(nn.Embedding(k+1,d_model))
            emb_channels = d_model
            
        past_channels+=emb_channels
        
        

        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=past_channels, context_window = past_steps, target_window=future_steps, patch_len=patch_len, stride=stride, 
                                  max_seq_len=past_steps+future_steps, n_layers=n_layer, d_model=d_model,
                                  n_heads=n_head, d_k=None, d_v=None, d_ff=hidden_size, norm=False, attn_dropout=dropout_rate,
                                  dropout=dropout_rate, act=activation(), key_padding_mask='auto', padding_var=None, 
                                  attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                                  pe='zeros', learn_pe=True, fc_dropout=dropout_rate, head_dropout=dropout_rate, padding_patch = 'end',
                                  pretrain_head=False, head_type='flatten', individual=False, revin=True, affine=False,
                                  subtract_last=remove_last, verbose=False)
            self.model_res = PatchTST_backbone(c_in=past_channels, context_window = past_steps, target_window=future_steps, patch_len=patch_len, stride=stride, 
                                  max_seq_len=past_steps+future_steps, n_layers=n_layer, d_model=d_model,
                                  n_heads=n_head, d_k=None, d_v=None, d_ff=hidden_size, norm=False, attn_dropout=dropout_rate,
                                  dropout=dropout_rate, act=activation(), key_padding_mask='auto', padding_var=None, 
                                  attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                                  pe='zeros', learn_pe=True, fc_dropout=dropout_rate, head_dropout=dropout_rate, padding_patch = 'end',
                                  pretrain_head=False, head_type='flatten', individual=False, revin=True, affine=False,
                                  subtract_last=remove_last, verbose=False)
        else:
            self.model = PatchTST_backbone(c_in=past_channels, context_window = past_steps, target_window=future_steps, patch_len=patch_len, stride=stride, 
                                  max_seq_len=past_steps+future_steps, n_layers=n_layer, d_model=d_model,
                                  n_heads=n_head, d_k=None, d_v=None, d_ff=hidden_size, norm=False, attn_dropout=dropout_rate,
                                  dropout=dropout_rate, act=activation(), key_padding_mask='auto', padding_var=None, 
                                  attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                                  pe='zeros', learn_pe=True, fc_dropout=dropout_rate, head_dropout=dropout_rate, padding_patch = 'end',
                                  pretrain_head=False, head_type='flatten', individual=False, revin=True, affine=False,
                                  subtract_last=remove_last, verbose=False)
    
        self.final_linear = nn.Sequential(nn.Linear(past_channels,past_channels//2),activation(),nn.Dropout(dropout_rate), nn.Linear(past_channels//2,out_channels)  )
    
    def forward(self, batch):           # x: [Batch, Input length, Channel]
        
        x_seq = batch['x_num_past'].to(self.device)#[:,:,idx_target]
        if 'x_cat_past' in batch.keys():
            cat_past = batch['x_cat_past'].to(self.device)
        tot = [x_seq]
        if 'x_cat_past' in batch.keys():
            tmp_emb = None
            for i in range(len(self.embs)):
                if i>0:
                    tmp_emb+=self.embs[i](cat_past[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_past[:,:,i])
            tot.append(tmp_emb)

        x_seq = torch.cat(tot,axis=2)
            
            
            
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x_seq)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
            
        x = self.final_linear(x)       
    
        return x.unsqueeze(3)
        
        
        
        