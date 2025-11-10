## Copyright https://github.com/yuqinie98/PatchTST/blob/main/LICENSE
## Modified for notation alignmenet and batch structure
## extended to what inside patchtst folder



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
from .utils import  get_scope
from .utils import  get_activation
from .patchtst.layers import series_decomp, PatchTST_backbone
from .utils import Embedding_cat_variables



  
class PatchTST(Base):
    handle_multivariate = True
    handle_future_covariates = False
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    
    def __init__(self, 
              
                 d_model:int,
                 patch_len:int,
                 kernel_size:int,
                 decomposition:bool=True,
                 activation:str='torch.nn.ReLU',
                 n_head:int=1,
                 n_layer:int=2,
                 stride:int=8,
                 remove_last:bool = False,
                 hidden_size:int=1048,
                 dropout_rate:float=0.1,
                 **kwargs)->None:
        """Initializes the model with specified parameters.https://github.com/yuqinie98/PatchTST/blob/main/
        
        Args:
            d_model (int): The dimensionality of the model.
            patch_len (int): The length of the patches.
            kernel_size (int): The size of the kernel for convolutional layers.
            decomposition (bool, optional): Whether to use decomposition. Defaults to True.
            activation (str, optional): The activation function to use. Defaults to 'torch.nn.ReLU'.
            n_head (int, optional): The number of attention heads. Defaults to 1.
            n_layer (int, optional): The number of layers in the model. Defaults to 2.
            stride (int, optional): The stride for convolutional layers. Defaults to 8.
            remove_last (bool, optional): Whether to remove the last layer. Defaults to False.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 1048.
            dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.1.
            **kwargs: Additional keyword arguments.
        
        Raises:
            ValueError: If the activation function is not recognized.
        

        """
        super().__init__(**kwargs)

        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
        if isinstance(activation, str):
            activation = get_activation(activation)
        else:
            beauty_string('There is a bug in pytorch lightening, the constructior is called twice ','info',self.verbose)
        
   
        self.save_hyperparameters(logger=False)
      
        self.remove_last = remove_last
     
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels

     
    
        self.past_channels+=emb_past_out_channel
        
    
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=self.past_channels, context_window = self.past_steps, target_window=self.future_steps, patch_len=patch_len, stride=stride, 
                                  max_seq_len=self.past_steps+self.future_steps, n_layers=n_layer, d_model=d_model,
                                  n_heads=n_head, d_k=None, d_v=None, d_ff=hidden_size, norm='BatchNorm', attn_dropout=dropout_rate,
                                  dropout=dropout_rate, act=activation(), key_padding_mask='auto', padding_var=None, 
                                  attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                                  pe='zeros', learn_pe=True, fc_dropout=dropout_rate, head_dropout=dropout_rate, padding_patch = 'end',
                                  pretrain_head=False, head_type='flatten', individual=False, revin=True, affine=False,
                                  subtract_last=remove_last, verbose=False)
            self.model_res = PatchTST_backbone(c_in=self.past_channels, context_window = self.past_steps, target_window=self.future_steps, patch_len=patch_len, stride=stride, 
                                  max_seq_len=self.past_steps+self.future_steps, n_layers=n_layer, d_model=d_model,
                                  n_heads=n_head, d_k=None, d_v=None, d_ff=hidden_size, norm='BatchNorm', attn_dropout=dropout_rate,
                                  dropout=dropout_rate, act=activation(), key_padding_mask='auto', padding_var=None, 
                                  attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                                  pe='zeros', learn_pe=True, fc_dropout=dropout_rate, head_dropout=dropout_rate, padding_patch = 'end',
                                  pretrain_head=False, head_type='flatten', individual=False, revin=True, affine=False,
                                  subtract_last=remove_last, verbose=False)
        else:
            self.model = PatchTST_backbone(c_in=self.past_channels, context_window = self.past_steps, target_window=self.future_steps, patch_len=patch_len, stride=stride, 
                                  max_seq_len=self.past_steps+self.future_steps, n_layers=n_layer, d_model=d_model,
                                  n_heads=n_head, d_k=None, d_v=None, d_ff=hidden_size, norm='BatchNorm', attn_dropout=dropout_rate,
                                  dropout=dropout_rate, act=activation(), key_padding_mask='auto', padding_var=None, 
                                  attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                                  pe='zeros', learn_pe=True, fc_dropout=dropout_rate, head_dropout=dropout_rate, padding_patch = 'end',
                                  pretrain_head=False, head_type='flatten', individual=False, revin=True, affine=False,
                                  subtract_last=remove_last, verbose=False)
    
    
        dim = self.past_channels+emb_fut_out_channel+self.future_channels
        self.final_layer = nn.Sequential(activation(),
                                         nn.Linear(dim, dim*2),
                                         activation(),
                                         nn.Linear(dim*2,self.out_channels*self.mul  ))


    
        #self.final_linear = nn.Sequential(nn.Linear(past_channels,past_channels//2),activation(),nn.Dropout(dropout_rate), nn.Linear(past_channels//2,out_channels)  )
    
    def can_be_compiled(self):
        return True  
    
    def forward(self, batch):           # x: [Batch, Input length, Channel]
        

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
        
        
        tot = [x_seq,emb_past]
    
        x_seq = torch.cat(tot,axis=2)

        if self.decomposition:
            res_init, trend_init = self.decomp_module(x_seq)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x_seq.permute(0,2,1)# x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        
        
        tmp_future.append(x)
        tmp_future = torch.cat(tmp_future,2)
        output = self.final_layer(tmp_future)
        return output.reshape(BS,self.future_steps,self.out_channels,self.mul)

        
        
        
        