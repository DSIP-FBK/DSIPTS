## Copyright https://github.com/luodhhh/ModernTCN?tab=MIT-1-ov-file
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
    if pl.__version__>='2.0.0':
        from .base_v2 import Base
        OLD_PL = False
    else:
        from .base import Base
        OLD_PL = True
from ..data_structure.utils import beauty_string
from .utils import  get_scope
from .utils import  get_activation
from .modern_tcn.layers import ModernTCNModel, series_decomp
from .utils import Embedding_cat_variables



  
class ModernTCN(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    
    def __init__(self, 
              
                patch_size:int=16,
                patch_stride:int=8,
                stem_ratio: int=6, 
                downsample_ratio:int=2,
                ffn_ratio:int=2,
                num_blocks: list=[1,1,1,1], 
                large_size: list = [31,29,27,13], 
                small_size: list =[5,5,5,5],
                dims:list =[256,256,256,256], 
                dw_dims:list =[256,256,256,256],
                small_kernel_merged: bool=False, 
                drop_backbone:float=0.05, 
                drop_head:float=0.0, 
                use_multi_scale:bool=True,
                revin:bool=True,
                affine:bool=False,
                subtract_last:bool=False,
                individual:bool=True,             
                decomposition: bool= True,
                activation:str='torch.nn.ReLU',
                 kernel_size:int=25,
                 **kwargs)->None:
        """Initializes the model with specified parameters.https://github.com/luodhhh/ModernTCN/blob/main/ModernTCN-Long-term-forecasting/models/ModernTCN.py
        
        
        """
        super().__init__(**kwargs)

        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
        if isinstance(activation, str):
            activation = get_activation(activation)
        else:
            beauty_string('There is a bug in pytorch lightening, the constructior is called twice ','info',self.verbose)
        
   
        self.save_hyperparameters(logger=False)
      
      
      
      
     
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels
    
        self.past_channels+=emb_past_out_channel
        
        dim = self.past_channels+emb_fut_out_channel+self.future_channels
        self.final_layer = nn.Sequential(activation(),
                                         nn.Linear(dim, dim*2),
                                         activation(),
                                         nn.Linear(dim*2,self.out_channels*self.mul  ))
    
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_res = ModernTCNModel(patch_size=patch_size,
                                            patch_stride=patch_stride,
                                            stem_ratio=stem_ratio, 
                                            downsample_ratio=downsample_ratio,
                                            ffn_ratio=ffn_ratio,
                                            num_blocks=num_blocks, 
                                            large_size=large_size, 
                                            small_size=small_size,
                                            dims=dims, 
                                            dw_dims=dw_dims,
                                            nvars=self.past_channels, 
                                            small_kernel_merged=small_kernel_merged, 
                                            backbone_dropout=drop_backbone, 
                                            head_dropout=drop_head, 
                                            use_multi_scale=use_multi_scale,
                                            revin=revin,
                                            affine=affine,
                                            subtract_last=subtract_last,
                                            seq_len=self.past_steps,
                                            c_in=self.past_channels,
                                            individual=individual,
                                            target_window=self.future_steps)
            self.model_trend = ModernTCNModel(patch_size=patch_size,
                                              patch_stride=patch_stride,
                                              stem_ratio=stem_ratio,
                                              downsample_ratio=downsample_ratio, 
                                              ffn_ratio=ffn_ratio, 
                                              num_blocks=num_blocks, 
                                              large_size=large_size,
                                              small_size=small_size,
                                              dims=dims,
                                              dw_dims=dw_dims,
                                            nvars=self.past_channels,  ##cosa si aspetta???
                                            small_kernel_merged=small_kernel_merged,
                                            backbone_dropout=drop_backbone,
                                            head_dropout=drop_head,
                                            use_multi_scale=use_multi_scale,
                                            revin=revin, 
                                            affine=affine,
                                            subtract_last=subtract_last,
                                            seq_len=self.past_steps,
                                            c_in=self.past_channels, 
                                            individual=individual,
                                            target_window=self.future_steps)
        else:
            self.model = ModernTCNModel(patch_size=patch_size,
                                        patch_stride=patch_stride,
                                        stem_ratio=stem_ratio,
                                        downsample_ratio=downsample_ratio,
                                        ffn_ratio=ffn_ratio, 
                                        num_blocks=num_blocks,
                                        large_size=large_size, 
                                        small_size=small_size, 
                                        dims=dims,
                                        dw_dims=dw_dims,
                                        nvars=self.past_channels, 
                                        small_kernel_merged=small_kernel_merged, 
                                        backbone_dropout=drop_backbone, 
                                        head_dropout=drop_head, 
                                        use_multi_scale=use_multi_scale, 
                                        revin=revin, 
                                        affine=affine,
                                        subtract_last=subtract_last,
                                        seq_len=self.past_steps, 
                                        c_in=self.past_channels,
                                        individual=individual, 
                                        target_window=self.future_steps)


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
        
        
        tot = [x_seq,emb_past]
    
        x = torch.cat(tot,axis=2)


        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

            res = self.model_res(res_init, None)
            trend = self.model_trend(trend_init, None)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)

            x = self.model(x, None)
            x = x.permute(0, 2, 1)
            
            
        tmp_future.append(x)
        tmp_future = torch.cat(tmp_future,2)
        output = self.final_layer(tmp_future)
        return output.reshape(BS,self.future_steps,self.out_channels,self.mul)
 

'''
    
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

        
        
        
'''