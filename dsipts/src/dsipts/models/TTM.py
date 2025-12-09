import torch
import numpy as np
from torch import  nn

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base


from .ttm.utils import get_model, get_frequency_token, count_parameters, DEFAULT_FREQUENCY_MAPPING
from ..data_structure.utils import beauty_string
from .utils import  get_scope

class TTM(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                model_path:str,
                prefer_l1_loss:bool,  # exog: set true to use l1 loss
                prefer_longer_context:bool,
                prediction_channel_indices,
                exogenous_channel_indices_cont,
                exogenous_channel_indices_cat,
                decoder_mode,
                freq,
                freq_prefix_tuning,
                fcm_context_length,
                fcm_use_mixer,
                fcm_mix_layers,
                fcm_prepend_past,
                enable_forecast_channel_mixing,
                force_return,
                few_shot = True,
                **kwargs)->None:
   
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        

        self.index_fut = list(exogenous_channel_indices_cont)

        if len(exogenous_channel_indices_cat)>0:

            self.index_fut_cat = [self.past_channels+c for c in list(exogenous_channel_indices_cat)]

        else:
            self.index_fut_cat = []
        self.freq = freq

        base_freq_token = get_frequency_token(self.freq)  # e.g., shape [n_token] or scalar
        # ensure it's a tensor of integer type
        if not torch.is_tensor(base_freq_token):
            base_freq_token = torch.tensor(base_freq_token)
        base_freq_token = base_freq_token.long()
        self.register_buffer("token", base_freq_token, persistent=True)
                
        
        self.model = get_model(
            model_path=model_path,
            context_length=self.past_steps,
            prediction_length=self.future_steps,
            prefer_l1_loss=prefer_l1_loss,
            prefer_longer_context=prefer_longer_context,
            num_input_channels=self.past_channels+len(self.embs_past), #giusto
            decoder_mode=decoder_mode,
            prediction_channel_indices=list(prediction_channel_indices),
            exogenous_channel_indices=self.index_fut + self.index_fut_cat,
            fcm_context_length=fcm_context_length,
            fcm_use_mixer=fcm_use_mixer,
            fcm_mix_layers=fcm_mix_layers,
            freq=freq,
            force_return=force_return,
            freq_prefix_tuning=freq_prefix_tuning,
            fcm_prepend_past=fcm_prepend_past,
            enable_forecast_channel_mixing=enable_forecast_channel_mixing,
            
        )
        hidden_size =  self.model.config.hidden_size
        self.model.prediction_head = torch.nn.Linear(hidden_size, self.out_channels*self.mul)
        if few_shot:
            self._freeze_backbone()
        self.zero_pad = (force_return=='zeropad') 
    def _freeze_backbone(self):
        """
        Freeze the backbone of the model.
        This is useful when you want to fine-tune only the head of the model.
        """
        beauty_string(f"Number of params before freezing backbone:{count_parameters(self.model)}",'info',self.verbose)
        
        # Freeze the backbone of the model
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        # Count params
        beauty_string(f"Number of params after freezing the backbone: {count_parameters(self.model)}",'info',self.verbose)
            
    
    def _scaler_past(self, input):
        for i, e in enumerate(self.embs_past):
            input[:,:,i] = input[:, :, i] / (e-1)
        return input
    def _scaler_fut(self, input):
        for i, e in enumerate(self.embs_fut):
            input[:,:,i] = input[:, :, i] / (e-1)
        return input

    def can_be_compiled(self):
        
        return True#True#not self.zero_pad  
  
    def forward(self, batch):
        x_enc = batch['x_num_past'].to(self.device)

        
        if self.zero_pad:
            B,L,C = batch['x_num_past'].shape
            x_enc = torch.zeros((B,512,C)).to(self.device)
            x_enc[:,-L:,:] = batch['x_num_past'].to(self.device)
        else:
            x_enc = batch['x_num_past'].to(self.device)
        original_indexes = batch['idx_target'][0].tolist()


        if 'x_cat_past' in batch.keys():
            if self.zero_pad:
                B,L,C = batch['x_cat_past'].shape
                x_mark_enc = torch.zeros((B,512,C)).to(self.device)
                x_mark_enc[:,-L:,:] = batch['x_cat_past'].to(torch.float32).to(self.device)
            else:
                x_mark_enc = batch['x_cat_past'].to(torch.float32).to(self.device)
                x_mark_enc = self._scaler_past(x_mark_enc)
            past_values = torch.cat((x_enc,x_mark_enc), axis=-1).type(torch.float32)
        else:
            past_values = x_enc
        B,L,C = past_values.shape
        future_values = torch.zeros((B,self.future_steps,C)).to(self.device)
        

   
        if 'x_num_future' in batch.keys(): 
            future_values[:,:,self.index_fut] = batch['x_num_future'].to(self.device)
        if 'x_cat_future' in batch.keys():
            x_mark_dec = batch['x_cat_future'].to(torch.float32).to(self.device)
            x_mark_dec = self._scaler_fut(x_mark_dec)
            future_values[:,:,self.index_fut_cat] = x_mark_dec
        

        #investigating!! problem with dynamo!
        #freq_token = get_frequency_token(self.freq).repeat(past_values.shape[0])

        batch_size = past_values.shape[0]
        freq_token = self.token.repeat(batch_size).long().to(self.device)


        res = self.model(
            past_values= past_values,
            future_values= future_values,# future_values if future_values.shape[0]>0 else None,
            past_observed_mask = None,
            future_observed_mask = None,
            output_hidden_states =  False,
            return_dict = False,
            freq_token= freq_token,#[0:past_values.shape[0]], ##investigating
            static_categorical_values = None
        )


        BS = res.shape[0]

        return res.reshape(BS,self.future_steps,-1,self.mul)
        
    