import torch
from torch import  nn
from .base import Base
from typing import List,Union

from .utils import  QuantileLossMO
from ..data_structure.utils import beauty_string
from .ttm.utils import get_model, get_frequency_token, count_parameters, RMSELoss


class TTM(Base):
    def __init__(self, 
                model_path:str,
                past_steps:int,
                future_steps:int,
                freq_prefix_tuning:bool,
                freq:str,
                prefer_l1_loss:bool,  # exog: set true to use l1 loss
                prefer_longer_context:bool,
                loss_type:str,
                num_input_channels,
                prediction_channel_indices,
                exogenous_channel_indices,
                decoder_mode,
                fcm_context_length,
                fcm_use_mixer,
                fcm_mix_layers,
                fcm_prepend_past,
                enable_forecast_channel_mixing,
                embs:List[int],
                remove_last = False,
                optim:Union[str,None]=None,
                optim_config:dict=None,
                scheduler_config:dict=None,
                verbose = False,
                use_quantiles=False,
                persistence_weight:float=0.0,
                quantiles:List[int]=[],
                
                **kwargs)->None:
   
        super(TTM, self).__init__(verbose)
        self.save_hyperparameters(logger=False)
        beauty_string("BE SURE TO SETUP split_params:  shift:  ${model_configs.future_steps} BECAUSE IT IS REQUIRED",'info',True)
        self.future_steps = future_steps
        self.use_quantiles = use_quantiles
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.persistence_weight = persistence_weight 
        self.loss_type = loss_type
        self.remove_last = remove_last
        self.embs = embs
        self.freq = freq
        self.extend_variables = False
        if len(quantiles)>0:
            assert len(quantiles)==3, beauty_string('ONLY 3 quantiles premitted','info',True)
            self.use_quantiles = True
            self.mul = len(quantiles)
            self.loss = QuantileLossMO(quantiles)
            self.extend_variables = True
            exogenous_channel_indices = [v+2 for v in exogenous_channel_indices]
            prediction_channel_indices.append(1)
            prediction_channel_indices.append(2)
            num_input_channels = num_input_channels + 2
        else:
            self.use_quantiles = False
            self.mul = 1
            if self.loss_type == 'mse':
                self.loss = nn.MSELoss(reduction="mean")
            elif self.loss_type == 'rmse':
                self.loss = RMSELoss()
            else:
                self.loss = nn.L1Loss()
        
        self.model = get_model(
            model_path=model_path,
            context_length=past_steps,
            prediction_length=future_steps,
            freq_prefix_tuning=freq_prefix_tuning,
            freq=freq,
            prefer_l1_loss=prefer_l1_loss,
            prefer_longer_context=prefer_longer_context,
            num_input_channels=num_input_channels,
            decoder_mode=decoder_mode,
            prediction_channel_indices=list(prediction_channel_indices),
            exogenous_channel_indices=list(exogenous_channel_indices),
            fcm_context_length=fcm_context_length,
            fcm_use_mixer=fcm_use_mixer,
            fcm_mix_layers=fcm_mix_layers,
            fcm_prepend_past=fcm_prepend_past,
            #distribution_output='normal',
            #loss='nll',
            #num_parallel_samples=3,
            #loss='mse',
            enable_forecast_channel_mixing=True,
        )
        self.__freeze_backbone()

    def __freeze_backbone(self):
        """
        Freeze the backbone of the model.
        This is useful when you want to fine-tune only the head of the model.
        """
        print(
            "Number of params before freezing backbone",
            count_parameters(self.model),
        )
        # Freeze the backbone of the model
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        # Count params
        print(
            "Number of params after freezing the backbone",
            count_parameters(self.model),
        )
    
    def __scaler(self, input):
        #new_data = torch.tensor([MinMaxScaler().fit_transform(step_data) for step_data in data])
        for i, e in enumerate(self.embs):
            input[:,:,i] = input[:, :, i] / (e-1)
        return input
    
    def __build_tupla_indexes(self, size, target_idx, current_idx):
        count = 0
        permute = list(range(size))
        for i in target_idx:
            permute[i], permute[current_idx[count]] = current_idx[count], permute[i]
            count += 1
        return tuple(permute)

    def __permute_indexes(self, values, target_idx, current_idx):
        if current_idx is None or target_idx is None:
            raise ValueError("Indexes cannot be None")
        if sorted(current_idx) != sorted(target_idx):
            return values[..., self.__build_tupla_indexes(values.shape[-1], target_idx, current_idx)]
        return values
    
    def forward(self, batch):
        x_enc = batch['x_num_past']
        original_indexes = batch['idx_target'][0].tolist()
        original_indexes_future = batch['idx_target_future'][0].tolist()

        if self.extend_variables:
            x_enc = torch.concat([x_enc, x_enc[...,original_indexes], x_enc[...,original_indexes]], dim=-1)
            original_indexes.append(x_enc.shape[-1]-2)
            original_indexes.append(x_enc.shape[-1]-1)

        if 'x_cat_past' in batch.keys():
            x_mark_enc = batch['x_cat_past'].to(torch.float32).to(self.device)
            x_mark_enc = self.__scaler(x_mark_enc)
            past_values = torch.cat((x_enc,x_mark_enc), axis=-1).type(torch.float32)
        else:
            past_values = x_enc
        
        x_dec = torch.tensor([]).to(self.device)
        if 'x_num_future' in batch.keys(): 
            x_dec = batch['x_num_future'].to(self.device)
            if self.extend_variables:
                x_dec = torch.concat([x_dec, x_dec[...,original_indexes_future], x_dec[...,original_indexes_future]], dim=-1)
                original_indexes_future.append(x_dec.shape[-1]-2)
                original_indexes_future.append(x_dec.shape[-1]-1)
        if 'x_cat_future' in batch.keys():
            x_mark_dec = batch['x_cat_future'].to(torch.float32).to(self.device)
            x_mark_dec = self.__scaler(x_mark_dec)
            future_values = torch.cat((x_dec, x_mark_dec), axis=-1).type(torch.float32)
        else:
            future_values = x_dec

        if self.remove_last:
            idx_target = batch['idx_target'][0]
            x_start = x_enc[:,-1,idx_target].unsqueeze(1)
            x_enc[:,:,idx_target]-=x_start 

        past_values = self.__permute_indexes(past_values, self.model.prediction_channel_indices, original_indexes)
        future_values = self.__permute_indexes(future_values, self.model.prediction_channel_indices, original_indexes_future)

        freq_token = get_frequency_token(self.freq).repeat(x_enc.shape[0])

        res = self.model(
            past_values= past_values,
            future_values= future_values,
            past_observed_mask = None,
            future_observed_mask = None,
            output_hidden_states =  False,
            return_dict = False,
            freq_token= freq_token,
            static_categorical_values = None
        )
        #args = None
        #res = self.model(**args)
        BS = res.shape[0]
        return res.reshape(BS,self.future_steps,-1,self.mul)
        
    