import torch
import torch.nn as nn
from .tft2 import sub_nn
from .base import  Base
from .utils import get_device, QuantileLossMO, L1Loss
from typing import List, Union
import logging

class TFT2(Base):
    def __init__(self, 
                 d_model: int,
                 out_channels:int,
                 past_steps:int,
                 future_steps: int, 
                 past_channels:int,
                 future_channels:int,
                 num_layers_RNN: int,
                 embs: list[int],
                 d_head: int,
                 n_head: int,
                 dropout_rate: float,
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[float]=[],
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None)->None:
        """_summary_

        Args:
            d_model (int): _description_
            out_channels (int): _description_
            past_steps (int): _description_
            future_steps (int): _description_
            past_channels (int): _description_
            future_channels (int): _description_
            num_layers_RNN (int): _description_
            embs (list[int]): _description_
            d_head (int): _description_
            n_head (int): _description_
            dropout_rate (float): _description_
            persistence_weight (float, optional): _description_. Defaults to 0.0.
            loss_type (str, optional): _description_. Defaults to 'l1'.
            quantiles (List[float], optional): _description_. Defaults to [].
            optim (Union[str,None], optional): _description_. Defaults to None.
            optim_config (dict, optional): _description_. Defaults to None.
            scheduler_config (dict, optional): _description_. Defaults to None.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)
        # assert out_channels==1, logging.info("ONLY ONE CHANNEL IMPLEMENTED")
        self.future_steps = future_steps
        self.d_model = d_model
        self.out_channels = out_channels

        self.target_linear = nn.Linear(out_channels, d_model) # same for past and fut! (same variable)
        self.aux_past_channels = past_channels - out_channels # -1 because one channel is occupied by the target variable
        self.linear_aux_past = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_past_channels)])
        self.aux_fut_channels = future_channels
        self.linear_aux_fut = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_fut_channels)])
        seq_len = past_steps+future_steps
        self.emb_cat_var = sub_nn.embedding_cat_variables(seq_len, future_steps, d_model, embs, self.device)
        self.rnn = sub_nn.LSTM_Model(num_var=out_channels, 
                                     d_model = d_model, 
                                     pred_step = future_steps, 
                                     num_layers = num_layers_RNN, 
                                     dropout = dropout_rate)

        self.res_conn1_past = sub_nn.ResidualConnection(d_model, dropout_rate)
        self.res_conn1_fut = sub_nn.ResidualConnection(d_model, dropout_rate)
        self.grn1_past = sub_nn.GRN(d_model, dropout_rate)
        self.grn1_fut = sub_nn.GRN(d_model, dropout_rate)
        self.InterpretableMultiHead = sub_nn.InterpretableMultiHead(d_model, d_head, n_head)
        self.res_conn2_att = sub_nn.ResidualConnection(d_model, dropout_rate)
        self.grn2_att = sub_nn.GRN(d_model, dropout_rate)
        self.res_conn3_out = sub_nn.ResidualConnection(d_model, dropout_rate)

        self.persistence_weight = persistence_weight 
        self.loss_type = loss_type

        # output, handling quantiles or not
        assert (len(quantiles) ==0) or (len(quantiles)==3)
        if len(quantiles)==0:
            self.mul = 1
            self.use_quantiles = False
            self.outLinear = nn.Linear(d_model, out_channels)
            if self.loss_type == 'mse':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()
        else:
            self.mul = len(quantiles)
            self.use_quantiles = True
            self.outLinear = nn.Linear(d_model, out_channels*len(quantiles))
            self.loss = QuantileLossMO(quantiles)
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config

    def forward(self, batch):

        num_past = batch['x_num_past'].to(self.device)
        # PAST TARGET NUMERICAL VARIABLE
        # always available: autoregressive variable
        # compute rnn prediction
        idx_target = batch['idx_target'][0]
        target_num_past = num_past[:,:,idx_target]
        target_emb_num_past = self.target_linear(target_num_past) # target_variables comunicating with each others
        target_num_fut_approx = self.rnn(target_emb_num_past)
        # embed future redictions
        target_emb_num_fut_approx = self.target_linear(target_num_fut_approx)

        ### create variable summary_past and summary_fut
        # at the beggining it is composed only by past and future target variable
        summary_past = target_emb_num_past.unsqueeze(2)
        summary_fut = target_emb_num_fut_approx.unsqueeze(2)
        # now we search for others categorical and numerical variables!


        ### PAST NUMERICAL VARIABLES
        if self.aux_past_channels>0: # so we have more numerical variables about past
            # AUX = AUXILIARY variables
            aux_num_past = self.remove_var(num_past, idx_target, 2) # remove the target index on the second dimension
            assert self.aux_past_channels == aux_num_past.size(2), logging.info(f"{self.aux_past_channels} LAYERS FOR PAST VARS AND {aux_num_past.shape(2)} VARS") # to check if we are using the expected number of variables about past
            aux_emb_num_past = torch.Tensor().to(aux_num_past.device)
            for i, layer in enumerate(self.linear_aux_past):
                aux_emb_past = layer(aux_num_past[:,:,[i]]).unsqueeze(2)
                aux_emb_num_past = torch.cat((aux_emb_num_past, aux_emb_past), dim=2)
            ## update summary about past
            summary_past = torch.cat((summary_past, aux_emb_num_past), dim=2)
        
        ### FUTURE NUMERICAL VARIABLES
        if self.aux_fut_channels>0: # so we have more numerical variables about future
            aux_num_fut = batch['x_num_future'].to(self.device)
            assert self.aux_fut_channels == aux_num_fut.size(2), logging.info(f"{self.aux_fut_channels} LAYERS FOR PAST VARS AND {aux_num_fut.size(2)} VARS")  # to check if we are using the expected number of variables about fut
            aux_emb_num_fut = torch.Tensor().to(aux_num_fut.device)
            for j, layer in enumerate(self.linear_aux_fut):
                aux_emb_fut = layer(aux_num_fut[:,:,[j]]).unsqueeze(2)
                aux_emb_num_fut = torch.cat((aux_emb_num_fut, aux_emb_fut), dim=2)
            ## update summary about future
            summary_fut = torch.cat((summary_fut, aux_emb_num_fut), dim=2)

        ### CATEGORICAL VARIABLES 
        if 'x_cat_past' in batch.keys() and 'x_cat_future' in batch.keys(): # if we have both
            # HERE WE ASSUME SAME NUMBER AND KIND OF VARIABLES IN PAST AND FUTURE
            cat_past = batch['x_cat_past'].to(self.device)
            cat_fut = batch['x_cat_future'].to(self.device)
            cat_full = torch.cat((cat_past, cat_fut), dim = 1)
            # EMB CATEGORICAL VARIABLES AND THEN SPLIT IN PAST AND FUTURE
            emb_cat_full = self.emb_cat_var(cat_full)
            cat_emb_past = emb_cat_full[:,:-self.future_steps,:,:]
            cat_emb_fut = emb_cat_full[:,-self.future_steps:,:,:]
            ## update summary
            # past
            summary_past = torch.cat((summary_past, cat_emb_past), dim=2)
            # future
            summary_fut = torch.cat((summary_fut, cat_emb_fut), dim=2)

        # >>> PAST:
        summary_past = torch.mean(summary_past, dim=2)
        # >>> FUTURE:
        summary_fut = torch.mean(summary_fut, dim=2)

        ### Residual Connection from LSTM
        summary_past = self.res_conn1_past(summary_past, target_emb_num_past)
        summary_fut = self.res_conn1_fut(summary_fut, target_emb_num_fut_approx)

        ### GRN1
        summary_past = self.grn1_past(summary_past)
        summary_fut = self.grn1_fut(summary_fut)

        ### INTERPRETABLE MULTI HEAD ATTENTION
        attention = self.InterpretableMultiHead(summary_fut, summary_past, target_emb_num_past)

        ### Residual Connection from ATT
        attention = self.res_conn2_att(attention, attention)

        ### GRN
        attention = self.grn2_att(attention)

        ### Resuidual Connection from GRN1
        out = self.res_conn3_out(attention, summary_fut)

        ### OUT
        out = self.outLinear(out)

        if self.mul>0:
            out = out.view(-1, self.future_steps, self.out_channels, self.mul)
        return out
    
    #function to extract from batch['x_num_past'] all variables except the one autoregressive
    def remove_var(self, tensor: torch.Tensor, indexes_to_exclude: int, dimension: int)-> torch.Tensor:
        """Function to remove variables from tensors in chosen dimension and position 

        Args:
            tensor (torch.Tensor): starting tensor
            indexes_to_exclude (int): index of the chosen dimension we want t oexclude
            dimension (int): dimension of the tensor on which we want to work

        Returns:
            torch.Tensor: new tensor without the chosen variables
        """

        remaining_idx = torch.tensor([i for i in range(tensor.size(dimension)) if i not in indexes_to_exclude]).to(tensor.device)
        # Select the desired sub-tensor
        extracted_subtensors = torch.index_select(tensor, dim=dimension, index=remaining_idx)
        
        return extracted_subtensors
    
    #python train.py  --config-dir=config_incube_anmartinelli --config-name=config_slurm architecture=tft2