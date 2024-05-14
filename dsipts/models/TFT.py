import torch
import torch.nn as nn
from .tft import sub_nn
from .base import  Base
from .utils import QuantileLossMO
from typing import List, Union
from ..data_structure.utils import beauty_string
from .utils import  get_scope

class TFT(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
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
                 scheduler_config:dict=None,
                 **kwargs)->None:
        """TEMPORAL FUSION TRANSFORMER - Multi-Horizon TimeSeries Forecasting

        - Direct Model: predicting all future step at once..
        - Multi-Output Forecasting: predicting one or more variables.
        - Multi-Horizon Forecasting: predicting variables at multiple future time steps.
        - Attention based: Enhance selection of relevant time steps in the past and learn long-term dependencies. Weights of attention as importance magnitude for each head.
        - RNN Enrichment: Enpowering the initial autoregressive process. The RNN (here LSTM) provides an initial approximation of the target varible(s), then improved by the rest of th Net.
        - Gating Mechanisms: Minimize the contribution of irrelevant variables.
        - Prediction Intervals (Quantile Regression): Outputting percentiles at each timestep. [10th, 50th, 90th] usually.  

        TFT facilitates Interpretability identifying:
        - Global importance of variables for past and for future
        - Temporal patterns
        - Significant events

        Args:
            d_model (int): general hidden dimension across the Net. Could be changed in subNets 
            out_channels (int): number of variables to predict
            past_steps (int): steps of the look-back window
            future_steps (int): steps in the future to be predicted
            past_channels (int): total number of variables available in the past
            future_channels (int): total number of variables available in the future
            num_layers_RNN (int): number of layers for recurrent NN (here LSTM)
            embs (list[int]): embedding dimensions for added categorical variables (here for pos_seq, is_fut, pos_fut)
            d_head (int): attention head dimension
            n_head (int): number of attention heads
            dropout_rate (float): dropout. Common rate for all dropout layers used.
            persistence_weight (float, optional): ASK TO GOBBI. Defaults to 0.0.
            loss_type (str, optional): Type of loss for prediction. Defaults to 'l1'.
            quantiles (List[float], optional):  list of quantiles to predict. If empty, only the exact value. Only empty list or lisst of len 3 allowed. Defaults to [].
            optim (Union[str,None], optional):  ASK TO GOBBI. Defaults to None.
            optim_config (dict, optional):  ASK TO GOBBI. Defaults to None.
            scheduler_config (dict, optional):  ASK TO GOBBI. Defaults to None.
        """


        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        # assert out_channels==1, logging.info("ONLY ONE CHANNEL IMPLEMENTED")
        self.future_steps = future_steps
        self.d_model = d_model
        self.out_channels = out_channels
        # linear to embed the target vartiable
        self.target_linear = nn.Linear(out_channels, d_model) # same for past and fut! (same variable)
        # number of variables in the past different from the target one(s)
        self.aux_past_channels = past_channels - out_channels # -1 because one channel is occupied by the target variable
        # one linear for each auxiliar past var
        self.linear_aux_past = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_past_channels)])
        # number of variables in the future used to predict the target one(s)
        self.aux_fut_channels = future_channels
        # one linear for each auxiliar future var
        self.linear_aux_fut = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_fut_channels)])
        # length of the full sequence, parameter used for the embedding of all categorical variables
        # - we assume that these are no available or available both for past and future
        seq_len = past_steps+future_steps
        self.emb_cat_var = sub_nn.embedding_cat_variables(seq_len, future_steps, d_model, embs, self.device)
        # Recurrent Neural Network for first aproximated inference of the target variable(s) - IT IS NON RE-EMBEDDED YET
        self.rnn = sub_nn.LSTM_Model(num_var=out_channels, 
                                     d_model = d_model, 
                                     pred_step = future_steps, 
                                     num_layers = num_layers_RNN, 
                                     dropout = dropout_rate)
        # PARTS OF TFT:
        # - Residual connections
        # - Gated Residual Network
        # - Interpretable MultiHead Attention
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
        assert (len(quantiles) ==0) or (len(quantiles)==3), beauty_string('Only 3 quantiles are availables, otherwise set quantiles=[]','block',True)
        if len(quantiles)==0:
            self.mul = 1
            self.use_quantiles = False
            self.outLinear = nn.Linear(d_model, out_channels)
            if self.loss_type == 'mse':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()
        else:
            assert len(quantiles)==3, beauty_string('ONLY 3 quantiles premitted','info',True)
            
            self.mul = len(quantiles)
            self.use_quantiles = True
            self.outLinear = nn.Linear(d_model, out_channels*len(quantiles))
            self.loss = QuantileLossMO(quantiles)
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config

    def forward(self, batch:dict) -> torch.Tensor:
        """Temporal Fusion Transformer

        Collectiong Data
        - Extract the autoregressive variable(s)
        - Embedding and compute a first approximated prediction
        - 'summary_past' and 'summary_fut' collecting data about past and future
        Concatenating on the dimension 2 all different datas, which will be mixed through a MEAN over that imension
        Info get from other tensor of the batch taken as input
        
        TFT actual computations
        - Residual Connection for y_past and summary_past
        - Residual Connection for y_fut and summary_fut
        - GRN1 for past and for fut
        - ATTENTION(summary_fut, summary_past, y_past) 
        - Residual Connection for attention itself
        - GRN2 for attention
        - Residual Connection for attention and summary_fut
        - Linear for actual values and reshape 

        Args:
            batch (dict): Keys used are ['x_num_past', 'idx_target', 'x_num_future', 'x_cat_past', 'x_cat_future']

        Returns:
            torch.Tensor: shape [B, self.future_steps, self.out_channels, self.mul] or [B, self.future_steps, self.out_channels] according to quantiles
        """

        num_past = batch['x_num_past'].to(self.device)
        # PAST TARGET NUMERICAL VARIABLE
        # always available: autoregressive variable
        # compute rnn prediction
        idx_target = batch['idx_target'][0]
        target_num_past = num_past[:,:,idx_target]
        target_emb_num_past = self.target_linear(target_num_past) # target_variables comunicating with each others
        target_num_fut_approx = self.rnn(target_emb_num_past)
        # embed future predictions
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
            assert self.aux_past_channels == aux_num_past.size(2), beauty_string(f"{self.aux_past_channels} LAYERS FOR PAST VARS AND {aux_num_past.shape(2)} VARS",'section',True) # to check if we are using the expected number of variables about past
            aux_emb_num_past = torch.Tensor().to(aux_num_past.device)
            for i, layer in enumerate(self.linear_aux_past):
                aux_emb_past = layer(aux_num_past[:,:,[i]]).unsqueeze(2)
                aux_emb_num_past = torch.cat((aux_emb_num_past, aux_emb_past), dim=2)
            ## update summary about past
            summary_past = torch.cat((summary_past, aux_emb_num_past), dim=2)
        
        ### FUTURE NUMERICAL VARIABLES
        if self.aux_fut_channels>0: # so we have more numerical variables about future
            aux_num_fut = batch['x_num_future'].to(self.device)
            assert self.aux_fut_channels == aux_num_fut.size(2), beauty_string(f"{self.aux_fut_channels} LAYERS FOR PAST VARS AND {aux_num_fut.size(2)} VARS",'section',True)  # to check if we are using the expected number of variables about fut
            aux_emb_num_fut = torch.Tensor().to(aux_num_fut.device)
            for j, layer in enumerate(self.linear_aux_fut):
                aux_emb_fut = layer(aux_num_fut[:,:,[j]]).unsqueeze(2)
                aux_emb_num_fut = torch.cat((aux_emb_num_fut, aux_emb_fut), dim=2)
            ## update summary about future
            summary_fut = torch.cat((summary_fut, aux_emb_num_fut), dim=2)
        import pdb
        pdb.set_trace()
        ### CATEGORICAL VARIABLES 
        if 'x_cat_past' in batch.keys() and 'x_cat_future' in batch.keys(): # if we have both
            # HERE WE ASSUME SAME NUMBER AND KIND OF VARIABLES IN PAST AND FUTURE
            cat_past = batch['x_cat_past'].to(self.device)
            cat_fut = batch['x_cat_future'].to(self.device)
            cat_full = torch.cat((cat_past, cat_fut), dim = 1)
            # EMB CATEGORICAL VARIABLES AND THEN SPLIT IN PAST AND FUTURE
            emb_cat_full = self.emb_cat_var(cat_full,batch['x_num_past'].device)
        else:
            emb_cat_full = self.emb_cat_var(num_past.shape[0],batch['x_num_past'].device)
            
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
    
