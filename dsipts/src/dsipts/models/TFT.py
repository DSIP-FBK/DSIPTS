import torch
import torch.nn as nn
from .tft import sub_nn

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base
from .utils import QuantileLossMO
from typing import List, Union
from ..data_structure.utils import beauty_string
from .utils import  get_scope
from .utils import Embedding_cat_variables

class TFT(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 d_model: int,
                 num_layers_RNN: int,
                 d_head: int,
                 n_head: int,
                 dropout_rate: float,
                
                 **kwargs)->None:
        """Initializes the model for time series forecasting with attention mechanisms and recurrent neural networks.
        
        This model is designed for direct forecasting, allowing for multi-output and multi-horizon predictions. It leverages attention mechanisms to enhance the selection of relevant past time steps and learn long-term dependencies. The architecture includes RNN enrichment, gating mechanisms to minimize the impact of irrelevant variables, and the ability to output prediction intervals through quantile regression.
        
        Key features include:
        - Direct Model: Predicts all future steps at once.
        - Multi-Output Forecasting: Capable of predicting one or more variables simultaneously.
        - Multi-Horizon Forecasting: Predicts variables at multiple future time steps.
        - Attention-Based Mechanism: Enhances the selection of relevant past time steps and learns long-term dependencies.
        - RNN Enrichment: Utilizes LSTM for initial autoregressive approximation, which is refined by the rest of the network.
        - Gating Mechanisms: Reduces the contribution of irrelevant variables.
        - Prediction Intervals: Outputs percentiles (e.g., 10th, 50th, 90th) at each time step.
        
        The model also facilitates interpretability by identifying:
        - Global importance of variables for both past and future.
        - Temporal patterns.
        - Significant events.
        
        Args:
            d_model (int): General hidden dimension across the network, adjustable in sub-networks.
            num_layers_RNN (int): Number of layers in the recurrent neural network (LSTM).
            d_head (int): Dimension of each attention head.
            n_head (int): Number of attention heads.
            dropout_rate (float): Dropout rate applied uniformly across all dropout layers.
            **kwargs: Additional keyword arguments for further customization.
        """
        
        
        


        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        # assert out_channels==1, logging.info("ONLY ONE CHANNEL IMPLEMENTED")
        self.d_model = d_model
        # linear to embed the target vartiable
        self.target_linear = nn.Linear(self.out_channels, d_model) # same for past and fut! (same variable)
        # number of variables in the past different from the target one(s)
        self.aux_past_channels = self.past_channels - self.out_channels # -1 because one channel is occupied by the target variable
        # one linear for each auxiliar past var
        self.linear_aux_past = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_past_channels)])
        # number of variables in the future used to predict the target one(s)
        self.aux_fut_channels = self.future_channels
        # one linear for each auxiliar future var
        self.linear_aux_fut = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_fut_channels)])
        # length of the full sequence, parameter used for the embedding of all categorical variables
        # - we assume that these are no available or available both for past and future
        seq_len = self.past_steps+self.future_steps
        
        
        ##in v.1.1.5 this is not working, past and future are different for categorical
        #self.emb_cat_var = sub_nn.embedding_cat_variables(seq_len, self.future_steps, d_model, embs, self.device)
        
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels
        
        
        # Recurrent Neural Network for first aproximated inference of the target variable(s) - IT IS NON RE-EMBEDDED YET
        self.rnn = sub_nn.LSTM_Model(num_var=self.out_channels, 
                                     d_model = d_model, 
                                     pred_step = self.future_steps, 
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

        self.outLinear = nn.Linear(d_model, self.out_channels*self.mul)

    def can_be_compiled(self):
        return False  
  

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
        '''
        ### CATEGORICAL VARIABLES changed in 1.1.5
        if 'x_cat_past' in batch.keys() and 'x_cat_future' in batch.keys(): # if we have both
            # HERE WE ASSUME SAME NUMBER AND KIND OF VARIABLES IN PAST AND FUTURE
            cat_past = batch['x_cat_past'].to(self.device)
            cat_fut = batch['x_cat_future'].to(self.device)
            cat_full = torch.cat((cat_past, cat_fut), dim = 1)
            # EMB CATEGORICAL VARIABLES AND THEN SPLIT IN PAST AND FUTURE
            emb_cat_full = self.emb_cat_var(cat_full,self.device)
        else:
            emb_cat_full = self.emb_cat_var(num_past.shape[0],self.device)
            
        cat_emb_past = emb_cat_full[:,:-self.future_steps,:,:]
        cat_emb_fut = emb_cat_full[:,-self.future_steps:,:,:]
        
        ## update summary
        # past
        summary_past = torch.cat((summary_past, cat_emb_past), dim=2)
        # future
        summary_fut = torch.cat((summary_fut, cat_emb_fut), dim=2)
        '''
        BS = num_past.shape[0]
        if 'x_cat_future' in batch.keys():
            emb_fut = self.emb_fut(BS,batch['x_cat_future'].to(self.device))
        else:
            emb_fut = self.emb_fut(BS,None)
        if 'x_cat_past' in batch.keys():
            emb_past = self.emb_past(BS,batch['x_cat_past'].to(self.device))
        else:
            emb_past = self.emb_past(BS,None)
            
        ## update summary
        # past

        summary_past = torch.cat((summary_past, emb_past.unsqueeze(-1).repeat((1,1,1,summary_past.shape[-1]))), dim=2)
        # future
        summary_fut = torch.cat((summary_fut, emb_fut.unsqueeze(-1).repeat((1,1,1,summary_past.shape[-1]))), dim=2)

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
    
