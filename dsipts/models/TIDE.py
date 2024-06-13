import torch
import torch.nn as nn
import numpy as np
from .tft import sub_nn
from .base import  Base
from .utils import  QuantileLossMO
from typing import List, Union
from ..data_structure.utils import beauty_string
from .utils import  get_scope

class TIDE(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 out_channels: int,
                 past_steps: int,
                 future_steps: int, 
                 past_channels: int,
                 future_channels: int,
                 embs: List[int],

                 # specific params
                 hidden_size:int,
                 d_model: int,
                 n_add_enc: int,
                 n_add_dec: int,
                 dropout_rate: float,
                 activation: str='',

                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[float]=[],
                 optim:Union[str,None]=None,
                 optim_config:Union[dict,None]=None,
                 scheduler_config:Union[dict,None]=None,
                 **kwargs)->None:
        """Long-term Forecasting with TiDE: Time-series Dense Encoder
        https://arxiv.org/abs/2304.08424

        This NN uses as subnet the ResidualBlocks, which is composed by skip connection and activation+dropout.
        Every encoder and decoder head is composed by one Residual Block, like the temporal decoder and the feature projection for covariates.

        Args:
            out_channels (int): number of variables to be predicted
            past_steps (int): Lookback window length
            future_steps (int): Horizon window length
            past_channels (int): number of past variables
            future_channels (int): number of future auxiliary variables 
            embs (List[int]): 
            hidden_size (int): first embedding size of the model ('r' in the paper)
            d_model (int): second embedding size (r^{tilda} in the model). Should be smaller than hidden_size
            n_add_enc (int): number of OTHERS heads for the encoder part in the NN. 1 is always used by default.
            n_add_dec (int): number of OTHERS heads for the decoder part in the NN. 1 is always used by default.
            dropout_rate (float): 
            activation (str, optional): activation function to be used in the Residual Block. E.g., 'nn.GELU'. Defaults to ''.
            persistence_weight (float, optional): Defaults to 0.0.
            loss_type (str, optional): Defaults to 'l1'.
            quantiles (List[float], optional): Defaults to [].
            optim (Union[str,None], optional): Defaults to None.
            optim_config (Union[dict,None], optional): Defaults to None.
            scheduler_config (Union[dict,None], optional): Defaults to None.
        """
        
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        # self.dropout = dropout_rate
        self.persistence_weight = persistence_weight 
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.loss_type = loss_type

                
        if len(quantiles)==0:
            self.mul = 1
            self.use_quantiles = False
            if self.loss_type == 'mse':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()
        else:
            assert len(quantiles)==3, beauty_string('ONLY 3 quantiles premitted','info',True)
            self.mul = len(quantiles)
            self.use_quantiles = True
            self.loss = QuantileLossMO(quantiles)
        
        self.hidden_size = hidden_size # r
        self.d_model = d_model # r^tilda
        self.past_steps = past_steps # lookback size
        self.future_steps = future_steps # horizon size
        self.past_channels = past_channels # psat_vars
        self.future_channels = future_channels # fut_vars
        self.output_channels = out_channels # target_vars

        # for other numerical variables in the past
        self.aux_past_channels = past_channels - out_channels
        self.linear_aux_past = nn.ModuleList([nn.Linear(1, self.hidden_size) for _ in range(self.aux_past_channels)])

        # for numerical variables in the future
        self.aux_fut_channels = future_channels
        self.linear_aux_fut = nn.ModuleList([nn.Linear(1, self.hidden_size) for _ in range(self.aux_fut_channels)])
        
        # embedding categorical for both past and future (ASSUMING BOTH AVAILABLE OR NO ONE)
        self.seq_len = past_steps + future_steps
        self.emb_cat_var = sub_nn.embedding_cat_variables(self.seq_len, future_steps, hidden_size, embs, self.device)

        ## FEATURE PROJECTION
        # past
        if self.aux_past_channels>0:
            self.feat_proj_past = ResidualBlock(2*hidden_size, d_model, dropout_rate, activation)
        else:
            self.feat_proj_past = ResidualBlock(hidden_size, d_model, dropout_rate, activation)
        # future
        if self.aux_fut_channels>0:
            self.feat_proj_fut = ResidualBlock(2*hidden_size, d_model, dropout_rate, activation)
        else:
            self.feat_proj_fut = ResidualBlock(hidden_size, d_model, dropout_rate, activation)

        # # ENCODER
        self.enc_dim_input = past_steps*self.output_channels + (past_steps+future_steps)*d_model 
        self.enc_dim_output = future_steps*d_model
        self.first_encoder = ResidualBlock(self.enc_dim_input, self.enc_dim_output, dropout_rate, activation)
        self.aux_encoder = nn.ModuleList([ResidualBlock(self.enc_dim_output, self.enc_dim_output, dropout_rate, activation) for _ in range(1, n_add_enc)])

        # # DECODER
        self.first_decoder = ResidualBlock(self.enc_dim_output, self.enc_dim_output, dropout_rate, activation)
        self.aux_decoder = nn.ModuleList([ResidualBlock(self.enc_dim_output, self.enc_dim_output, dropout_rate, activation) for _ in range(1, n_add_dec)])

        ## TEMPORAL DECOER
        self.temporal_decoder = ResidualBlock(2*d_model, out_channels*self.mul, dropout_rate, activation)

        # linear for Y lookback
        self.linear_target = nn.Linear(past_steps*out_channels, future_steps*out_channels*self.mul)


    def forward(self, batch:dict)-> float:
        """training process of the diffusion network

        Args:
            batch (dict): variables loaded

        Returns:
            float: total loss about the prediction of the noises over all subnets extracted
        """

        # LOADING AUTOREGRESSIVE CONTEXT OF TARGET VARIABLES

        num_past = batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        y_past = num_past[:,:,idx_target]
        B = y_past.shape[0]

        # LOADING EMBEDDING CATEGORICAL VARIABLES
        emb_cat_past, emb_cat_fut = self.cat_categorical_vars(batch)
      
        emb_cat_past = torch.mean(emb_cat_past, dim = 2)
        emb_cat_fut = torch.mean(emb_cat_fut, dim = 2)
      
        ### LOADING PAST AND FUTURE NUMERICAL VARIABLES
        # load in the model auxiliar numerical variables

        if self.aux_past_channels>0: # if we have more numerical variables about past
            aux_num_past = self.remove_var(num_past, idx_target, 2) # remove the autoregressive variable
            assert self.aux_past_channels == aux_num_past.size(2),  beauty_string(f"{self.aux_past_channels} LAYERS FOR PAST VARS AND {aux_num_past.size(2)} VARS",'section',True) # to check if we are using the expected number of variables about past
            # concat all embedded vars and mean of them
            aux_emb_num_past = torch.Tensor().to(self.device)
            for i, layer in enumerate(self.linear_aux_past):
                aux_emb_past = layer(aux_num_past[:,:,[i]]).unsqueeze(2)
                aux_emb_num_past = torch.cat((aux_emb_num_past, aux_emb_past), dim=2)
            aux_emb_num_past = torch.mean(aux_emb_num_past, dim = 2)
        else: 
            aux_emb_num_past = None # non available vars
            
        if self.aux_fut_channels>0: # if we have more numerical variables about future
            # AUX means AUXILIARY variables
            aux_num_fut = batch['x_num_future'].to(self.device)
            assert self.aux_fut_channels == aux_num_fut.size(2), beauty_string(f"{self.aux_fut_channels} LAYERS FOR PAST VARS AND {aux_num_fut.size(2)} VARS",'section',True)  # to check if we are using the expected number of variables about fut
            # concat all embedded vars and mean of them
            aux_emb_num_fut = torch.Tensor().to(self.device)
            for j, layer in enumerate(self.linear_aux_fut):
                aux_emb_fut = layer(aux_num_fut[:,:,[j]]).unsqueeze(2)
                aux_emb_num_fut = torch.cat((aux_emb_num_fut, aux_emb_fut), dim=2)
            aux_emb_num_fut = torch.mean(aux_emb_num_fut, dim = 2)
        else:
            aux_emb_num_fut = None # non available vars

        # past^tilda
        if self.aux_past_channels>0:
            emb_past = torch.cat((emb_cat_past, aux_emb_num_past), dim = 2) # [B, L, 2R] #
            proj_past = self.feat_proj_past(emb_past, True) # [B, L, R^tilda] #
        else:
            proj_past = self.feat_proj_past(emb_cat_past, True) # [B, L, R^tilda] #

        # fut^tilda
        if self.aux_fut_channels>0:
            emb_fut = torch.cat((emb_cat_fut, aux_emb_num_fut), dim = 2) # [B, H, 2R] #
            proj_fut = self.feat_proj_fut(emb_fut, True) # [B, H, R^tilda] #
        else:
            proj_fut = self.feat_proj_fut(emb_cat_fut, True) # [B, H, R^tilda] #
            
        concat = torch.cat((y_past.view(B, -1), proj_past.view(B, -1), proj_fut.view(B, -1)), dim = 1) # [B, L*self.mul + (L+H)*R^tilda] #
        dense_enc = self.first_encoder(concat)
        for lay_enc in self.aux_encoder:
            dense_enc = lay_enc(dense_enc)

        dense_dec = self.first_decoder(dense_enc)
        for lay_dec in self.aux_decoder:
            dense_dec = lay_dec(dense_dec)
        
        temp_dec_input = torch.cat((dense_dec.view(B, self.future_steps, self.d_model), proj_fut), dim = 2)
        temp_dec_output = self.temporal_decoder(temp_dec_input, False)
        temp_dec_output = temp_dec_output.view(B, self.future_steps, self.output_channels, self.mul)

        linear_regr = self.linear_target(y_past.view(B, -1))
        linear_output = linear_regr.view(B, self.future_steps, self.output_channels, self.mul)

        output = temp_dec_output + linear_output
        return output


    # function to concat embedded categorical variables
    def cat_categorical_vars(self, batch:dict):
        """Extracting categorical context about past and future

        Args:
            batch (dict): Keys checked -> ['x_cat_past', 'x_cat_future']

        Returns:
            List[torch.Tensor, torch.Tensor]: cat_emb_past, cat_emb_fut
        """
        cat_past = None
        cat_fut = None
        # GET AVAILABLE CATEGORICAL CONTEXT
        if 'x_cat_past' in batch.keys():
            cat_past = batch['x_cat_past'].to(self.device)
        if 'x_cat_future' in batch.keys():
            cat_fut = batch['x_cat_future'].to(self.device)
        # CONCAT THEM, according to self.emb_cat_var usage  
        if cat_past is None:
            emb_cat_full = self.emb_cat_var(batch['x_num_past'].shape[0],self.device)

        else:
            cat_full = torch.cat((cat_past, cat_fut), dim = 1)
            emb_cat_full = self.emb_cat_var(cat_full,self.device)
        cat_emb_past = emb_cat_full[:,:self.past_steps,:,:]
        cat_emb_fut = emb_cat_full[:,-self.future_steps:,:,:]

        return cat_emb_past, cat_emb_fut

    #function to extract from batch['x_num_past'] all variables except the one autoregressive
    def remove_var(self, tensor: torch.Tensor, indexes_to_exclude: list, dimension: int)-> torch.Tensor:
        """Function to remove variables from tensors in chosen dimension and position 

        Args:
            tensor (torch.Tensor): starting tensor
            indexes_to_exclude (list): index of the chosen dimension we want t oexclude
            dimension (int): dimension of the tensor on which we want to work (not list od dims!!)

        Returns:
            torch.Tensor: new tensor without the chosen variables
        """

        remaining_idx = torch.tensor([i for i in range(tensor.size(dimension)) if i not in indexes_to_exclude]).to(tensor.device)
        # Select the desired sub-tensor
        extracted_subtensors = torch.index_select(tensor, dim=dimension, index=remaining_idx)
        
        return extracted_subtensors


class ResidualBlock(nn.Module):
    def __init__(self, in_size:int, out_size:int, dropout_rate:float, activation_fun:str=''):
        """Residual Block as basic layer of the archetecture. 

        MLP with one hidden layer, activation and skip connection
        Basically dimension d_model, but better if input_dim and output_dim are explicit

        in_size and out_size to handle dimensions at different stages of the NN

        Args:
            in_size (int): 
            out_size (int): 
            dropout_rate (float): 
            activation_fun (str, optional): activation function to use in the Residual Block. Defaults to nn.ReLU.
        """
        super().__init__()

        self.direct_linear = nn.Linear(in_size, out_size, bias = False)

        if activation_fun=='':
            self.act = nn.ReLU()
        else:
            activation = eval(activation_fun)
            self.act = activation()
        self.lin = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.final_norm = nn.LayerNorm(out_size)


    def forward(self, x, apply_final_norm = True):
        direct_x = self.direct_linear(x)

        x = self.dropout(self.lin(self.act(x)))

        out = x + direct_x
        if apply_final_norm:
            return self.final_norm(out)
        return out