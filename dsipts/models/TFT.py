
from torch import  nn
import torch
from .base import  Base
from .utils import get_device, QuantileLossMO, L1Loss
from .tft import encoder, decoder, embedding_nn
import math
from typing import List

class TFT(Base):
    
    def __init__(self,
                 use_target_past:bool,
                 use_yprec_fut:bool,
                 past_steps:int,
                 future_steps:int,
                 past_channels:int,
                 future_channels:int,
                 embs:List[int],
                 d_model:int,
                 num_heads:int,
                 head_size:int,
                 fw_exp:int,
                 dropout:float,
                 n_layer_encoder:int,
                 n_layer_decoder:int,
                 num_layers_RNN:int,
                 out_channels:int,
                 quantiles:List[int]=[],
                 optim_config:dict=None,
                 scheduler_config:dict=None)->None:
        """TFT model

        Strategies:
        - 0 ['ONLY CAT']: / only x_cat_past / x_cat_fut /// direct method
        - 1 ['CONCAT']: / concat(x_cat_past, with x_num_past) and VarSel(concat)-Encoder(concat, concat, concat) / x_cat_fut /// direct method
        - 2 ['ONLY NUM']: / only x_num_past / x_num_fut /// iterative method

        Args:
            use_target_past (bool): _description_
            use_yprec_fut (bool): _description_
            past_steps (int): number of past datapoints used
            future_steps (int): number of future lag to predict
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables
            d_model (int): dimension of embedded variables
            n_layer_enc (int): number of Encoder layers
            n_layer_dec (int): number of Decoder layers
            n_layer_LSTM (int): number of LSTM layers
            num_heads (int): number of heads used in the Attention
            head_size (int): size for variable reshaping in Attention heads
            fw_exp (int): multiplicative term for forward expansion in the Attention part, d_model-> d_model*fw_exp-> d_model
            dropout (float): _
            out_channels (int): number of output channels
            quantiles (List[int], optional): use quantile loss il len(quantiles) = 0 (usually 0.1,0.5, 0.9) or L1loss in case len(quantiles)==0. Defaults to [].
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
        """
        
        ##pytotch lightening stuff
        self.save_hyperparameters(logger=False)
        
        super().__init__()
        self.past_steps = past_steps
        self.past_channels = past_channels
        self.future_steps = future_steps
        self.seq_len = past_steps + future_steps
        self.d_model = d_model
        self.head_size = d_model # it can vary according to strategies
        self.emb_cat_var = embedding_nn.embedding_cat_variables(self.seq_len, future_steps, d_model, embs)
        self.emb_num_past_var = embedding_nn.embedding_num_variables(past_steps, past_channels, d_model)
        self.emb_y_var = embedding_nn.embedding_target(d_model)
        # Encoder (past)
        self.EncVariableSelection = embedding_nn.Encoder_Var_Selection(use_target_past, len(embs)+3, out_channels, d_model, dropout)
        self.EncLSTM = embedding_nn.Encoder_LSTM(num_layers_RNN, d_model, dropout)
        self.EncGRN = embedding_nn.GRN(d_model, dropout)
        self.Encoder = encoder.Encoder(n_layer_encoder, d_model, num_heads, self.head_size, fw_exp, dropout)
        # Decoder (future)
        self.DecVariableSelection = embedding_nn.Decoder_Var_Selection(use_yprec_fut, len(embs)+3, out_channels, d_model, dropout)
        self.DecLSTM = embedding_nn.Decoder_LSTM(num_layers_RNN, d_model, dropout)
        self.DecGRN = embedding_nn.GRN(d_model, dropout)
        self.Decoder = decoder.Decoder(n_layer_decoder, d_model, num_heads, self.head_size, fw_exp, future_steps, dropout)
        # PostTransformer (future)
        self.postTransformer = embedding_nn.postTransformer(d_model, dropout)
        if len(quantiles)==0:
            self.mul = 1
            self.outLinear = nn.Linear(d_model, out_channels)
        else:
            self.mul = 3
            self.outLinear = nn.Linear(d_model, out_channels*len(quantiles))
        self.final_linear = nn.ModuleList()
        # for _ in range(out_channels*self.mul):
        #     self.final_linear.append(nn.Sequential(nn.Linear(d_model,d_model*2),nn.ReLU(),nn.Dropout(0,2),
        #                                            nn.Linear(d_model*2,d_model),nn.ReLU(),nn.Dropout(0,2),
        #                                            nn.Linear(d_model,d_model//2),nn.ReLU(),nn.Dropout(0,2),
        #                                            nn.Linear(d_model//2,1)))

        assert (len(quantiles) ==0) or (len(quantiles)==3)
        if len(quantiles)>0:
            self.use_quantiles = True
            self.mul = 3 
        else:
            self.use_quantiles = False
            self.mul = 1
            
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
  
        if  self.use_quantiles:
            self.loss = QuantileLossMO(quantiles)
        else:
            self.loss = L1Loss()
        
    def forward(self, batch:dict)->torch.tensor:
        """It is mandatory to implement this method

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        ## ONLY CATEGORICAL, just to start
        import pdb
        pdb.set_trace()
        embed_num_past = self.emb_num_past_var(batch['x_num_past'])
        # embed_past = torch.cat((embed_num_past, embed_categorical_past), dim = 2)
        if 'x_cat_past' in batch.keys():
            x_cat_past = batch['x_cat_past']
            x_cat_future = batch['x_cat_future']
            
            embed_categorical = self.emb_cat_var(torch.cat((x_cat_past,x_cat_future), dim=1))
            embed_categorical_past = embed_categorical[:,:self.past_steps,:,:]
            embed_categorical_future = embed_categorical[:,-self.future_steps:,:,:]

            variable_selection_past = self.EncVariableSelection(embed_categorical_past)
            past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
            encoding = self.EncGRN(past_LSTM)
            encoded = self.Encoder(encoding, encoding, encoding)

            variable_selection_fut = self.DecVariableSelection(embed_categorical_future)
            fut_LSTM = self.DecLSTM(variable_selection_fut, hn, cn)
            decoding = self.DecGRN(fut_LSTM)
            decoded = self.Decoder(decoding, encoded, encoded)
            out = self.postTransformer(decoded, decoding, fut_LSTM)
            out = self.outLinear(out)
            B,L,_ = out.shape
        return out.reshape(B,L,-1,self.mul)

    
    def inference(self, batch:dict)->torch.tensor:
        """Care here, we need to implement it because for predicting the N-step it will use the prediction at step N-1. TODO fix if because I did not implement the
        know continuous variable presence here

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        return self(batch)