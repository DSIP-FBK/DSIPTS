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
                 use_yprec_fut: bool,
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
        """TFT model 'arXiv:1912.09363v3 [stat.ML] 27 Sep 2020'

        Strategies:
        - use_target_past: choose if you want to consider also the past numerical variables for the encoding part
        - use_y_prec_fut: choose if for prediction you want to use the previous values of the target variables. If True, the process becomes iterative (also slower)

        Args:
            use_target_past (bool): usage of past numerical variables
            use_yprec_fut (bool): usage of iterative procedure 
            past_steps (int): past context steps
            future_steps (int): future steps to be predicted
            past_channels (int): number of variables used in past steps
            future_channels (int): [now not used] number of future numerical variables for predictions
            embs (List[int]): embedding dimensions for Embedding Layers of categorical variables
            d_model (int): dimension of the model
            num_heads (int): number of heads used in Encoder and Decoder multihead attentions
            head_size (int): size of tensors 
            fw_exp (int): _description_
            dropout (float): _description_
            n_layer_encoder (int): _description_
            n_layer_decoder (int): _description_
            num_layers_RNN (int): _description_
            out_channels (int): _description_
            quantiles (List[int], optional): _description_. Defaults to [].
            optim_config (dict, optional): _description_. Defaults to None.
            scheduler_config (dict, optional): _description_. Defaults to None.
        """
        ##pytotch lightening stuff
        self.save_hyperparameters(logger=False)
        
        super().__init__()
        # strategy applied
        self.use_target_past = use_target_past
        self.use_yprec_fut = use_yprec_fut
        
        # params for structure of data and model
        self.past_steps = past_steps
        self.past_channels = past_channels
        self.future_steps = future_steps
        self.seq_len = past_steps + future_steps
        self.d_model = d_model
        self.out_channels = out_channels
        self.head_size = head_size # it can vary according to strategies
        
        # NN of the model:

        # - Starting Embedding
        self.emb_cat_var = embedding_nn.embedding_cat_variables(self.seq_len, future_steps, d_model, embs)
        if use_target_past:
            self.emb_num_past_var = embedding_nn.embedding_num_past_variables(past_channels, d_model)
        if use_yprec_fut:
            self.emb_num_fut_var = embedding_nn.embedding_num_future_variables(future_steps, out_channels, d_model)

        # - Encoder (past)
        self.EncVariableSelection = embedding_nn.Encoder_Var_Selection(self.use_target_past, len(embs)+3, past_channels, d_model, dropout)
        self.EncLSTM = embedding_nn.Encoder_LSTM(num_layers_RNN, d_model, dropout)
        self.EncGRN = embedding_nn.GRN(d_model, dropout)
        self.Encoder = encoder.Encoder(n_layer_encoder, d_model, num_heads, self.head_size, fw_exp, dropout)
        
        # - Decoder (future)
        self.DecVariableSelection = embedding_nn.Decoder_Var_Selection(self.use_yprec_fut, len(embs)+3, out_channels+1, d_model, dropout)
        self.DecLSTM = embedding_nn.Decoder_LSTM(num_layers_RNN, d_model, dropout)
        self.DecGRN = embedding_nn.GRN(d_model, dropout)
        self.Decoder = decoder.Decoder(n_layer_decoder, d_model, num_heads, self.head_size, fw_exp, future_steps, dropout)
        
        # - PostTransformer (future)
        self.postTransformer = embedding_nn.postTransformer(d_model, dropout)

        # quantiles defines the number of predicted values: only y or also 0.1 and 0.9 quantiles
        # init the last linear, loss and mul according tothe size of quantiles
        assert (len(quantiles) ==0) or (len(quantiles)==3)
        if len(quantiles)==0:
            self.mul = 1
            self.use_quantiles = False
            self.outLinear = nn.Linear(d_model, out_channels)
            self.loss = L1Loss()
        else:
            self.mul = 3
            self.use_quantiles = True
            self.outLinear = nn.Linear(d_model, out_channels*len(quantiles))
            self.loss = QuantileLossMO(quantiles)

        # ask to aGobbi
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
  
    def forward(self, batch:dict) -> torch.Tensor:
        """ --- TFT ---
        This Model can perform iterative and direct predictions:\n
        ITERATIVE:\n
        yhat_i(q,t,tau) = f_q( tau, y_{i,t-k:t}, x_{i,t-k:t+tau} )\n
        DIRECT:\n
        yhat_i(q,t,tau) = f_q( tau, x_{i,t-k:t+tau} )\n
         
        - q is the quantile taken in consideration,
        - t the time when the prediction starts, 
        - tau the future step we are predicting

        Structure of the Model:
        - Gating Mechanisms: GRN and GLU give the model the flexibility 
        to apply non-linear processing only where needed.
        - Variable Selection Network: it is intended as instance-wise variable selection 
        removing any unnecessary noisy inputs and focunsing on the most saient ones
        - Interpretable Multi-head Attention: weights shared across all heads
        - Quantile output: the model can predict the single value of y or its quantiles.

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.Tensor: [Bs, future_steps, -1, number of quantiles computed for each timestep]
        """
        
        # categorical past var
        x_cat_past = batch['x_cat_past'].to(self.device)
        x_cat_future = batch['x_cat_future'].to(self.device)
        #embed all categorical values
        embed_categorical = self.emb_cat_var(torch.cat((x_cat_past,x_cat_future), dim=1))
        # split for past and future
        embed_categorical_past = embed_categorical[:,:self.past_steps,:,:]
        embed_categorical_future = embed_categorical[:,-self.future_steps:,:,:]

        ### PAST #############
        #Variable Selection
        if self.use_target_past:
            # numerical past var
            embed_num_past = self.emb_num_past_var(batch['x_num_past'].to(self.device))
            variable_selection_past = self.EncVariableSelection(embed_categorical_past, embed_num_past)
        else:
            variable_selection_past = self.EncVariableSelection(embed_categorical_past)
        # LSTM and GRN
        past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
        encoding = self.EncGRN(past_LSTM)
        # Encoder
        encoded = self.Encoder(encoding, encoding, encoding)

        ### FUTURE ###########
        if self.use_yprec_fut:
            # init output tensor on the right device
            device = batch['x_num_past'].device.type
            output = torch.Tensor().to(device)
            # init decoder_out to store actual value predicted of the target variable
            idx_target = batch['idx_target'][0,0].item()
            decoder_out = batch['x_num_past'][:,-1,idx_target].unsqueeze(1).unsqueeze(2).to(self.device)

            # start iterative procedure
            for tau in range(1,self.future_steps+1):
                embed_tau_y = self.emb_num_fut_var(decoder_out)
                variable_selection_fut = self.DecVariableSelection(embed_categorical_future[:,:tau,:,:], embed_tau_y)
                fut_LSTM = self.DecLSTM(variable_selection_fut, hn, cn)
                pred_decoding = self.DecGRN(fut_LSTM)
                pred_decoded = self.Decoder(pred_decoding, encoded, encoded)
                out = self.postTransformer(pred_decoded, pred_decoding, fut_LSTM)
                out = self.outLinear(out) # [B, tau, self.mul]
                # self.mul, by assert in __init__, ==1 or ==3
                if self.mul==1:
                    decoder_out = torch.cat((decoder_out, out[:,-1,:].unsqueeze(2)), dim = 1)
                    output = torch.cat((output, out[:,-1,:].unsqueeze(1)), dim = 1)
                else:
                    # self.mul ==3 -> store the median quantile[q=0.5] predicted in the dec_out
                    dec_out = out[:,:,1].unsqueeze(2)
                    # cat to decoder_out only the actual value
                    decoder_out = torch.cat((decoder_out, dec_out[:,-1,:].unsqueeze(2)), dim = 1)
                    # cat to output all the quantiles predicted
                    output = torch.cat((output, out[:,-1,:].unsqueeze(1)), dim = 1)
            # ignore the first value y_0 used to start the iterative procedure
            out = decoder_out[:,1:,:]
            B, L, _ = output.shape
            return output.reshape(B,L,-1,self.mul)
        else:
            # direct prediction mod
            variable_selection_fut = self.DecVariableSelection(embed_categorical_future)
            fut_LSTM = self.DecLSTM(variable_selection_fut, hn, cn)
            decoding = self.DecGRN(fut_LSTM)
            decoded = self.Decoder(decoding, encoded, encoded)
            out = self.postTransformer(decoded, decoding, fut_LSTM)
            out = self.outLinear(out)
            B, L, _ = out.shape
            return out.reshape(B,L,-1,self.mul)
        # daje roma
    
    def inference(self, batch:dict)->torch.tensor:
        """Exactly equal to forward method

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        return self(batch) # exactly the same computations of forward