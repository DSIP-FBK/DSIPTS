import torch
import torch.nn as nn
from .tft2 import sub_nn
from .base import  Base
from .utils import get_device, QuantileLossMO, L1Loss
from typing import List, Union

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
                 n_cross_att: int,
                 n_fut_att: int,
                 dropout_rate: float,
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[float]=[],
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None)->None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.future_steps = future_steps
        self.d_model = d_model
        self.out_channels = out_channels
        self.n_cross_att = n_cross_att
        self.cross_att_val_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_cross_att)])
        self.n_fut_att = n_fut_att
        self.fut_att_val_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_fut_att)])
        self.register_buffer('tril',torch.tril(torch.ones(future_steps, future_steps))) # create the variable 'self.tril'
        self.x_linear = nn.Linear(past_channels, d_model)
        seq_len = past_steps+future_steps
        self.emb_cat_var = sub_nn.embedding_cat_variables(seq_len, future_steps, d_model, embs, self.device) # [12, 31, 24, 4]
        self.rnn = sub_nn.LSTM_Model(d_model, d_model, future_steps, num_layers_RNN, dropout_rate)
        self.grn_past = sub_nn.GRN(d_model, dropout_rate)
        self.grn_fut = sub_nn.GRN(d_model, dropout_rate)
        self.grn_output = sub_nn.GRN(d_model, dropout_rate)
        self.out_glu = sub_nn.GLU(d_model)
        self.cross_dropout = nn.Dropout(dropout_rate)
        self.fut_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)
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
        # to device
        tot = []
        x_past = batch['x_num_past'].to(self.device)
        if 'x_cat_past' in batch.keys():
            cat_past = batch['x_cat_past'].to(self.device)
            tot.append(cat_past)
        if 'x_cat_future' in batch.keys():
            cat_fut = batch['x_cat_future'].to(self.device)
            tot.append(cat_fut)

        # EMBEDDING CATEGORICAL VARIABLES
        # embed all categorical variables and split in past and future ones
        if len(tot)>0:
            cat_full = torch.cat(tot, dim = 1)
        else:
            cat_full = torch.tensor(x_past.shape[0]).to(self.device) ##ACCROCCHIO PER FARE ANDARE LE COSE SE NON HO CATEGORICHE
        emb_cat_full = self.emb_cat_var(cat_full)
        cat_emb_past = emb_cat_full[:,:-self.future_steps,:,:]
        cat_emb_fut = emb_cat_full[:,-self.future_steps:,:,:]

        # EMBEDDING PAST VALUES
        # extract only past values (availables ones for forecasting) and embed them
        x_emb_past = self.x_linear(x_past)
        
        # COMPUTE APPROXIMATION OF FUTURE VALUES
        # using embedded past values use lstm to generate an approximation of actual future values, then embed them to respect hidden_size of the model 
        x_fut_approx = self.rnn(x_emb_past) # actual future_steps predictions that now will be improved
        
        import pdb
        pdb.set_trace()
        x_emb_fut_approx = self.x_linear(x_fut_approx.unsqueeze(2))

        # EMBEDDING APPROXIMATED FUTUTRE VALUES
        # tensor summaring past using categorical and past numerical vars
        past_emb = torch.cat((cat_emb_past, x_emb_past.unsqueeze(2)), dim=2)
        past_emb = torch.mean(past_emb, dim=2)
        past_emb_cat = torch.mean(cat_emb_past, dim=2)

        # tensor summaring future using categorical and approximated future numerical vars
        fut_emb = torch.cat((cat_emb_fut, x_emb_fut_approx.unsqueeze(2)), dim=2)
        fut_emb = torch.mean(fut_emb, dim=2)
        # fut_emb = fut_emb + x_emb_fut_approx # skip connection
        fut_emb_cat = torch.mean(cat_emb_fut, dim=2)

        # GRN on both past and fut
        past_grn = self.grn_past(past_emb)
        fut_grn = self.grn_fut(fut_emb)
        
        past_grn = past_grn + x_emb_past # skip connection
        fut_grn = fut_grn + x_emb_fut_approx # skip connection

        # Cross head computation of past and future post grn
        out = x_emb_fut_approx
        for cross_layer in self.cross_att_val_layers:
            cross_wei = fut_grn @ past_grn.transpose(-2,-1) * (self.d_model**-0.5) # (B, future_steps, hidden_size) @ (B, hidden_size, past_steps) -> (B, future_steps, past_steps)
            cross_wei = nn.functional.softmax(cross_wei, dim=-1)
            cross_wei = self.cross_dropout(cross_wei)
            cross_val = cross_wei @ past_emb_cat # (B, future_steps, past_steps) @ (B, past_steps, hidden_size) -> (B, future_steps, hidden_size)
            cross_val = cross_layer(cross_val)
            out = out + cross_val # skip connection
        
        for fut_layer in self.fut_att_val_layers:
            fut_wei = out @ fut_grn.transpose(-2,-1) * (self.d_model**-0.5)
            fut_wei = fut_wei.masked_fill(self.tril == 0, float('-inf'))
            fut_wei = nn.functional.softmax(fut_wei, dim=-1)
            fut_wei = self.fut_dropout(fut_wei)
            fut_val = fut_wei @ fut_emb_cat
            fut_val = fut_layer(fut_val)
            out = out + fut_val

        # last GRN on predicted values
        out_grn = self.grn_output(out)
        out_grn = self.out_dropout(out_grn)
        out_glu = self.out_glu(out_grn)
        out_glu = out_glu + fut_grn # skip connection

        # get actual values
        out = self.outLinear(out_glu)
        if self.mul>0:
            out = out.view(-1, self.future_steps, self.out_channels, self.mul)
        else:
            out = out.squeeze()

        return out