
from torch import  nn
import torch
from .base import  Base
from .utils import get_device, QuantileLossMO,L1Loss
import math
from typing import List,Union


def generate_square_subsequent_mask(dim1: int, dim2: int):
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):
    """Copied from git
    """

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        return  self.pe[:,:x.size(1), :].repeat(x.shape[0],1,1)


class Attention(Base):              
    
    def __init__(self,
                 past_channels:int,
                 future_channels:int,
                 d_model:int,
                 num_heads:int,
                 past_steps:int,
                 future_steps:int,
                 dropout:float,
                 n_layer_encoder:int,
                 n_layer_decoder:int,
                 embs:List[int],
                 cat_emb_dim:int,
                 out_channels:int,
                 quantiles:List[int]=[],
                 n_classes:int=0,
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None)->None:
        """ Attention model. Using an encoder (past) decoder (future) with cross attention and masks. 
            helping classes (Categorical for instance).

        Args:
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int):  number of future numeric variables 
            d_model (int):  dimension of the attention model
            num_heads (int): heads equal in the encoder and encoder
            past_steps (int): number of past datapoints used 
            future_steps (int): number of future lag to predict
            dropout (float):  dropout used in the attention layers and positional encoder
            n_layer_encoder (int):  layers to use in the encoder
            n_layer_decoder (int):  layers to use in the decoder
            embs (List[int]): list of the initial dimension of the categorical variables
            cat_emb_dim (int): final dimension of each categorical variable
            out_channels (int): number of output channels
            quantiles (List[int], optional): we can use quantile loss il len(quantiles) = 0 (usually 0.1,0.5, 0.9) or L1loss in case len(quantiles)==0. Defaults to [].
            n_classes (int): number of classes (0 in regression)
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional):  configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
        """
        
        ##pytotch lightening stuff
        self.save_hyperparameters(logger=False)
        
        super().__init__()
        self.future_steps = future_steps
        assert (len(quantiles) ==0) or (len(quantiles)==3)

    
        if n_classes==0:
            self.is_classification = False
            if len(quantiles)>0:
                self.use_quantiles = True
                self.mul = len(quantiles)
                self.loss = QuantileLossMO(quantiles)
            else:
                self.use_quantiles = False
                self.mul = 1
                self.loss = L1Loss()
        else:
            self.is_classification = True
            self.use_quantiles = False
            self.mul = n_classes
            self.loss = torch.nn.CrossEntropyLoss()
            #assert out_channels==1, "Classification require only one channel"
                
                
            
        self.optim = optim
 
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config

        self.pe = PositionalEncoding( cat_emb_dim, dropout=dropout, max_len=past_steps+future_steps+1)
        self.emb_list = nn.ModuleList()
        if embs is not None:
            for k in embs:
                self.emb_list.append(nn.Embedding(k+1,cat_emb_dim))

        self.initial_layer_decoder = nn.Conv1d(in_channels=len(embs)*cat_emb_dim+cat_emb_dim+future_channels, out_channels= d_model, kernel_size=1, stride=1,padding='same')
        self.initial_layer_encoder = nn.Conv1d(in_channels=len(embs)*cat_emb_dim+cat_emb_dim+past_channels, out_channels= d_model, kernel_size=1, stride=1,padding='same')


        encoder_layer = nn.TransformerEncoderLayer( d_model=d_model, nhead=num_heads, dim_feedforward=d_model, dropout=dropout,batch_first=True ,norm_first=True)
        self.encoder = nn.TransformerEncoder( encoder_layer=encoder_layer, num_layers=n_layer_encoder, norm=None)
        decoder_layer = nn.TransformerDecoderLayer( d_model=d_model, nhead=num_heads, dim_feedforward=d_model, dropout=dropout,batch_first=True ,norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,num_layers=n_layer_decoder,norm=None)
        
 
        
        self.final_linear = nn.ModuleList()
        for _ in range(out_channels*self.mul):
            self.final_linear.append(nn.Sequential(nn.Linear(d_model,d_model*2),nn.ReLU(),nn.Dropout(0,2),
                                                   nn.Linear(d_model*2,d_model),nn.ReLU(),nn.Dropout(0,2),
                                                   nn.Linear(d_model,d_model//2),nn.ReLU(),nn.Dropout(0,2),
                                                   nn.Linear(d_model//2,1) ))

  


            
 
     
        
    def forward(self,batch:dict)->torch.tensor:
        """It is mandatory to implement this method

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
   
        x_past = batch['x_num_past'].to(self.device)

        tmp = [x_past,self.pe(x_past[:,:,0])]
        if 'x_cat_past' in batch.keys():
            x_cat_past = batch['x_cat_past'].to(self.device)
            for i in range(len(self.emb_list)):
                tmp.append(self.emb_list[i](x_cat_past[:,:,i]))

            
        x = torch.cat(tmp,2)
        ##BS x L x channels
        x = self.initial_layer_encoder( x.permute(0,2,1)).permute(0,2,1)       
        enc_past_steps = x.shape[1]

        src = self.encoder(x)

        ##decoder part
        if 'x_num_future' in batch.keys():
            x_future = batch['x_num_future'].to(self.device)
            tmp = [x_future,self.pe(x_future[:,:,0])]
            add_pe = False

        else:
            tmp = []
            add_pe = True
        if 'x_cat_future' in batch.keys():
            x_cat_future = batch['x_cat_future'].to(self.device)
            for i in range(len(self.emb_list)):
                tmp.append(self.emb_list[i](x_cat_future[:,:,i]))
            if add_pe:
                tmp.append(self.pe(x_cat_future[:,:,i]))
        if len(tmp)==0:
            SystemError('Please give me something for the future')
        y = torch.cat(tmp,2)
        
        y = self.initial_layer_decoder( y.permute(0,2,1)).permute(0,2,1)       
        forecast_window = y.shape[1]
        

      
        
        
        tgt_mask = generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=forecast_window
            ).to(self.device)

        src_mask = generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=enc_past_steps
            ).to(self.device)
        
        decoder_output = self.decoder(
            tgt=y,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            )

        res = []
        for j in range(len(self.final_linear)):
            res.append(self.final_linear[j](decoder_output))
       
        ##BxLxC
        res = torch.cat(res,2)
        B,L,_ = res.shape
        return res.reshape(B,L,-1,self.mul)
   
    

    
    def inference(self, batch:dict)->torch.tensor:
        """Care here, we need to implement it because for predicting the N-step it will use the prediction at step N-1. TODO fix if because I did not implement the
        know continuous variable presence here

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        tmp_x_past= batch.get('x_num_past',None)    
        tmp_cat_past= batch.get('x_cat_past',None)
        tmp_x_future= batch.get('x_num_future',None)    
        tmp_cat_future= batch.get('x_cat_future',None)
        
        tmp = {}
        if tmp_x_past is not None:
            tmp_x_past.to(self.device)
            tmp['x_num_past'] = tmp_x_past
        if tmp_cat_past is not None:
            tmp_cat_past.to(self.device)
            tmp['x_cat_past'] = tmp_cat_past
        if tmp_x_future is not None:
            tmp_x_future.to(self.device)
            tmp['x_num_future'] = tmp_x_future[:,0:1,:]
        if tmp_cat_future is not None:
            tmp_cat_future.to(self.device)
            tmp['x_cat_future'] = tmp_cat_future[:,0:1,:]
        ##TODO questo funziona solo senza meteo! 
        
        with torch.set_grad_enabled(False):
            y = []
            count = 0 
            for i in range(self.future_steps):
                y_i = self(tmp)
                ##quantile loss!
                if self.use_quantiles:
                    pred = y_i[:,-1:,:,1]
                else:
                    pred = y_i[:,-1:,:,0]
                if tmp_x_future is not None:
                    ##TODO questo funziona solo senza meteo a meno che non decida di mettere un ordine alle variabili
                    tmp['x_num_future'] =torch.cat([tmp['x_num_future'].to(self.device),pred],1)
                count+=1
                if tmp_cat_future is not None:
                    tmp['x_cat_future'] = tmp_cat_future[:,0:count+1,:]
                
                y.append(y_i[:,-1:,:,:])#.detach().cpu().numpy())
            
            return torch.cat(y,1)     