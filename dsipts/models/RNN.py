
from torch import  nn
import torch
from .base import Base
from .utils import QuantileLossMO,Permute, get_device,L1Loss,get_activation
from typing import List,Union
import logging


class MyBN(nn.Module):
    def __init__(self,channels):
        self.BN = nn.BatchNorm1d(channels)
    def forward(self,x):
        return self.BN(x.permute(0,2,1)).permute(0,2,1)

class RNN(Base):

    
    def __init__(self, 
                 past_steps:int,
                 future_steps:int,
                 past_channels:int,
                 future_channels:int,
                 embs:List[int],
                 cat_emb_dim:int,
                 hidden_RNN:int,
                 num_layers_RNN:int,
                 kind:str,
                 kernel_size:int,
                 sum_emb:bool,
                 out_channels:int,
                 activation:str='torch.nn.ReLU',
                 dropout_rate:float=0.1,
                 use_bn:bool=False,
                 quantiles:List[int]=[],
                  n_classes:int=0,
                  optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None)->None:
        """ Recurrent model with an encoder decoder structure

        Args:
            past_steps (int):  number of past datapoints used 
            future_steps (int): number of future lag to predict
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            embs (List): list of the initial dimension of the categorical variables
            cat_emb_dim (int): final dimension of each categorical variable
            hidden_RNN (int): hidden size of the RNN block
            num_layers_RNN (int): number of RNN layers
            kind (str): one among GRU or LSTM
            kernel_size (int): kernel size in the encoder convolutional block
            sum_emb (bool): if true the contribution of each embedding will be summed-up otherwise stacked
            out_channels (int):  number of output channels
            activation (str, optional): activation fuction function pytorch. Default torch.nn.ReLU
            dropout_rate (float, optional): dropout rate in Dropout layers
            use_bn (bool, optional): if true BN layers will be added and dropouts will be removed
            quantiles (List[int], optional): we can use quantile loss il len(quantiles) = 0 (usually 0.1,0.5, 0.9) or L1loss in case len(quantiles)==0. Defaults to [].
            n_classes (int): number of classes (0 in regression)
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
        """
        
        if activation == 'torch.nn.SELU':
            logging.info('SELU do not require BN')
            use_bn = False
        if type(activation)==str:
            activation = get_activation(activation)
        else:
            logging.info('There is a bug in pytorch lightening, the constructior is called twice ')
        
        super(RNN, self).__init__()
        self.save_hyperparameters(logger=False)
        #self.device = get_device()
        self.past_steps = past_steps
        self.future_steps = future_steps

        self.num_layers_RNN = num_layers_RNN
        self.hidden_RNN = hidden_RNN
        self.past_channels = past_channels 
        self.future_channels = future_channels 
        self.embs = nn.ModuleList()
        self.sum_emb = sum_emb
        self.kind = kind
        
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
        
        emb_channels = 0
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config

        for k in embs:
            self.embs.append(nn.Embedding(k+1,cat_emb_dim))
            emb_channels+=cat_emb_dim
            
            
        if sum_emb and (emb_channels>0):
            emb_channels = cat_emb_dim
            logging.info('Using sum')
        else:
            logging.info('Using stacked')


        self.initial_linear_encoder =  nn.Sequential(nn.Linear(past_channels,4),
                                                     activation(),
                                                     
                                                    MyBN(4) if use_bn else nn.Dropout(dropout_rate) ,
                                                     nn.Linear(4,8),
                                                     activation(),
                                                    MyBN(8) if use_bn else nn.Dropout(dropout_rate) ,
                                                     nn.Linear(8,hidden_RNN//8))
        self.initial_linear_decoder =  nn.Sequential(nn.Linear(future_channels,4),
                                                     activation(),
                                                     MyBN(4) if use_bn else nn.Dropout(dropout_rate) ,
                                                     nn.Linear(4,8),
                                                     activation(),
                                                     MyBN(8) if use_bn else nn.Dropout(dropout_rate) ,
                                                     nn.Linear(8,hidden_RNN//8))
        
        
        self.conv_encoder = nn.Sequential(Permute(), nn.Conv1d(emb_channels+hidden_RNN//8, hidden_RNN//8, kernel_size, stride=1,padding='same'),Permute(),nn.Dropout(0.3))
        
        if future_channels+emb_channels==0:
            ## occhio che vuol dire che non ho futuro , per ora ci metto una pezza e uso hidden dell'encoder
            self.conv_decoder =  nn.Sequential(Permute(),nn.Conv1d(hidden_RNN, hidden_RNN//8, kernel_size=kernel_size, stride=1,padding='same'),   Permute())
        else:
            self.conv_decoder =  nn.Sequential(Permute(),nn.Conv1d(future_channels+emb_channels, hidden_RNN//8, kernel_size=kernel_size, stride=1,padding='same'),   Permute())
            
            
        if self.kind=='lstm':
            self.Encoder = nn.LSTM(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
            self.Decoder = nn.LSTM(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
        elif self.kind=='gru':
            self.Encoder = nn.GRU(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
            self.Decoder = nn.GRU(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
        else:
            logging.error('Speciky kind= lstm or gru please')
        self.final_linear = nn.ModuleList()
        for _ in range(out_channels*self.mul):
            self.final_linear.append(nn.Sequential(nn.Linear(hidden_RNN,hidden_RNN//2), 
                                            activation(),
                                            MyBN(hidden_RNN//2) if use_bn else nn.Dropout(dropout_rate) ,
                                            nn.Linear(hidden_RNN//2,hidden_RNN//4),
                                            activation(),
                                            MyBN(hidden_RNN//4) if use_bn else nn.Dropout(dropout_rate) ,
                                            nn.Linear(hidden_RNN//4,hidden_RNN//8),
                                            activation(),
                                            MyBN(hidden_RNN//8) if use_bn else nn.Dropout(dropout_rate) ,
                                            nn.Linear(hidden_RNN//8,1)))

  

    def forward(self, batch):
        """It is mandatory to implement this method

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        x =  batch['x_num_past'].to(self.device)

        if 'x_cat_future' in batch.keys():
            cat_future = batch['x_cat_future'].to(self.device)
        if 'x_cat_past' in batch.keys():
            cat_past = batch['x_cat_past'].to(self.device)
        if 'x_num_future' in batch.keys():
            x_future = batch['x_num_future'].to(self.device)
        else:
            x_future = None
        import pdb
        pdb.set_trace()
        tmp = [self.initial_linear_encoder(x)]
        
        for i in range(len(self.embs)):
            if self.sum_emb:
                if i>0:
                    tmp_emb+=self.embs[i](cat_past[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_past[:,:,i])
            else:
                tmp.append(self.embs[i](cat_past[:,:,i]))
        if self.sum_emb and (len(self.embs)>0):
            tmp.append(tmp_emb)
        tot = torch.cat(tmp,2)
            
        out, hidden = self.Encoder(self.conv_encoder(tot))      

        tmp = []
        for i in range(len(self.embs)):
            if self.sum_emb:
                if i>0:
                    tmp_emb+=self.embs[i](cat_future[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_future[:,:,i])
            else:
                tmp.append(self.embs[i](cat_future[:,:,i]))   
        if self.sum_emb and (len(self.embs)):
            tmp.append(tmp_emb)
            
        if x_future is not None:
            tmp.append(x_future)
            

        if len(tmp)>0:
            tot = torch.cat(tmp,2)
        else:
            tot = out
        out, _ = self.Decoder(self.conv_decoder(tot),hidden)  
        #else:
        #    out, _ = self.Decoder(self.conv_decoder(out),hidden)  
        res = []

      
        for j in range(len(self.final_linear)):
            res.append(self.final_linear[j](out))
            
        res = torch.cat(res,2)
        ##BxLxC
        B,L,_ = res.shape
        
      
        return res.reshape(B,L,-1,self.mul)

    