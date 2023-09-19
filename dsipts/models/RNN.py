
from torch import  nn
import torch
from .base import Base
from .utils import QuantileLossMO,Permute,get_activation
from typing import List,Union
from ..data_structure.utils import beauty_string


class MyBN(nn.Module):
    def __init__(self,channels):
        super(MyBN, self).__init__()
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
                 remove_last = False,
                 dropout_rate:float=0.1,
                 use_bn:bool=False,
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[int]=[],
                 n_classes:int=0,
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None,
                 **kwargs)->None:
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
            remove_last (bool, optional): if True the model learns the difference respect to the last seen point
            dropout_rate (float, optional): dropout rate in Dropout layers
            use_bn (bool, optional): if true BN layers will be added and dropouts will be removed
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            quantiles (List[int], optional): we can use quantile loss il len(quantiles) = 0 (usually 0.1,0.5, 0.9) or L1loss in case len(quantiles)==0. Defaults to [].
            n_classes (int): number of classes (0 in regression)
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
        """
        super().__init__(**kwargs)

        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
            use_bn = False
        if isinstance(activation, str):
            activation = get_activation(activation)
        else:
            beauty_string('There is a bug in pytorch lightening, the constructior is called twice ','info',self.verbose)
        
        self.save_hyperparameters(logger=False)
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.persistence_weight = persistence_weight 
        self.loss_type = loss_type
        self.num_layers_RNN = num_layers_RNN
        self.hidden_RNN = hidden_RNN
        self.past_channels = past_channels 
        self.future_channels = future_channels 
        self.embs = nn.ModuleList()
        self.sum_emb = sum_emb
        self.kind = kind
        self.remove_last = remove_last
        if n_classes==0:
            self.is_classification = False
            if len(quantiles)>0:
                self.use_quantiles = True
                self.mul = len(quantiles)
                self.loss = QuantileLossMO(quantiles)
            else:
                self.use_quantiles = False
                self.mul = 1
                if self.loss_type == 'mse':
                    self.loss = nn.MSELoss()
                else:
                    self.loss = nn.L1Loss()
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
            beauty_string('Using sum','info',self.verbose)
        else:
            beauty_string('Using stacked','info',self.verbose)

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
            beauty_string('Speciky kind= lstm or gru please','section',True)
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
        
        if self.remove_last:
            idx_target = batch['idx_target'][0]

            x_start = x[:,-1,idx_target].unsqueeze(1)
            ##BxC
            x[:,:,idx_target]-=x_start        
        
        tmp = [self.initial_linear_encoder(x)]
        
        tmp_emb = None
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
        res = []

      
        for j in range(len(self.final_linear)):
            res.append(self.final_linear[j](out))
            
        res = torch.cat(res,2)
        ##BxLxC
        B,L,_ = res.shape
        res = res.reshape(B,L,-1,self.mul)
        
        if self.remove_last:
            res+=x_start.unsqueeze(1)
      
        return res

    