
from torch import  nn
import torch
from .base import Base
from .utils import QuantileLossMO,Permute, get_activation
from typing import List, Union
from ..data_structure.utils import beauty_string
import numpy as np
from .utils import  get_scope

torch.autograd.set_detect_anomaly(True)

class GLU(nn.Module):
    def __init__(self, d_model: int):
        """Gated Linear Unit, 'Gate' block in TFT paper 
        Sub net of GRN: linear(x) * sigmoid(linear(x))
        No dimension changes

        Args:
            d_model (int): model dimension
        """
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated Linear Unit
        Sub net of GRN: linear(x) * sigmoid(linear(x))
        No dimension changes: [bs, seq_len, d_model]

        Args:
            x (torch.Tensor)

        Returns:
            torch.Tensor
        """

        ##here comes something like BSxL
        x1 = (self.activation(self.linear(x.unsqueeze(2)))/6.0).squeeze()
        out = x1*x #element-wise multiplication
        
        ##get the score
        score = torch.sign(x1).mean()
        return out,score

class Block(nn.Module):
    def __init__(self,input_channels:int,kernel_size:int,output_channels:int,input_size:int,sum_layers:bool ):
    
    
        super(Block, self).__init__()

        self.dilations = nn.ModuleList()
        self.steps = int(np.floor(np.log2(input_size)))-1

        if self.steps <=1:
            self.steps = 1
       
        for i in range(self.steps):
            #dilation
            self.dilations.append(nn.Conv1d(input_channels, output_channels, kernel_size, stride=1,padding='same',dilation=2**i))
            s = max(2**i-1,1)
            k = 2**(i+1)+1
            p = int(((s-1)*input_size + k - 1)/2)
            self.dilations.append(nn.Conv1d(input_channels, output_channels, k, stride=s,padding=p))
      

            
            
        self.sum_layers = sum_layers
        mul = 1 if sum_layers else self.steps*2 
        self.conv_final = nn.Conv1d(output_channels*mul, output_channels*mul, kernel_size, stride=1,padding='same')
        self.out_channels = output_channels*mul
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = Permute()(x)
        tmp = []
        for i in range(self.steps):

            tmp.append(self.dilations[i](x))

        if self.sum_layers:
            tmp = torch.stack(tmp)
            tmp = tmp.sum(axis=0)
        else:
            tmp = torch.cat(tmp,1)
        
        return Permute()(tmp)
        
        

class DilatedConvED(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
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
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[int]=[],
                 dropout_rate:float=0.1,
                 use_bn:bool=False,
                 use_cumsum:bool=True,
                 use_bilinear:bool=False,
                 n_classes:int=0,
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None,
                 **kwargs)->None:
        """ Custom encoder-decoder 
        
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
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            quantiles (List[int], optional): we can use quantile loss il len(quantiles) = 0 (usually 0.1,0.5, 0.9) or L1loss in case len(quantiles)==0. Defaults to [].
            dropout_rate (float, optional): dropout rate in Dropout layers
            use_bn (bool, optional): if true BN layers will be added and dropouts will be removed
            use_cumsum (bool, optional): if true use cumsum of future representation else it uses the same covariate represnetation for each future step
            n_classes (int): number of classes (0 in regression)
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.

        """
        super().__init__(**kwargs)
        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
            use_bn = False
        if isinstance(activation,str):
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
        self.use_cumsum = use_cumsum
        self.kind = kind
        self.out_channels = out_channels
        self.use_bilinear= use_bilinear
        #import pdb
        #pdb.set_trace()
        if n_classes==0:
            self.is_classification = False
            if len(quantiles)>0:
                assert len(quantiles)==3, beauty_string('ONLY 3 quantiles premitted','info',True)
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
    

   
    
        self.initial_linear_encoder =  nn.Sequential(Permute(),
                                                    nn.Conv1d(past_channels, (past_channels+hidden_RNN//4)//2, kernel_size, stride=1,padding='same'),
                                                    activation(),
                                                    nn.BatchNorm1d(  (past_channels+hidden_RNN//4)//2) if use_bn else nn.Dropout(dropout_rate) ,
                                                    nn.Conv1d( (past_channels+hidden_RNN//4)//2, hidden_RNN//4, kernel_size, stride=1,padding='same'),
                                                    Permute())

        self.initial_linear_decoder =   nn.Sequential(Permute(),
                                                    nn.Conv1d(future_channels, (future_channels+hidden_RNN//4)//2, kernel_size, stride=1,padding='same'),
                                                    activation(),
                                                    nn.BatchNorm1d(  (future_channels+hidden_RNN//4)//2) if use_bn else nn.Dropout(dropout_rate) ,
                                                    nn.Conv1d( (future_channels+hidden_RNN//4)//2, hidden_RNN//4, kernel_size, stride=1,padding='same'),
                                                    Permute())
        self.conv_encoder = Block(emb_channels+hidden_RNN//4,kernel_size,hidden_RNN//2,self.past_steps,sum_emb)
        

        if future_channels+emb_channels==0:
            ## occhio che vuol dire che non ho passato , per ora ci metto una pezza e uso hidden dell'encoder
            self.conv_decoder = Block(hidden_RNN,kernel_size,hidden_RNN//2,self.future_steps,sum_emb) 
        else:
            self.conv_decoder = Block(future_channels+emb_channels,kernel_size,hidden_RNN//2,self.future_steps,sum_emb) 
            #nn.Sequential(Permute(),nn.Linear(past_steps,past_steps*2),  nn.PReLU(),nn.Dropout(0.2),nn.Linear(past_steps*2, future_steps),nn.Dropout(0.3),nn.Conv1d(hidden_RNN, hidden_RNN//8, 3, stride=1,padding='same'),   Permute())
        if self.kind=='lstm':
            self.Encoder = nn.LSTM(input_size= self.conv_encoder.out_channels,#, hidden_RNN//4,
                                   hidden_size=hidden_RNN//2,
                                   num_layers = num_layers_RNN,
                                   batch_first=True,bidirectional=True)
            self.Decoder = nn.LSTM(input_size= self.conv_decoder.out_channels,#, hidden_RNN//4,
                                   hidden_size=hidden_RNN//2,
                                   num_layers = num_layers_RNN,
                                   batch_first=True,bidirectional=True)
        elif self.kind=='gru':
            self.Encoder = nn.GRU(input_size=self.conv_encoder.out_channels,#, hidden_RNN//4,
                                  hidden_size=hidden_RNN//2,
                                  num_layers = num_layers_RNN,
                                  batch_first=True,bidirectional=True)
            self.Decoder = nn.GRU(input_size= self.conv_decoder.out_channels,#, hidden_RNN//4,
                                  hidden_size=hidden_RNN//2,
                                  num_layers = num_layers_RNN,
                                  batch_first=True,bidirectional=True)
        else:
            beauty_string('Specify kind lstm or gru please','section',True)
        self.final_linear_decoder = nn.Sequential(nn.Linear((hidden_RNN//2*2)*num_layers_RNN ,hidden_RNN*2), 
                                            activation(),
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.BatchNorm1d(hidden_RNN*2) if use_bn else nn.Dropout(dropout_rate) ,
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.Linear(hidden_RNN*2,hidden_RNN),
                                            activation(),
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.BatchNorm1d(hidden_RNN) if use_bn else nn.Dropout(dropout_rate) ,
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.Linear(hidden_RNN,self.mul))
        
        if use_bilinear:
            self.bilinear = torch.nn.Bilinear((hidden_RNN//2*2)*num_layers_RNN,(hidden_RNN//2*2)*num_layers_RNN,hidden_RNN*2)
            self.final_linear_decoder = nn.Sequential(
                                                activation(),
                                                Permute() if use_bn else nn.Identity() ,
                                                nn.BatchNorm1d(hidden_RNN*2) if use_bn else nn.Dropout(dropout_rate) ,
                                                Permute() if use_bn else nn.Identity() ,
                                                nn.Linear(hidden_RNN*2,hidden_RNN),
                                                activation(),
                                                Permute() if use_bn else nn.Identity() ,
                                                nn.BatchNorm1d(hidden_RNN) if use_bn else nn.Dropout(dropout_rate) ,
                                                Permute() if use_bn else nn.Identity() ,
                                                nn.Linear(hidden_RNN ,self.mul))
        


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
            
 
        tmp = [self.initial_linear_encoder(x)]
        
        if 'x_cat_past' in batch.keys():
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

        out_past, hidden_past = self.Encoder(self.conv_encoder(tot))      
        
            
        ## hidden  = 2 x bs x channels_out_encoder
        ## out = BS x len x channels_out_encoder
        tmp = []
        for i in range(len(self.embs)):
            if self.sum_emb:
                if i>0:
                    tmp_emb+=self.embs[i](cat_future[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_future[:,:,i])
            else:
                tmp.append(self.embs[i](cat_future[:,:,i]))   
        if self.sum_emb and (len(self.embs)>0):
            tmp.append(tmp_emb)
            
        if x_future is not None:
            tmp.append(x_future)
     
     
     
     
        if len(tmp)>0:
            tot = torch.cat(tmp,2)
            out_future, hidden_future = self.Decoder(self.conv_decoder(tot))  
        else:
            out_future, hidden_future = self.Decoder(self.conv_decoder(out_past))  
        ##hidden state of the past --> initial state

        if self.kind=='lstm':
            hidden_past = hidden_past[0] 
            
        #past= 2num_layers_RNNxBSxhidden_RNN//2
        # furture = BSx L x    hidden_RNN//2 --> BSxLxC
        BS = hidden_past.shape[1]
        N = hidden_past.shape[0]//2
        past = hidden_past.permute(1,0,2).reshape(BS,-1) #BSx2NxC --> BSx2CN
        future = out_future.repeat(1,1,N)

        if self.use_bilinear:
            final = self.bilinear(future,past.unsqueeze(2).repeat(1,1,self.future_steps).permute(0,2,1)).permute(0,2,1)
        else:
            if self.use_cumsum:
                final = torch.cumsum(future,axis=1).permute(0,2,1)+past.unsqueeze(2).repeat(1,1,self.future_steps)
            else:
                import pdb
                pdb.set_trace()
                final = future.permute(0,2,1)+past.unsqueeze(2).repeat(1,1,self.future_steps)
            
        res= self.final_linear_decoder(final.permute(0,2,1)).reshape(BS,self.future_steps,self.out_channels,self.mul)
            

        
      
        return res

