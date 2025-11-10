
from torch import  nn
import torch

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base
from .utils import QuantileLossMO,Permute, get_activation
from typing import List, Union
from ..data_structure.utils import beauty_string
import numpy as np
from .utils import  get_scope
from .utils import Embedding_cat_variables
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
                 sum_layers: bool,
                 hidden_RNN:int,
                 num_layers_RNN:int,
                 kind:str,
                 kernel_size:int,
                 dropout_rate:float=0.1,
                 use_bn:bool=False,
                 use_cumsum:bool=True,
                 use_bilinear:bool=False,
                 activation: str='torch.nn.ReLU',

                 **kwargs)->None:
        """Initialize the model with specified parameters.
        
        Args:
            sum_layers (bool): Flag indicating whether to sum layers in the encoder/decoder blocks.
            hidden_RNN (int): Number of hidden units in the RNN.
            num_layers_RNN (int): Number of layers in the RNN.
            kind (str): Type of RNN to use ('lstm' or 'gru').
            kernel_size (int): Size of the convolutional kernel.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.1.
            use_bn (bool, optional): Flag to use batch normalization. Defaults to False.
            use_cumsum (bool, optional): Flag to use cumulative sum. Defaults to True.
            use_bilinear (bool, optional): Flag to use bilinear layers. Defaults to False.
            activation (str, optional): Activation function to use. Defaults to 'torch.nn.ReLU'.
            **kwargs: Additional keyword arguments.
        
        Raises:
            ValueError: If the specified activation function is not recognized or if the kind is not 'lstm' or 'gru'.
        

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

        self.num_layers_RNN = num_layers_RNN
        self.hidden_RNN = hidden_RNN
        self.use_cumsum = use_cumsum
        self.kind = kind
        self.use_bilinear= use_bilinear

        
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels


   
    
        self.initial_linear_encoder =  nn.Sequential(Permute(),
                                                    nn.Conv1d(self.past_channels, (self.past_channels+hidden_RNN//4)//2, kernel_size, stride=1,padding='same'),
                                                    activation(),
                                                    nn.BatchNorm1d(  (self.past_channels+hidden_RNN//4)//2) if use_bn else nn.Dropout(dropout_rate) ,
                                                    nn.Conv1d( (self.past_channels+hidden_RNN//4)//2, hidden_RNN//4, kernel_size, stride=1,padding='same'),
                                                    Permute())

        self.initial_linear_decoder =   nn.Sequential(Permute(),
                                                    nn.Conv1d(self.future_channels, (self.future_channels+hidden_RNN//4)//2, kernel_size, stride=1,padding='same'),
                                                    activation(),
                                                    nn.BatchNorm1d(  (self.future_channels+hidden_RNN//4)//2) if use_bn else nn.Dropout(dropout_rate) ,
                                                    nn.Conv1d( (self.future_channels+hidden_RNN//4)//2, hidden_RNN//4, kernel_size, stride=1,padding='same'),
                                                    Permute())
        self.conv_encoder = Block(emb_past_out_channel+hidden_RNN//4,kernel_size,hidden_RNN//2,self.past_steps,sum_layers)
        

        if self.future_channels+emb_fut_out_channel==0:
            ## occhio che vuol dire che non ho passato , per ora ci metto una pezza e uso hidden dell'encoder
            self.conv_decoder = Block(hidden_RNN,kernel_size,hidden_RNN//2,self.future_steps,sum_layers) 
        else:
            self.conv_decoder = Block(self.future_channels+emb_fut_out_channel,kernel_size,hidden_RNN//2,self.future_steps,sum_layers) 
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
    def can_be_compiled(self):
        return True  


    def forward(self, batch):
        """It is mandatory to implement this method

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        x =  batch['x_num_past'].to(self.device)
        BS = x.shape[0]
        if 'x_cat_future' in batch.keys():
            emb_fut = self.emb_fut(BS,batch['x_cat_future'].to(self.device))
        else:
            emb_fut = self.emb_fut(BS,None)
        if 'x_cat_past' in batch.keys():
            emb_past = self.emb_past(BS,batch['x_cat_past'].to(self.device))
        else:
            emb_past = self.emb_past(BS,None)
            
        if 'x_num_future' in batch.keys():
            x_future = batch['x_num_future'].to(self.device)
        else:
            x_future = None        
            
 
        tmp = [self.initial_linear_encoder(x),emb_past]

        tot = torch.cat(tmp,2)

        out_past, hidden_past = self.Encoder(self.conv_encoder(tot))      
        
            
        ## hidden  = 2 x bs x channels_out_encoder
        ## out = BS x len x channels_out_encoder
        tmp = [emb_fut]
        
            
        if x_future is not None:
            tmp.append(x_future)
     
     
     
     
        if len(tmp)>0:
            tot = torch.cat(tmp,2)
            out_future, hidden_future = self.Decoder(self.conv_decoder(tot))  
        else:
            out_future, hidden_future = self.Decoder(self.conv_decoder(out_past))
            out_future = out_future[:,-1:,].repeat(1,self.future_steps,1) ##worakaround to check
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
                final = future.permute(0,2,1)+past.unsqueeze(2).repeat(1,1,self.future_steps)
            
        res= self.final_linear_decoder(final.permute(0,2,1)).reshape(BS,self.future_steps,self.out_channels,self.mul)
            

        
      
        return res

