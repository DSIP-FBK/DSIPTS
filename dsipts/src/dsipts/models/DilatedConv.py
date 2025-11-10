
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
torch.autograd.set_detect_anomaly(True)
from .utils import  get_scope
from .utils import Embedding_cat_variables

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
        
        

class DilatedConv(Base):
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
                 activation: str='torch.nn.ReLU',
                 remove_last = False,
                 dropout_rate:float=0.1,
                 use_bn:bool=False,
                 use_glu:bool=True,
                 glu_percentage: float=1.0,
                 
                 **kwargs)->None:
        """Custom encoder-decoder 
        
        Args:
            sum_layers (bool): Flag indicating whether to sum the layers.
            hidden_RNN (int): Number of hidden units in the RNN.
            num_layers_RNN (int): Number of layers in the RNN.
            kind (str): Type of RNN to use (e.g., 'LSTM', 'GRU').
            kernel_size (int): Size of the convolutional kernel.
            activation (str, optional): Activation function to use. Defaults to 'torch.nn.ReLU'.
            remove_last (bool, optional): Flag to indicate whether to remove the last element in the sequence. Defaults to False.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.1.
            use_bn (bool, optional): Flag to indicate whether to use batch normalization. Defaults to False.
            use_glu (bool, optional): Flag to indicate whether to use Gated Linear Units (GLU). Defaults to True.
            glu_percentage (float, optional): Percentage of GLU to apply. Defaults to 1.0.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None
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
        self.kind = kind
        self.use_glu = use_glu
        self.glu_percentage = torch.tensor(glu_percentage).to(self.device)
        self.remove_last = remove_last
                
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels

        if self.use_glu:
            self.past_glu = nn.ModuleList()
            self.future_glu = nn.ModuleList()
            for _ in range(self.past_channels):
                self.past_glu.append(GLU(1))
            
            for _ in range(self.future_channels):
                self.future_glu.append(GLU(1))
    
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
        
        #nn.Sequential(Permute(), nn.Conv1d(emb_channels+hidden_RNN//8, hidden_RNN//8, kernel_size, stride=1,padding='same'),Permute(),nn.Dropout(0.3))

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
        self.final_linear = nn.ModuleList()
        for _ in range(self.out_channels*self.mul):
            self.final_linear.append(nn.Sequential(nn.Linear(hidden_RNN+emb_fut_out_channel+self.future_channels,hidden_RNN*2), 
                                            activation(),
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.BatchNorm1d(hidden_RNN*2) if use_bn else nn.Dropout(dropout_rate) ,
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.Linear(hidden_RNN*2,hidden_RNN),
                                            activation(),
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.BatchNorm1d(hidden_RNN) if use_bn else nn.Dropout(dropout_rate) ,
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.Linear(hidden_RNN,hidden_RNN//2),
                                            activation(),
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.BatchNorm1d(hidden_RNN//2) if use_bn else nn.Dropout(dropout_rate) ,
                                            Permute() if use_bn else nn.Identity() ,
                                            nn.Linear(hidden_RNN//2,hidden_RNN//4),
                                            activation(),
                                            nn.Linear(hidden_RNN//4,1)))
        
        self.return_additional_loss = True
        

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
            xf = torch.clone(x_future)
        else:
            x_future = None     
            
        if self.remove_last:
            idx_target = batch['idx_target'][0]

            x_start = x[:,-1,idx_target].unsqueeze(1)
            ##BxC
            x[:,:,idx_target]-=x_start        
            
    
        ## first GLU
        score = 0
        xp =  torch.clone(x)
        
        if self.use_glu:
            score_past_tot = 0
            score_future_tot = 0
            
            for i in range(len(self.past_glu)):
                x[:,:,i],score = self.past_glu[i](xp[:,:,i])
                score_past_tot+=score
            score_past_tot/=len(self.past_glu)
            
            if x_future is not None:
                for i in range(len(self.future_glu)):
                    x_future[:,:,i],score = self.future_glu[i](xf[:,:,i])
                    score_future_tot+=score
                score_future_tot/=len(self.future_glu)
            score = 0.5*(score_past_tot+score_future_tot)
        tmp = [self.initial_linear_encoder(x),emb_past]
        
  

        tot = torch.cat(tmp,2)
        out, hidden = self.Encoder(self.conv_encoder(tot))      
        tmp = [emb_fut]
        if x_future is not None:
            tmp.append(x_future)
        tot = torch.cat(tmp,2)
        out, _ = self.Decoder(self.conv_decoder(tot),hidden)  
        res = []
        tmp = torch.cat([tot,out],axis=2)


        for j in range(self.out_channels*self.mul):
            res.append(self.final_linear[j](tmp))

        res = torch.cat(res,2)
        ##BxLxC
        B = res.shape[0]
        res = res.reshape(B,self.future_steps,-1,self.mul)
        if self.remove_last:
            res+=x_start.unsqueeze(1)

      
        return res, score

    def inference(self, batch:dict)->torch.tensor:

        res, score = self(batch)
        return res