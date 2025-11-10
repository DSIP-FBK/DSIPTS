
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
from .utils import QuantileLossMO,Permute,get_activation
from typing import List,Union
from ..data_structure.utils import beauty_string
from .utils import  get_scope
from .xlstm.xLSTM import xLSTM
from .utils import Embedding_cat_variables
torch.autograd.set_detect_anomaly(True)

class MyBN(nn.Module):
    def __init__(self,channels):
        super(MyBN, self).__init__()
        self.BN = nn.BatchNorm1d(channels)
    def forward(self,x):
        return self.BN(x.permute(0,2,1)).permute(0,2,1)

class RNN(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    
    
    
    def __init__(self, 
                
                 hidden_RNN:int,
                 num_layers_RNN:int,
                 kind:str,
                 kernel_size:int,
                 activation:str='torch.nn.ReLU',
                 remove_last = False,
                 dropout_rate:float=0.1,
                 use_bn:bool=False,
                 num_blocks:int=4, 
                 bidirectional:bool=True,
                 lstm_type:str='slstm',
                
                 **kwargs)->None:
        """Initialize a recurrent model with an encoder-decoder structure.
        
        Args:
            hidden_RNN (int): Hidden size of the RNN block.
            num_layers_RNN (int): Number of RNN layers.
            kind (str): Type of RNN to use, either 'gru' or 'lstm' or `xlstm`.
            kernel_size (int): Kernel size in the encoder convolutional block.
            activation (str, optional): Activation function from PyTorch. Default is 'torch.nn.ReLU'.
            remove_last (bool, optional): If True, the model learns the difference with respect to the last seen point. Default is False.
            dropout_rate (float, optional): Dropout rate in Dropout layers. Default is 0.1.
            use_bn (bool, optional): If True, Batch Normalization layers will be added and Dropouts will be removed. Default is False.
            num_blocks (int, optional): Number of xLSTM blocks (only for xLSTM). Default is 4.
            bidirectional (bool, optional): If True, the RNN is bidirectional. Default is True.
            lstm_type (str, optional): Type of LSTM to use (only for xLSTM), either 'slstm' or 'mlstm'. Default is 'slstm'.
            **kwargs: Additional keyword arguments.
        
        
        Raises:
            ValueError: If the specified kind is not 'lstm', 'gru', or 'xlstm'.
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

        self.num_layers_RNN = num_layers_RNN
        self.hidden_RNN = hidden_RNN

        self.kind = kind
        self.remove_last = remove_last
        
        
        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        emb_fut_out_channel = self.emb_fut.output_channels

        


     
        self.initial_linear_encoder =  nn.Sequential(nn.Linear(self.past_channels,4),
                                                     activation(),
                                                     
                                                    MyBN(4) if use_bn else nn.Dropout(dropout_rate) ,
                                                     nn.Linear(4,8),
                                                     activation(),
                                                    MyBN(8) if use_bn else nn.Dropout(dropout_rate) ,
                                                     nn.Linear(8,hidden_RNN//8))
        self.initial_linear_decoder =  nn.Sequential(nn.Linear(self.future_channels,4),
                                                     activation(),
                                                     MyBN(4) if use_bn else nn.Dropout(dropout_rate) ,
                                                     nn.Linear(4,8),
                                                     activation(),
                                                     MyBN(8) if use_bn else nn.Dropout(dropout_rate) ,
                                                     nn.Linear(8,hidden_RNN//8))
        
        
        self.conv_encoder = nn.Sequential(Permute(), nn.Conv1d(emb_past_out_channel+hidden_RNN//8, hidden_RNN//8, kernel_size, stride=1,padding='same'),Permute(),nn.Dropout(0.3))
        
        if self.future_channels+emb_fut_out_channel==0:
            ## occhio che vuol dire che non ho futuro , per ora ci metto una pezza e uso hidden dell'encoder
            self.conv_decoder =  nn.Sequential(Permute(),nn.Conv1d(hidden_RNN, hidden_RNN//8, kernel_size=kernel_size, stride=1,padding='same'),   Permute())
        else:
            self.conv_decoder =  nn.Sequential(Permute(),nn.Conv1d(self.future_channels+emb_fut_out_channel, hidden_RNN//8, kernel_size=kernel_size, stride=1,padding='same'),   Permute())
            
            
        if self.kind=='lstm':
            self.Encoder = nn.LSTM(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
            self.Decoder = nn.LSTM(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
        elif self.kind=='gru':
            self.Encoder = nn.GRU(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
            self.Decoder = nn.GRU(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,batch_first=True)
        elif self.kind=='xlstm':
            self.Encoder = xLSTM(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,num_blocks=num_blocks,dropout=dropout_rate, bidirectional=bidirectional, lstm_type=lstm_type)
            self.Decoder = xLSTM(input_size= hidden_RNN//8,hidden_size=hidden_RNN,num_layers = num_layers_RNN,num_blocks=num_blocks,dropout=dropout_rate, bidirectional=bidirectional, lstm_type=lstm_type)
      
        else:
            beauty_string('Speciky kind= lstm or gru please','section',True)
        self.final_linear = nn.ModuleList()
        for _ in range(self.out_channels*self.mul):
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
    def can_be_compiled(self):
        return True  
  

    def forward(self, batch):
  
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
        
        if self.remove_last:
            idx_target = batch['idx_target'][0]

            x_start = x[:,-1,idx_target].unsqueeze(1)
            ##BxC
            x[:,:,idx_target]-=x_start        
        
        tmp = [self.initial_linear_encoder(x),emb_past]
        
        
        tot = torch.cat(tmp,2)

        out, hidden = self.Encoder(self.conv_encoder(tot))      

        tmp = [emb_fut]
                   
        if x_future is not None:
            tmp.append(x_future)
            
        if len(tmp)>0:
            tot = torch.cat(tmp,2)
        else:
            tot = out
        out, _ = self.Decoder(self.conv_decoder(tot[:,-1:,:].repeat(1,self.future_steps,1)),hidden)  
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

    