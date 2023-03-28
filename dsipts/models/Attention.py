
from torch import  nn
import torch
from .base import  Base
from .utils import get_device, QuantileLossMO
import math



def generate_square_subsequent_mask(dim1: int, dim2: int):
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

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


    def __init__(self, past_channels,future_channels,d_model, num_heads,past_steps,future_steps,dropout,n_layer_encoder,n_layer_decoder,embs,cat_emb_dim,out_channels,quantiles=[],optim_config=None,scheduler_config=None):
        self.save_hyperparameters(logger=False)
        # d_model: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.future_steps = future_steps
        assert (len(quantiles) ==0) or (len(quantiles)==3)
        if len(quantiles)>0:
            self.use_quantiles = True
            self.mul = 3 
        else:
            self.use_quantiles = False
            self.mul = 1
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
                                                   nn.Linear(d_model//2,1)))

  
        if  self.use_quantiles:
            self.loss = QuantileLossMO(quantiles)
        else:
            self.loss = nn.L1Loss()
        
 
     
        
    def forward(self,batch):
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
        else:
            tmp = []
        if 'x_cat_future' in batch.keys():
            x_cat_future = batch['x_cat_future'].to(self.device)
            for i in range(len(self.emb_list)):
                tmp.append(self.emb_list[i](x_cat_future[:,:,i]))
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

    
    
    
    def inference(self, batch):
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