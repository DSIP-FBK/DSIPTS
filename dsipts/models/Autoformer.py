
from torch import  nn
import torch
from .base import Base
from typing import List,Union
from ..data_structure.utils import beauty_string
from .utils import  get_activation
from .autoformer.layers import AutoCorrelation, AutoCorrelationLayer, Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp,PositionalEmbedding


  
class Autoformer(Base):

    
    def __init__(self, 
                 past_steps:int,
                 future_steps:int,
                 label_len: int,
                 past_channels:int,
                 future_channels:int,
                 out_channels:int,
                 d_model:int,
                 embs:List[int],
                 kernel_size:int,
                 activation:str='torch.nn.ReLU',
                 factor: int=5,
                 n_head:int=1,
                 n_layer_encoder:int=2,
                 n_layer_decoder:int=2,
                 hidden_size:int=1048,
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[int]=[],
                 dropout_rate:float=0.1,
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None)->None:
        """

        Args:
            past_steps (int): number of past datapoints used , not used here
            future_steps (int): number of future lag to predict
            label_len (int): overlap len
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            out_channels (int):  number of output channels
            d_model (int):  dimension of the attention model
            embs (List): list of the initial dimension of the categorical variables
            embed_type (int): type of embedding
            kernel_size (int): kernel_size
            activation (str, optional): activation fuction function pytorch. Default torch.nn.ReLU
            n_head (int, optional): number of heads
            n_layer_encoder (int, optional): number of encoding layers
            n_layer_decoder (int, optional): number of decoding layers
            factor (int): num of routers in Cross-Dimension Stage of TSA (c) see the paper            
            out_channels (int):  number of output channels
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            quantiles (List[int], optional): NOT USED YET
            dropout_rate (float, optional):  dropout rate in Dropout layers. Defaults to 0.1.
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
        """
        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info')
            use_bn = False
        if type(activation)==str:
            activation = get_activation(activation)
        else:
            beauty_string('There is a bug in pytorch lightening, the constructior is called twice ','info')
        
   
        super(Autoformer, self).__init__()
        self.save_hyperparameters(logger=False)
        self.use_quantiles = False
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.loss_type = loss_type
        self.persistence_weight = persistence_weight 

        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

        
        self.seq_len = past_steps
        self.label_len = label_len
        self.pred_len = future_steps

        # Decomp
        self.decomp = series_decomp(kernel_size)

        self.embs = nn.ModuleList()
        emb_channels = 0
        for k in embs:
            self.embs.append(nn.Embedding(k+1,d_model))
            emb_channels = d_model
            
        #past_channels+=emb_channels
        #future_channels+=emb_channels
        
        self.linear_encoder = nn.Sequential(nn.Linear(past_channels,past_channels*2),activation() ,nn.Dropout(dropout_rate),nn.Linear(past_channels*2,d_model*2),activation() ,nn.Dropout(dropout_rate),nn.Linear(d_model*2,d_model))
        
        self.linear_decoder = nn.Sequential(nn.Linear(future_channels,future_channels*2),activation() ,nn.Dropout(dropout_rate),nn.Linear(future_channels*2,d_model*2),activation() ,nn.Dropout(dropout_rate),nn.Linear(d_model*2,d_model))
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout_rate,
                                        output_attention=False),
                        d_model, n_head),
                    d_model,
                    hidden_size,
                    moving_avg=kernel_size,
                    dropout=dropout_rate,
                    activation=activation
                ) for l in range(n_layer_encoder)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout_rate,
                                        output_attention=False),
                        d_model, n_head),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout_rate,
                                        output_attention=False),
                        d_model, n_head),
                    d_model,
                    out_channels,
                    hidden_size,
                    moving_avg=kernel_size,
                    dropout=dropout_rate,
                    activation=activation,
                )
                for l in range(n_layer_decoder)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, out_channels, bias=True)
        )
        self.pee = PositionalEmbedding(d_model=d_model)
        self.ped = PositionalEmbedding(d_model=d_model)

    def forward(self, batch):
        
        #self.decoder.device = self.device
        #self.encoder.device = self.device

        idx_target = batch['idx_target'][0]
        idx_target_future = batch['idx_target_future'][0]
        x_seq = batch['x_num_past'].to(self.device)#[:,:,idx_target]
        
        
        
        if 'x_cat_future' in batch.keys():
            cat_future = batch['x_cat_future'].to(self.device)
        if 'x_cat_past' in batch.keys():
            cat_past = batch['x_cat_past'].to(self.device)
        if 'x_num_future' in batch.keys():
            x_future = batch['x_num_future'].to(self.device)
        
        x_future[:,:-self.pred_len,idx_target_future] = -100
        
        

        pee = self.pee(x_seq).repeat(x_seq.shape[0],1,1)
        ped = self.ped(torch.zeros(x_seq.shape[0], self.pred_len+self.label_len).float()).repeat(x_seq.shape[0],1,1)
        if 'x_cat_past' in batch.keys():
            for i in range(len(self.embs)):
                if i>0:
                    tmp_emb+=self.embs[i](cat_past[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_past[:,:,i])
            pee+=tmp_emb
    
        
        
        if 'x_cat_future' in batch.keys():
            for i in range(len(self.embs)):
                if i>0:
                    tmp_emb+=self.embs[i](cat_future[:,:,i])
                else:
                    tmp_emb=self.embs[i](cat_future[:,:,i])
            ped+=tmp_emb
            

        
        mean = torch.mean(x_seq, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)

        zeros = torch.zeros([x_future.shape[0], self.pred_len, x_seq.shape[2]], device=x_seq.device)
        seasonal_init, trend_init = self.decomp(x_seq)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        import pdb
        pdb.set_trace()
        enc_out = self.linear_encoder(x_seq)+pee
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.linear_decoder(x_future)+ped
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part


  
        return dec_out[:, -self.pred_len:, :].unsqueeze(3)  # [B, L, D]
         
        
