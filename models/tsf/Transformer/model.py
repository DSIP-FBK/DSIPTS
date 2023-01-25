import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder 

class Transformer(nn.Module):
    def __init__(self, seq_len, lag, 
                time_emb, y_emb, full_emb, context_size,
                n_enc, n_dec, heads, forward_exp, dropout, device):
        super(Transformer, self).__init__()

        # PARAMS
        self.lag = lag
        self.time_emb_size = time_emb
        self.full_emb_size = full_emb
        self.context_size = context_size

        # CONV OF Y
        self.convy = nn.Conv1d(1,y_emb, kernel_size=7, padding='same')

        # EMB OF X
        self.emb_month = nn.Embedding(12+1, time_emb)
        self.emb_day = nn.Embedding(31+1, time_emb)
        self.emb_hour = nn.Embedding(23+1, time_emb)
        self.emb_dow = nn.Embedding(7+1, time_emb)
        self.proj = nn.Linear(context_size*time_emb, full_emb, bias = False) 
        # EMB OF POSITIONS
        self.emb_position = nn.Embedding(seq_len+1, full_emb)
        # EMB OF PAST/FUTURE
        self.given_embed = nn.Embedding(2, full_emb) # given = {0,1} wrt past or fututre
        # EMB OF Is_Low
        self.low = nn.Embedding(2, full_emb) # given = {0,1} wrt past or fututre
        
        self.Encoder = Encoder(full_embed_size=full_emb, n_enc=n_enc,
                                heads=heads, forward_expansion=forward_exp, dropout=dropout, device=device)
        
        self.Decoder = Decoder(full_embed_size=full_emb, n_dec=n_dec,
                                heads=heads, forward_expansion=forward_exp, dropout=dropout, device=device)
        
        self.linear_out = nn.Linear(full_emb, 1, bias = False)
        self.device = device

    def forward(self, ds, y, low): 
        
        seq_len = ds.shape[1]
        bs = ds.shape[0]
        split = seq_len-self.lag

        # PAST/FUTURE:
        past = self.given_embed(torch.zeros(bs, split).int().to(self.device))
        future = self.given_embed(torch.ones(bs, self.lag).int().to(self.device))
        time = torch.cat((past, future),dim=1)

        # LOW:
        low = self.low(low).to(self.device)
        
        # X:
        embm = self.emb_month(ds[:,:,1])
        embd = self.emb_day(ds[:,:,2])
        embh = self.emb_hour(ds[:,:,3])
        embw = self.emb_dow(ds[:,:,4])
        x = torch.cat((embm, embd, embh, embw), dim = 2).to(self.device)
        assert self.context_size*self.time_emb_size==x.shape[2]
        x = self.proj(x)

        # POS:
        positions = torch.arange(0,seq_len).repeat(bs,1).to(self.device)
        emb_pos = self.emb_position(positions)

        x = (x + time + emb_pos + low)/4
        
        # VALUES OF Y FOR ALL STEPS:
        y = torch.unsqueeze(y,dim=2).to(self.device)

        for i in range(self.lag):
            past_x = x[:,:split+i,:]
            past_y = y[:,:split+i,:]

            future_x = x[:,split+i:split+i+1,:]
            
            # MASKS:
            encoder_mask = None
            decoder_mask = None # autoregressive model
            
            # Encoder
            enc_context = self.Encoder(past_x, encoder_mask)

            # Decoder
            out = self.Decoder(future_x, enc_context, past_y, encoder_mask, decoder_mask)

            # Linear
            out = self.linear_out(out)
            
            y[:,split+i,:] = out[:,0,:]

        return y[:,-self.lag:]