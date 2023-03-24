import torch
import torch.nn as nn

class Embed(nn.Module):
    # Embed all categorical variables concatenating them together (context of 8 variables)
    # Embed y[:,:-self.lag,:] with a Linear
    # Embed y[:,-self.lag-1:-1,:] with a Linear, maintaining the shift
    # nn.Linear y_lin to transform predicted y if self.prec 
    def __init__(self, mix, prec, tft, seq_len, lag, n_embd, device) :
        super().__init__()
        self.mix = mix
        self.pewc = prec
        self.tft = tft
        self.lag = lag
        self.device = device
        # time
        self.emb_m = nn.Embedding(12+1, n_embd)
        self.emb_d = nn.Embedding(31+1, n_embd)
        self.emb_h = nn.Embedding(24, n_embd)
        self.emb_dow = nn.Embedding(7, n_embd)
        # pos
        self.emb_pos = nn.Embedding(seq_len, n_embd)
        # past or future
        self.emb_past_or_fut = nn.Embedding(2, n_embd)
        # future_pos
        self.emb_future_pos = nn.Embedding(lag+1, n_embd)
        # is_future
        self.emb_is_future = nn.Embedding(2, n_embd)
        # is low or not
        self.emb_low = nn.Embedding(3, n_embd)
        # y
        self.y_lin = nn.Linear(1,n_embd) # linear used as embedding for continuous variable
    
    def forward(self, x, y):
        # import pdb
        # pdb.set_trace()
        B, T, C = x.shape

        emb_m = self.emb_m(x[:,:,1]).unsqueeze(2)
        emb_d = self.emb_d(x[:,:,2]).unsqueeze(2)
        emb_h = self.emb_h(x[:,:,3]).unsqueeze(2)
        emb_dow = self.emb_dow(x[:,:,4]).unsqueeze(2)
        # position inside the sequence
        pos_seq = torch.arange(0,T).repeat(B,1).to(self.device)
        emb_pos_seq = self.emb_pos(pos_seq).unsqueeze(2)
        # position in the future (0 if not)
        pos_fut = torch.cat((torch.zeros((T-self.lag), dtype=torch.long),torch.arange(1,self.lag+1)))
        pos_fut = pos_fut.repeat(B,1).to(self.device)
        emb_pos_fut = self.emb_future_pos(pos_fut).unsqueeze(2)
        # is future or not
        pos_is_fut = torch.cat((torch.zeros((T-self.lag), dtype=torch.long),torch.ones((self.lag), dtype=torch.long)))
        pos_is_fut = pos_is_fut.repeat(B,1).to(self.device)
        emb_is_fut = self.emb_is_future(pos_is_fut).unsqueeze(2)
        # low
        emb_low = self.emb_low(x[:,:,5]).unsqueeze(2)
        emb_x = torch.cat(
                (emb_m, emb_d, emb_h, emb_dow, emb_pos_seq, emb_pos_fut, emb_is_fut, emb_low),
                dim=2)
        # Y
        emb_y_past = self.y_lin(y.unsqueeze(2)[:,:-self.lag,:].float())
        emb_y_fut = self.y_lin(y.unsqueeze(2)[:,-self.lag-1:-1,:].float())

        return emb_x, emb_y_past, emb_y_fut, self.y_lin
        # Embed all categorical variables concatenating them together (context of 8 variables)
        # Embed y[:,:-self.lag,:] with a Linear
        # Embed y[:,-self.lag-1:-1,:] with a Linear, maintaining the shift
        # nn.Linear y_lin to transform predicted y if self.prec
    
class Embed_tft(nn.Module):
    def __init__(self, mix, prec, tft, seq_len, lag, n_embd, device) -> None:
        super().__init__()
        self.mix = mix
        self.prec = prec
        self.tft = tft
        self.lag = lag
        self.device = device
        self.dropout = 0.3
        self.emb_init = 1 # or 2

        self.embeds = [12+1, 31+1, 24, 7, 3, seq_len, lag+1, 2]
        
        self.module_n_embd = nn.ModuleList([
            nn.Embedding(emb_dim, n_embd) for emb_dim in self.embeds
        ])
        self.GRN_n_embd = nn.ModuleList([
        GRN(n_embd, self.dropout) for _ in self.module_n_embd
        ])
        self.y_lin_n_embd = nn.Linear(1, n_embd, bias = False) # y_lin

        #only for variable selection weights
        self.module_1 = nn.ModuleList([
            nn.Embedding(emb_dim, 1) for emb_dim in self.embeds
        ])
        self.GRN_1 = nn.ModuleList([
        GRN(1, self.dropout) for _ in self.module_1
        ])
        self.y_lin_1 = nn.Linear(1, 1, bias = False)

        self.num_layers = 3
        self.hidden_size = n_embd
        self.LSTM_enc = nn.LSTM(input_size=n_embd, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first = True)
        self.LSTM_enc_dropout = nn.Dropout(self.dropout)
        self.LSTM_enc_GLU = GLU(n_embd)
        self.enc_norm = nn.LayerNorm(n_embd)
        # self.GRN_enc = GRN(n_embd, self.dropout)

        self.LSTM_dec = nn.LSTM(input_size=n_embd, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first = True)
        self.LSTM_dec_dropout = nn.Dropout(self.dropout)
        self.LSTM_dec_GLU = GLU(n_embd)
        self.dec_norm = nn.LayerNorm(n_embd)
        # self.GRN_dec = GRN(n_embd, self.dropout)

        self.softmax_past = nn.Softmax(dim=2)
        self.softmax_fut = nn.Softmax(dim=2)

    def forward(self, x, y):
        B, T, C = x.shape

        #* Starting Emb
        # emb_*_1: dislike?

        pos_seq = torch.arange(0,T)
        pos_seq = pos_seq.repeat(B,1).unsqueeze(2).to(self.device)
        pos_fut = torch.cat((torch.zeros((T-self.lag), dtype=torch.long),torch.arange(1,self.lag+1)))
        pos_fut = pos_fut.repeat(B,1).unsqueeze(2).to(self.device)
        pos_is_fut = torch.cat((torch.zeros((T-self.lag), dtype=torch.long),torch.ones((self.lag), dtype=torch.long)))
        pos_is_fut = pos_is_fut.repeat(B,1).unsqueeze(2).to(self.device)
        to_be_emb = torch.cat((x[:,:,1:], pos_seq, pos_fut, pos_is_fut),dim=2)
        # print(f'to_be_emb.shape = {to_be_emb.shape}')

        #todo: 
        # embed all
        # Variable Selection + LSTM + GRN only on past
        # return variables embedded
        # Decoder part is better if completely in model_tft.py, easier to recall for inference
        # Move LSTM full net to model_tft.py
        # .
        
        # only X
        emb_x_n_embd = torch.Tensor().to(self.device)
        for index, layer in enumerate(self.module_n_embd):
            emb = layer(to_be_emb[:, :, index])
            emb = self.GRN_n_embd[index](emb)
            emb_x_n_embd = torch.cat((emb_x_n_embd, emb.unsqueeze(2)),dim=2)

        emb_x_1 = torch.Tensor().to(self.device)
        for index, layer in enumerate(self.module_1):
            emb = layer(to_be_emb[:, :, index])
            emb = self.GRN_1[index](emb)
            emb_x_1 = torch.cat((emb_x_1, emb),dim=2)
        
        # only Y
        emb_y_n_embd = self.y_lin_n_embd(y.unsqueeze(2).float())
        emb_y_1 = self.y_lin_1(y.unsqueeze(2).float())
        
        #PAST
        # x
        emb_x_n_embd_past = emb_x_n_embd[:,:-self.lag,:,:] # ([8, 191, 8, 4])
        emb_x_1_past = emb_x_1[:,:-self.lag,:] # ([8, 191, 8])
        # y
        emb_y_n_embd_past = emb_y_n_embd[:,:-self.lag,:] # ([8, 191, 4])
        emb_y_1_past = emb_y_1[:,:-self.lag,:] # ([8, 191, 1])

        #FUT
        # x 
        emb_x_n_embd_fut = emb_x_n_embd[:,-self.lag:,:,:] # ([8, 65, 8, 4])
        emb_x_1_fut = emb_x_1[:,-self.lag:,:] # ([8, 65, 8])
        # y 
        emb_y_n_embd_fut = emb_y_n_embd[:,-self.lag-1:-1,:] # ([8, 65, 4]) ## ONLY IF PREC = TRUE
        emb_y_1_fut = emb_y_1[:,-self.lag-1:-1,:] # ([8, 65, 1]) ## ONLY IF PREC = TRUE

        # MIX 
        #* Variable Selection Encoder
        if self.mix:
            emb_x_n_embd_past = torch.cat((emb_x_n_embd_past, emb_y_n_embd_past.unsqueeze(2)),dim=2)
            emb_x_1_past = torch.cat((emb_x_1_past, emb_y_1_past),dim=2)
        emb_x_1_past = self.softmax_past(emb_x_1_past)
        emb_past = emb_x_n_embd_past*emb_x_1_past.unsqueeze(3) # ([8, 191, 9, 4])
        emb_past = torch.sum(emb_past, 2)/emb_past.size(2) # ([8, 191, 4])

        #PREC
        #* Variable Selection Decoder
        if self.prec:
            emb_x_n_embd_fut = torch.cat((emb_x_n_embd_fut, emb_y_n_embd_fut.unsqueeze(2)),dim=2)
            emb_x_1_fut = torch.cat((emb_x_1_fut, emb_y_1_fut),dim=2)
        emb_x_1_fut = self.softmax_fut(emb_x_1_fut)
        emb_fut = emb_x_n_embd_fut*emb_x_1_fut.unsqueeze(3) # ([8, 65, 9, 4])
        emb_fut = torch.sum(emb_fut, 2)/emb_fut.size(2) # ([8, 65, 4])

        # LSTM ENCODER + dropout, gate (GLU), add & norm
        h_0_enc = torch.zeros(self.num_layers, emb_past.size(0), emb_past.size(2)).to(self.device)
        c_0_enc = torch.zeros(self.num_layers, emb_past.size(0), emb_past.size(2)).to(self.device)
        lstm_enc, _ = self.LSTM_enc(emb_past, (h_0_enc,c_0_enc))
        
        lstm_enc = self.LSTM_enc_dropout(lstm_enc) # dropout
        output_enc = self.enc_norm(self.LSTM_enc_GLU(lstm_enc) + emb_past) 

        # LSTM DECODER + dropout, gate (GLU), add & norm
        h_0_dec = torch.zeros(self.num_layers, emb_fut.size(0), emb_fut.size(2)).to(self.device)
        c_0_dec = torch.zeros(self.num_layers, emb_fut.size(0), emb_fut.size(2)).to(self.device)
        lstm_dec, _ = self.LSTM_dec(emb_fut, (h_0_dec,c_0_dec))
        
        lstm_dec = self.LSTM_dec_dropout(lstm_dec) # dropout
        output_dec = self.dec_norm(self.LSTM_dec_GLU(lstm_dec) + emb_fut)

        if self.mix:
            return output_enc, output_dec
        else:
            return emb_y_n_embd_past, output_enc, output_dec

    def dec_emb(self, x_fut, predictions):
        # todo: emb x_fut and predictions for each step of iter_forward
        emb_x_n_embd = torch.Tensor().to(self.device)
        for index, layer in enumerate(self.module_n_embd):
            emb = layer(to_be_emb[:, :, index])
            emb = self.GRN_n_embd[index](emb)
            emb_x_n_embd = torch.cat((emb_x_n_embd, emb.unsqueeze(2)),dim=2)

        emb_x_1 = torch.Tensor().to(self.device)
        for index, layer in enumerate(self.module_1):
            emb = layer(to_be_emb[:, :, index])
            emb = self.GRN_1[index](emb)
            emb_x_1 = torch.cat((emb_x_1, emb),dim=2)

        emb_pred_n_embd = self.y_lin_n_embd(predictions.unsqueeze(2).float())
        emb_pred_1 = self.y_lin_1(predictions.unsqueeze(2).float())

        emb_x_n_embd_fut = torch.cat((emb_x_n_embd_fut, emb_pred_n_embd.unsqueeze(2)),dim=2)
        emb_x_1_fut = torch.cat((emb_x_1_fut, emb_pred_1),dim=2)
        emb_fut = emb_x_n_embd_fut*emb_x_1_fut.unsqueeze(3) # ([8, 65, 9, 4])
        emb_fut = torch.sum(emb_fut, 2)/emb_fut.size(2) # ([8, 65, 4])
        # used for prec = True
        # x_fut and prediction already [:, :tau, :]
        h_0_dec = torch.zeros(self.num_layers, x_fut.size(0), x_fut.size(2)).to(self.device)
        c_0_dec = torch.zeros(self.num_layers, x_fut.size(0), x_fut.size(2)).to(self.device)
        lstm_dec, (hn_dec,cn_dec) = self.LSTM_dec(x_fut, (h_0_dec,c_0_dec))
        # dropout, gate (GLU), add & norm
        lstm_dec = self.LSTM_dec_dropout(lstm_dec) # dropout
        output_dec = self.dec_norm(self.LSTM_dec_GLU(lstm_dec) + x_fut)

        return output_dec

class GLU(nn.Module):
    # sub net of GRN 
    def __init__(self, n_embd) :
        super().__init__()
        self.linear1 = nn.Linear(n_embd, n_embd)
        self.linear2 = nn.Linear(n_embd, n_embd)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.sigmoid(self.linear1(x))
        x2 = self.linear2(x)
        out = x1*x2 #element-wise multiplication
        return out
    
class GRN(nn.Module):
    # one GRN for each variable 
    def __init__(self, n_embd, dropout) :
        super().__init__()
        self.norm = nn.LayerNorm(n_embd)
        self.glu = GLU(n_embd)
        self.elu = nn.ELU()
        self.linear1 = nn.Linear(n_embd, n_embd) 
        self.linear2 = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        eta1 = self.elu(self.linear1(x))
        eta2 = self.dropout(self.linear2(eta1))
        out = self.norm(x + self.glu(eta2))
        return out
    

if __name__=='__main__':
    from dataloading import dataloading

    bs = 8
    bs_test = 4
    seq_len = 256
    lag = 65
    hour = 24
    hour_test = 24
    train = True
    path_data = '/home/andrea/timeseries/data/edison/processed.pkl' 
    train_dl, val_dl, test_dl, scaler_y = dataloading(batch_size=bs, batch_size_test=bs_test, 
                                                        seq_len=seq_len, lag=lag,
                                                        hour_learning=hour, 
                                                        hour_inference=hour_test, 
                                                        train_bool=train,
                                                        path=path_data)
    
    x, y = next(iter(train_dl))
    
    mix = True
    prec = True
    tft = True
    n_embd = 4
    device = 'cpu'
    TFT = Embed_tft(mix, prec, tft, seq_len, lag, n_embd, device)
    emb = TFT(x, y)
    # print(emb.shape)
    import pdb
    pdb.set_trace()



