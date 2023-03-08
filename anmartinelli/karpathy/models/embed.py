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
        emb_is_fut = self.emb_future_pos(pos_is_fut).unsqueeze(2)
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
        self.pewc = prec
        self.tft = tft
        self.lag = lag
        self.device = device
        # time
        self.module_list = nn.ModuleList([
        nn.Embedding(12+1, n_embd), #emb_m
        nn.Embedding(31+1, n_embd), #emb_d
        nn.Embedding(24, n_embd), # emb_h
        nn.Embedding(7, n_embd), # emb_dow
        nn.Embedding(seq_len, n_embd), # emb_pos
        nn.Embedding(2, n_embd), # emb_past_or_fut
        nn.Embedding(lag+1, n_embd), # emb_future_pos
        nn.Embedding(2, n_embd), # emb_is_future
        nn.Embedding(3, n_embd), # emb_low
        nn.Linear(1,n_embd) # y_lin
    ])
    def forward(self, x, y):
        B, T, C = x.shape
        pos_seq = torch.arange(0,T)
        pos_seq = pos_seq.repeat(B,1).unsqueeze(2)

        pos_fut = torch.cat((torch.zeros((T-lag), dtype=torch.long),torch.arange(1,lag+1)))
        pos_fut = pos_fut.repeat(B,1).unsqueeze(2)

        pos_is_fut = torch.cat((torch.zeros((T-lag), dtype=torch.long),torch.ones((lag), dtype=torch.long)))
        pos_is_fut = pos_is_fut.repeat(B,1).unsqueeze(2)

        to_be_emb = torch.cat((x[:,:,1:], pos_seq, pos_fut, pos_is_fut),dim=2)
        print(f'to_be_emb.shape = {to_be_emb.shape}')
        embedded = torch.Tensor()
        import pdb
        pdb.set_trace()
        for index, layer in enumerate(self.module_list):
            print(index)
            if index < 9:
                emb = layer(to_be_emb[:, :, index])
            else:
                emb = layer(y).unsqueeze(2)
            embedded = torch.cat((embedded, emb),dim=2)
        print('FLATTEN IMPLEMENTED')
        return embedded


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
        out = x1*x2
        return out
    
class GRN(nn.Module):
    # one GRN for each variable 
    def __init__(self, n_embd, dropout) :
        super().__init__()
        self.norm = nn.LayerNorm(n_embd)
        self.glu = GLU(n_embd)
        self.elu = nn.ELU()
        self.linear1 = nn.Linear(n_embd,n_embd) 
        self.linear2 = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        eta2 = self.elu(self.linear2(x))
        eta1 = self.dropout(self.linear1(eta2))
        out = self.norm(x + self.glu(eta1))
        return out
    

if __name__=='__main__':
    from dataloading import dataloading

    bs = 8
    bs_test = 4
    seq_len = 256
    lag = 60
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
    n_embd = 1
    device = 'cpu'
    TFT = Embed_tft(mix, prec, tft, seq_len, lag, n_embd, device)
    emb = TFT(x, y)
    print(emb.shape)
    import pdb
    pdb.set_trace()



