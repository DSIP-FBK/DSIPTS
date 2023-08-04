import torch
import torch.nn as nn

class embedding_cat_variables(nn.Module):
    # at the moment cat_past and cat_fut together
    def __init__(self, seq_len: int, lag: int, hiden_size: int, emb_dims: list, device):
        """Class for embedding categorical variables, adding 3 positional variables during forward

        Args:
            seq_len (int): length of the sequence (sum of past and future steps)
            lag (int): number of future step to be predicted
            hiden_size (int): dimension of all variables after they are embedded
            emb_dims (list): size of the dictionary for embedding. One dimension for each categorical variable
            device : -
        """
        super().__init__()
        self.seq_len = seq_len
        self.lag = lag
        self.device = device
        self.cat_embeds = emb_dims + [seq_len, lag+1, 2] # 
        self.cat_n_embd = nn.ModuleList([
            nn.Embedding(emb_dim, hiden_size) for emb_dim in self.cat_embeds
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """All components of x are concatenated with 3 new variables for data augmentation, in the order:
        - pos_seq: assign at each step its time-position
        - pos_fut: assign at each step its future position. 0 if it is a past step
        - is_fut: explicit for each step if it is a future(1) or past one(0)

        Args:
            x (torch.Tensor): [bs, seq_len, num_vars]

        Returns:
            torch.Tensor: [bs, seq_len, num_vars+3, n_embd] 
        """
        if len(x.shape)==0:
            no_emb = True
            B = x.item()
        else:
            no_emb = False
            B, _, _ = x.shape
        
        pos_seq = self.get_pos_seq(bs=B).to(x.device)
        pos_fut = self.get_pos_fut(bs=B).to(x.device)
        is_fut = self.get_is_fut(bs=B).to(x.device)
        if no_emb:
            cat_vars = torch.cat((pos_seq, pos_fut, is_fut), dim=2)
        else:
            cat_vars = torch.cat((x, pos_seq, pos_fut, is_fut), dim=2)
        cat_n_embd = self.get_cat_n_embd(cat_vars)
        return cat_n_embd

    def get_pos_seq(self, bs):
        pos_seq = torch.arange(0, self.seq_len)
        pos_seq = pos_seq.repeat(bs,1).unsqueeze(2).to(self.device)
        return pos_seq
    
    def get_pos_fut(self, bs):
        pos_fut = torch.cat((torch.zeros((self.seq_len-self.lag), dtype=torch.long),torch.arange(1,self.lag+1)))
        pos_fut = pos_fut.repeat(bs,1).unsqueeze(2).to(self.device)
        return pos_fut
    
    def get_is_fut(self, bs):
        is_fut = torch.cat((torch.zeros((self.seq_len-self.lag), dtype=torch.long),torch.ones((self.lag), dtype=torch.long)))
        is_fut = is_fut.repeat(bs,1).unsqueeze(2).to(self.device)
        return is_fut
    
    def get_cat_n_embd(self, cat_vars):
        cat_n_embd = torch.Tensor().to(cat_vars.device)
        for index, layer in enumerate(self.cat_n_embd):
            emb = layer(cat_vars[:, :, index])
            cat_n_embd = torch.cat((cat_n_embd, emb.unsqueeze(2)),dim=2)
        return cat_n_embd

class LSTM_Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out[:, -1, :])  # Take the last output of the sequence
        return out
    
class GLU(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.sigmoid(self.linear1(x))
        x2 = self.linear2(x)
        out = x1*x2 #element-wise multiplication
        return out
    
class GRN(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim) 
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eta1 = self.elu(self.linear1(x))
        eta2 = self.dropout(self.linear2(eta1))
        out = self.norm(x + self.glu(eta2))
        return out