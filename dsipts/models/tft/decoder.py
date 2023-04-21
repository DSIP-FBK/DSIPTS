import torch
import torch.nn as nn

#* DECODER
class Head_selfDec(nn.Module):
    """ one head of self-att for decoder"""

    def __init__(self, n_embd, head_size, lag, dropout):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.val = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(lag, lag))) # create the variable 'self.tril'
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T = x.shape[1]

        queries = self.query(x)
        keys = self.key(x)
        wei = queries @ keys.transpose(-2,-1) * self.head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] ==0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        values = self.val(x)
        out = wei @ values
        return out

class Head_crossDec(nn.Module):
    """ one head of cross-att for decoder"""

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.val = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        queries = self.query(queries)
        keys = self.key(keys)
        wei = queries @ keys.transpose(-2,-1) * self.head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        values = self.val(values)
        out = wei @ values
        return out

#   MULTIHEAD selfDECODER    #
class MultiHead_selfDec(nn.Module):

    def __init__(self, n_embd, num_heads, head_size, lag, dropout) :
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.heads = nn.ModuleList([Head_selfDec(n_embd, head_size, lag, dropout) for _ in range(num_heads)])
        
    def forward(self, x):
        B, L = x.shape[:2]
        out = torch.zeros(B, L, self.head_size)
        for h in self.heads:
            out += h(x)
        out = out/self.num_heads
        return out

#   MULTIHEAD crossDECODER    #
class MultiHead_crossDec(nn.Module):

    def __init__(self, n_embd, num_heads, head_size, dropout) :
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.heads = nn.ModuleList([Head_crossDec(n_embd, head_size, dropout) for _ in range(num_heads)])
        
    def forward(self, queries, keys, values):
        B, L = queries.shape[:2]
        out = torch.zeros(B, L, self.head_size)
        for h in self.heads:
            out += h(queries, keys, values)
        out = out/self.num_heads
        return out

#   FFN    #
class FFN(nn.Module):

    def __init__(self, n_embd, fw_exp) :
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, fw_exp*n_embd),
            nn.ReLU(),
            nn.Linear(fw_exp*n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

#   DECODER-LAYER    #
class DecoderLayer(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, fw_exp, lag, dropout) :
        super().__init__()
        self.self_heads = MultiHead_selfDec(n_embd, num_heads, head_size, lag, dropout)
        self.cross_heads = MultiHead_crossDec(n_embd, num_heads, head_size, dropout)
        self.ffn = FFN(n_embd, fw_exp)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.norm3 = nn.LayerNorm(n_embd)

    def forward(self, q, k, v): # q = x_future, k = output of encoder, v = y_past
        q = q + self.self_heads(self.norm1(q))
        q = q + self.cross_heads(self.norm2(q), k, v)
        q = q + self.ffn(self.norm3(q))
        return q

#   DECODER   #
class Decoder(nn.Module):
    def __init__(self, n_dec, n_embd, num_heads, head_size, fw_exp, lag, dropout) :
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(n_embd, num_heads, head_size, fw_exp, lag, dropout)
                                        for _ in range(n_dec)])
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, q, k, v):
        decoding = q
        for layer in self.layers:
            decoding = layer(decoding, k, v)
        return decoding