import torch
import torch.nn as nn

#* ENCODER
class Head_selfEnc(nn.Module):
    """ one head of self-att for encoder """

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.val = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, T, C = q.shape

        q = self.query(q)
        k = self.key(k)
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.val(v)
        out = wei @ v
        return out

class MultiHeadEnc(nn.Module):

    def __init__(self, n_embd, num_heads, head_size, dropout) :
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([Head_selfEnc(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        
    def forward(self, q, k, v):
        out = torch.cat([h(q, k, v) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
        # out = torch.sum([h(q, k, v) for h in self.heads], dim=-1)/self.num_heads
        # return out

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

#   ENCODER-LAYER    #
class EncoderLayer(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, fw_exp, dropout) :
        super().__init__()
        self.heads = MultiHeadEnc(n_embd, num_heads, head_size, dropout)
        self.ffn = FFN(n_embd, fw_exp)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, q, k, v):
        q = q + self.heads(self.norm1(q), k, v)
        q = q + self.ffn(self.norm2(q))
        return q

#*   ENCODER   #
class Encoder(nn.Module):
    def __init__(self, n_enc, n_embd, num_heads, head_size, fw_exp, dropout) :
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(n_embd, num_heads, head_size, fw_exp, dropout) 
                                        for _ in range(n_enc)])
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, q, k, v):
        encoding = q
        for layer in self.layers:
            encoding = layer(encoding, k, v)
        return encoding