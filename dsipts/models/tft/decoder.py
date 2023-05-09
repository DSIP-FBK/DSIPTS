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
        # self.val = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(lag, lag))) # create the variable 'self.tril'
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        T = queries.shape[1]

        queries = self.query(queries)
        keys = self.key(keys)
        wei = queries @ keys.transpose(-2,-1) * self.head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] ==0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # values = self.val(values)
        out = wei @ values
        return out

class Head_crossDec(nn.Module):
    """ one head of cross-att for decoder"""

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # self.val = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        queries = self.query(queries)
        keys = self.key(keys)
        wei = queries @ keys.transpose(-2,-1) * (self.head_size**-0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # values = self.val(values)
        out = wei @ values
        return out

#   MULTIHEAD selfDECODER    #
class MultiHead_selfDec(nn.Module):

    def __init__(self, n_embd, num_heads, head_size, lag, dropout) :
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.heads = nn.ModuleList([Head_selfDec(n_embd, head_size, lag, dropout) for _ in range(num_heads)])
        self.common_values_linear = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, queries, keys, values):
        device = queries.device.type
        B, L = queries.shape[:2]
        out = torch.zeros(B, L, self.head_size).to(device)
        # we want for 'values' tensor to have the same weights shared across all heads
        common_values = self.common_values_linear(values)
        for h in self.heads:
            out += h(queries, keys, common_values)
        out = out/self.num_heads
        return out

#   MULTIHEAD crossDECODER    #
class MultiHead_crossDec(nn.Module):

    def __init__(self, n_embd, num_heads, head_size, dropout) :
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.heads = nn.ModuleList([Head_crossDec(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.common_values_linear = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, queries, keys, values):
        device = queries.device.type
        B, L = queries.shape[:2]
        out = torch.zeros(B, L, self.head_size).to(device)
        # we want for 'values' tensor to have the same weights shared across all heads
        common_values = self.common_values_linear(values)
        for h in self.heads:
            out += h(queries, keys, common_values)
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
        self.linear1_to_embd = nn.Linear(head_size, n_embd)
        self.cross_heads = MultiHead_crossDec(n_embd, num_heads, head_size, dropout)
        self.linear2_to_embd = nn.Linear(head_size, n_embd)
        self.ffn = FFN(n_embd, fw_exp)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.norm3 = nn.LayerNorm(n_embd)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor: 
        # q = x_future, k = output of encoder, v = output of encoder
        # decoder self attention over future values
        q = q + self.linear1_to_embd( self.self_heads(self.norm1(q)))
        # cross attention among decoder and encoder
        q = q + self.linear2_to_embd( self.cross_heads(self.norm2(q), k, v))
        # Feed Forward Network
        q = q + self.ffn(self.norm3(q))
        return q

#   DECODER   #
class Decoder(nn.Module):
    def __init__(self, n_dec: int, n_embd: int, num_heads: int, head_size: int, fw_exp: int, lag: int, dropout: float) :
        """Decoder

        Args:
            n_dec (int): number of decoder layers
            n_embd (int): model dimension
            num_heads (int): number of heads for each layer 
            head_size (int): size of layers' heads
            fw_exp (int): multiplicative factor for expansion in FFN
            lag (int): maximum number of future steps computable. Used in Head_selfDec
            dropout (float):
        """
        super().__init__()
        # list of decoder layers
        self.layers = nn.ModuleList([DecoderLayer(n_embd, num_heads, head_size, fw_exp, lag, dropout)
                                        for _ in range(n_dec)])
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Decoder: 
        - Multi Head SelfAttention on Decoder
        - Multi Head CrossAttention among Decoder queries and Encoder keys and values
        - Feed Forward Network

        Args:
            q (torch.Tensor): queries 
            k (torch.Tensor): keys
            v (torch.Tensor): values

        Returns:
            torch.Tensor: decoded tensor
        """
        decoding = q
        # iterate queries over decoder layers
        for layer in self.layers:
            decoding = layer(decoding, k, v)
        # final normalization
        decoding = self.norm(decoding)
        return decoding
    