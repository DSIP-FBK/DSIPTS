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
        # self.val = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        
        import pdb
        pdb.set_trace()
        queries = self.query(queries)
        keys = self.key(keys)
        wei = queries @ keys.transpose(-2,-1) * self.head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # values = self.val(values)
        out = wei @ values
        return out

class MultiHeadEnc(nn.Module):

    def __init__(self, n_embd, num_heads, head_size, dropout) :
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.heads = nn.ModuleList([Head_selfEnc(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.common_values_linear = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, queries, keys, values):
        
        import pdb
        pdb.set_trace()
        device = queries.device.type
        B, L = queries.shape[:2]
        out = torch.zeros(B, L, self.head_size).to(device)
        # we want for 'values' tensor to have the same weights shared across all heads
        common_values = self.common_values_linear(values)
        for h in self.heads:
            out += h(queries, keys, common_values)
        out = out/self.num_heads
        return out

class FFN(nn.Module):

    def __init__(self, n_embd, fw_exp) :
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, fw_exp*n_embd),
            nn.ReLU(),
            nn.Linear(fw_exp*n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

#   ENCODER-LAYER    #
class EncoderLayer(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, fw_exp, dropout) :
        super().__init__()
        self.heads = MultiHeadEnc(n_embd, num_heads, head_size, dropout)
        self.linear_to_embd = nn.Linear(head_size, n_embd)
        self.ffn = FFN(n_embd, fw_exp)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, queries, keys, values):
        
        import pdb
        pdb.set_trace()
        queries = queries + self.heads(self.norm1(queries), keys, values)
        queries = self.linear_to_embd(queries)
        queries = queries + self.ffn(self.norm2(queries))
        return queries

#   ENCODER   #
class Encoder(nn.Module):
    def __init__(self, n_enc: int, n_embd: int, num_heads: int, head_size: int, fw_exp: int, dropout: float):
        """Encoder

        Args:
            n_enc (int): number of encoder layers
            n_embd (int): model dimension
            num_heads (int): number of heads for each layer
            head_size (int): size of layers heads
            fw_exp (int): multiplicative factor for expansion in FFN
            dropout (float): 
        """
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(n_embd, num_heads, head_size, fw_exp, dropout) 
                                        for _ in range(n_enc)])
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Encoder:
        - Multi Head SelfAttention on Encoder
        - Feed Forward Network

        Args:
            q (torch.Tensor): queries
            k (torch.Tensor): keys
            v (torch.Tensor): values

        Returns:
            torch.Tensor: encoded tensor
        """
        import pdb
        pdb.set_trace()
        encoding = queries
        # iterate queries over encoder layers
        for layer in self.layers:
            encoding = layer(encoding, keys, values)
        # final normalization
        encoding = self.norm(encoding)
        return encoding