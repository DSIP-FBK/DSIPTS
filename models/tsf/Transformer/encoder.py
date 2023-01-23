import torch.nn as nn
from attention import CustomAttention

class EncoderBlock(nn.Module):
    def __init__(self, full_embed_size,
                heads, forward_expansion, dropout, device):
        super(EncoderBlock, self).__init__()
        self.full_embed_size = full_embed_size
        self.device = device
        self.attention = CustomAttention(heads=heads, shape_size=full_embed_size, device=device)
        # self.softmax = nn.Softmax()
        self.norm1 = nn.LayerNorm(full_embed_size) 
        
        self.linear1 = nn.Linear(full_embed_size, forward_expansion*full_embed_size, bias = False)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(forward_expansion*full_embed_size, full_embed_size, bias = False)
        self.norm2 = nn.LayerNorm(full_embed_size)
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, mask):
    
        # Self Att
        x = self.attention(queries, keys, values, mask)
        x = x + values
        x = self.norm1(x)
        
        # FFN
        ffn = self.linear1(x.float())
        ffn = self.ReLU(ffn)
        ffn = self.linear2(ffn)
        ffn = ffn + x
        ffn = self.norm2(ffn)
        # out = self.dropout(ffn)
        
        return ffn # out
        
class Encoder(nn.Module):
    def __init__(self, full_embed_size, n_enc, 
                heads, forward_expansion, dropout, device):
        super(Encoder, self).__init__()
        self.num_layers_enc = n_enc
        self.device = device
        self.norm = nn.LayerNorm(full_embed_size)
        self.layers_enc = nn.ModuleList(
            [EncoderBlock(full_embed_size=full_embed_size, heads=heads,
             forward_expansion=forward_expansion, dropout=dropout, device=device) for _ in range(n_enc)] )
        
    def forward(self, qkv, mask):
        
        encoding = qkv
        # import pdb 
        # pdb.set_trace()
        for layer in self.layers_enc:
            encoding = layer(encoding, qkv, qkv, mask)
            encoding = self.norm(encoding)
            
        return encoding
