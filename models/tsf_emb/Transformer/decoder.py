import torch.nn as nn
from attention import CustomAttention

class DecoderBlock(nn.Module):
    def __init__(self, full_embed_size,
                heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.full_embed_size = full_embed_size
        self.device = device
        self.masked_attention = CustomAttention(heads=heads, shape_size=full_embed_size, device=device)
        self.target_attention = CustomAttention(heads=heads, shape_size=1, device=device)
        self.norm1 = nn.LayerNorm(full_embed_size)
        self.linear1 = nn.Linear(full_embed_size,forward_expansion*full_embed_size, bias = False)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(forward_expansion*full_embed_size,full_embed_size, bias = False)
        self.norm2 = nn.LayerNorm(full_embed_size)
        self.norm3 = nn.LayerNorm(full_embed_size)
        # self.dropout = nn.Dropout(dropout)
    
    def forward(self, decoding, enc_context, target_seq, encoder_mask, decoder_mask):
        
        # First Mask
        masked_query = self.masked_attention(decoding, decoding, decoding, decoder_mask)
        masked_query = decoding + masked_query
        masked_query = self.norm1(masked_query)
        # masked_query = self.dropout(masked_query)

        # Second Mask
        target_att = self.target_attention(masked_query, enc_context, target_seq, encoder_mask)
        target_att = target_att + masked_query
        target_att = self.norm2(target_att)
        # target_att = self.dropout(target_att)

        # FFN
        ffn = self.linear1(target_att.float())
        ffn = self.ReLU(ffn)
        ffn = self.linear2(ffn)
        ffn = ffn + target_att
        ffn = self.norm3(ffn)
        # out = self.dropout(ffn)
        
        return ffn # out

class Decoder(nn.Module):
    def __init__(self, full_embed_size, n_dec,
                heads, forward_expansion, dropout, device):
        super(Decoder, self).__init__()
        self.num_layers_dec = n_dec
        self.device = device
        self.norm = nn.LayerNorm(full_embed_size)
        self.layers_dec = nn.ModuleList(
            [DecoderBlock(full_embed_size=full_embed_size, heads=heads, 
            forward_expansion=forward_expansion, dropout=dropout, device=device) for _ in range(self.num_layers_dec)] )
               
    def forward(self, target_seq, enc_context, past_y, encoder_mask, decoder_mask):
        
        decoding = target_seq
        for layer in self.layers_dec:
            decoding = layer(decoding, enc_context, past_y, encoder_mask, decoder_mask)
            decoding = self.norm(decoding)
            
        return decoding