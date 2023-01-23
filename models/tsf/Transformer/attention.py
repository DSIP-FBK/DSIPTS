import torch.nn as nn
import torch
import numpy as np

class CustomAttention(nn.Module):
    def __init__(self, heads, shape_size, device):
        super(CustomAttention,self).__init__()
        self.heads = heads
        self.shape_size = shape_size
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.linear_concat = nn.Linear(shape_size*heads, shape_size, bias = False)

    def forward(self, queries, keys, values, mask):
        
        # bs, dim_queries, context_size_q = queries.shape
        # bs, dim_keys, context_size_k = keys.shape
        # bs, dim_values, context_size_v = values.shape

        # assert dim_keys==dim_values
        # assert context_size_q==context_size_k
        
        # Scaled Dot Products Attention
        # import pdb
        # pdb.set_trace()
        
        att = torch.Tensor().to(self.device)
        for _ in range(self.heads):
            qk = torch.einsum('sdQ,SDK->sdD',[queries,keys])
            if mask is not None:
                qk = qk.masked_fill(mask==float('-inf'), float('-inf')) # element-wise multiplication to apply mask
            
            attention_pooling = self.softmax(qk/np.sqrt(self.shape_size))
            attention = torch.einsum('sdD,SDv -> sdv',[attention_pooling.double(), values.double()]).float()
            att = torch.cat((att, attention), dim=2)
        
        multi_head_attention = self.linear_concat(att)
        
        return multi_head_attention