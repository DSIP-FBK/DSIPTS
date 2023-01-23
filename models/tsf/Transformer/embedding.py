import torch
from torch import nn


class Time2Vec(nn.Module):
    def __init__(self, input_dim=6, embed_dim=512, act_function=torch.sin):
        assert embed_dim % input_dim == 0
        super(Time2Vec, self).__init__()
        self.enabled = embed_dim > 0
        if self.enabled:
            self.embed_dim = embed_dim // input_dim
            self.input_dim = input_dim
            self.embed_weight = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.embed_bias = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.act_function = act_function

    def forward(self, x):
        if self.enabled:
            x = torch.diag_embed(x)
            # x.shape = (bs, sequence_length, input_dim, input_dim)

            x_affine = torch.matmul(x, self.embed_weight) + self.embed_bias
            # x_affine.shape = (bs, sequence_length, input_dim, time_embed_dim)

            x_affine_0, x_affine_remain = torch.split(x_affine, [1, self.embed_dim - 1], dim=-1)
            x_affine_remain = self.act_function(x_affine_remain)
            x_output = torch.cat([x_affine_0, x_affine_remain], dim=-1)
            x_output = x_output.view(x_output.size(0), x_output.size(1), -1)
            # x_output.shape = (bs, sequence_length, input_dim * time_embed_dim)
        else:
            x_output = x
        return x_output

class FFLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding,self).__init__()

    def forward(self, context,y):
        N = context.shape[1]
        N_bs = min(self.bs, context.shape[0])
        
        positions = torch.arange(0,N).repeat(N_bs,self.batch_size,1).to(self.device)
        positions.requires_grad=False
        embm = self.emb_month(context[:,:,1])
        embd = self.emb_day(context[:,:,2])
        embh = self.emb_hour(context[:,:,3])
        embw = self.emb_dow(context[:,:,4])
        embp = self.emb_position(positions[:,:,0])
        
        emb = torch.cat((embm, embd, embh, embw, embp), dim = 2)
        assert self.context_size*self.embed_size==emb.shape[2]

