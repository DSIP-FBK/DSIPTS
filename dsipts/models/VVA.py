
from torch import  nn
import torch
from .base import Base
from .utils import QuantileLossMO,Permute, get_device, get_activation
from typing import List, Union
from .vva.minigpt import Block
import numpy as np
import logging
import math
from torch.nn import functional as F

torch.autograd.set_detect_anomaly(True)

        

class VVA(Base):

    
    def __init__(self, 
                 past_steps:int,
                 future_steps:int,
                 past_channels:int,
                 future_channels:int,
                 embs:List[int],
                 d_model:int,
                 max_voc_size:int,
                 token_split: int,
                 num_layers:int,
                 dropout_rate:float,
                 n_heads:int,
                 out_channels:int,
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[int]=[],
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None)->None:
        """ Custom encoder-decoder 
        
        Args:
            past_steps (int):  number of past datapoints used 
            future_steps (int): number of future lag to predict
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            embs (List): list of the initial dimension of the categorical variables
            cat_emb_dim (int): final dimension of each categorical variable
            hidden_RNN (int): hidden size of the RNN block
            num_layers_RNN (int): number of RNN layers
            kind (str): one among GRU or LSTM
            kernel_size (int): kernel size in the encoder convolutional block
            sum_emb (bool): if true the contribution of each embedding will be summed-up otherwise stacked
            out_channels (int):  number of output channels
            activation (str, optional): activation fuction function pytorch. Default torch.nn.ReLU
            remove_last (bool, optional): if True the model learns the difference respect to the last seen point
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            quantiles (List[int], optional): we can use quantile loss il len(quantiles) = 0 (usually 0.1,0.5, 0.9) or L1loss in case len(quantiles)==0. Defaults to [].
            dropout_rate (float, optional): dropout rate in Dropout layers
            use_bn (bool, optional): if true BN layers will be added and dropouts will be removed
            use_glu (bool,optional): use GLU for feature selection. Defaults to True.
            glu_percentage (float, optiona): percentage of features to use. Defaults to 1.0.
            n_classes (int): number of classes (0 in regression)
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.

        """

        
        super(VVA, self).__init__()


        self.block_size = past_steps//token_split + future_steps//token_split -1
        self.save_hyperparameters(logger=False)
        self.sentence_length = future_steps//token_split
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(max_voc_size, d_model),
            wpe = nn.Embedding(self.block_size, d_model),
            drop = nn.Dropout(dropout_rate),
            h = nn.ModuleList([Block( d_model,dropout_rate,n_heads,dropout_rate,self.block_size) for _ in range(num_layers)]), ##care can be different dropouts
            ln_f = nn.LayerNorm(d_model),
        ))
        self.lm_head = nn.Linear(d_model, max_voc_size, bias=False)


        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layers))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))
        
        
        
        
        
        
        
        self.use_quantiles = True
        self.is_classification = True
        self.scheduler_config = scheduler_config
        self.optim_config = optim_config
        self.optim = self.scheduler_config = self.configure_optimizers()

        
    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.optim_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.optim_config.lr, betas=self.optim_config.betas)
        return optimizer

    def compute_loss(self,batch,y_hat):
        """
        custom loss calculation
        
        :meta private:
        """
        return F.cross_entropy(y_hat.view(-1, y_hat.size(-1)), batch['y_emb'].view(-1), ignore_index=-1)


    def forward(self, batch):
        b, t = batch['x_emb'].size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=self.device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(batch['x_emb']) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits


    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None,num_samples=100):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if do_sample:
            idx = idx.repeat(num_samples,1,1)
            for _ in range(max_new_tokens):
                tmp = []
                for i in range(num_samples):
                    idx_cond = idx[i,:,:] if idx.size(2) <= self.block_size else idx[i,:, -self.block_size:]
                    logits = self({'x_emb':idx_cond})
                    logits = logits[:, -1, :] / temperature
                    if top_k is not None:
                        v, _ = torch.topk(logits, top_k)
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1, replacement=True)
                    tmp.append(idx_next)
                tmp = torch.cat(tmp,dim=1).T.unsqueeze(2)
                idx = torch.cat((idx, tmp), dim=2)
            return idx
        else:
            for _ in range(max_new_tokens):
                
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
                # forward the model to get the logits for the index in the sequence
                logits = self({'x_emb':idx_cond})
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # either sample from the distribution or take the most likely element
                _, idx_next = torch.topk(probs, k=1, dim=-1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)

            return idx.unsqueeze(0)

    def inference(self, batch:dict)->torch.tensor:
        x = batch['x_emb'].to(self.device)
        
        # isolate the input pattern alone
        inp = x[:, :self.sentence_length]
        
        # let the model sample the rest of the sequence
        cat = self.generate(inp, self.sentence_length, do_sample=True,num_samples=3) # using greedy argmax, not samplingv ##todo here add sampling
        sol_candidate = cat[:,:, self.sentence_length:] 

       
        return sol_candidate.permute(1,2,0)
    
