
from torch import  nn
import torch
from .base import Base
from .utils import QuantileLossMO,Permute, get_device, get_activation
from typing import List, Union
from .vva.minigpt import Block
from .vva.vqvae import VQVAE
import numpy as np
import logging
import math
from torch.nn import functional as F

torch.autograd.set_detect_anomaly(True)

        

class VQVAEA(Base):

    
    def __init__(self, 
                 past_steps:int,
                 future_steps:int,
                 past_channels:int,
                 future_channels:int,
                 hidden_channels:int,
                 embs:List[int],
                 d_model:int,
                 max_voc_size:int,
                 num_layers:int,
                 dropout_rate:float,
                 commitment_cost:float,
                 decay:float,
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

        
        super(VQVAEA, self).__init__()
        self.save_hyperparameters(logger=False)
        self.d_model = d_model
        self.max_voc_size = max_voc_size
        self.future_steps = future_steps
        ##PRIMA VQVAE
        assert out_channels==1, logging.info('Working only for one singal')
        self.vqvae = VQVAE(in_channels=1, hidden_channels=hidden_channels,out_channels=1,num_embeddings= max_voc_size,embedding_dim=d_model,commitment_cost=commitment_cost,decay=decay  )
        
        ##POI GPT


        self.block_size = past_steps//2 + future_steps//2 -1
        self.sentence_length = future_steps//2
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(max_voc_size, d_model),
            wpe = nn.Embedding(self.block_size, d_model),
            drop = nn.Dropout(dropout_rate),
            h = nn.ModuleList([Block( d_model,dropout_rate,n_heads,dropout_rate,self.block_size) for _ in range(num_layers)]), ##care can be different dropouts
            ln_f = nn.LayerNorm(d_model),
            lm_head = nn.Linear(d_model, max_voc_size, bias=False)
        ))
        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        logging.info("number of parameters: %.2fM" % (n_params/1e6,))
        
        self.use_quantiles = False
        self.is_classification = True
        self.optim_config = optim_config
       
        

        
    def configure_optimizers(self):
        
    
        #return torch.optim.Adam(self.vqvae.parameters(), lr=self.optim_config.lr_vqvae,
        #                                   weight_decay=self.optim_config.weight_decay_vqvae)


        return torch.optim.AdamW([
                {'params':self.vqvae.parameters(),'lr':self.optim_config.lr_vqvae,'weight_decay':self.optim_config.weight_decay_vqvae},
                {'params':self.transformer.parameters(),'lr':self.optim_config.lr_gpt,'weight_decay':self.optim_config.weight_decay_gpt},
            ])
       

        
    def gpt(self,tokens):
    
    
        b, t = tokens.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=self.device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(tokens) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.transformer.lm_head(x)
        return logits

    def forward(self, batch):
        
        ##VQVAE
        #current_epoch = self.current_epoch 
        #if current_epoch < 1000:
        #    self.vqvae.train()
        #    loss_gpt = 100
        #else:
        #    self.vqvae.eval()
        idx_target = batch['idx_target'][0]

        #(tensor([194, 163, 174, 176, 160, 168, 175]),
        # tensor([ -1,  -1,  -1, 160, 168, 175, 160]))

           
        data = batch['x_num_past'][:,:,idx_target]
        
        vq_loss, data_recon, perplexity,quantized_x,encodings_x = self.vqvae(data.permute(0,2,1))
        recon_error = F.mse_loss(data_recon.squeeze(), data.squeeze()) 
        loss_vqvae = recon_error + vq_loss
       
        if self.current_epoch > 250:
            with torch.no_grad():
                _, _, _,quantized_y,encodings_y = self.vqvae(batch['y'].permute(0,2,1))
            
            ##GPT
            tokens = torch.cat([encodings_x.argmax(dim=2),encodings_y.argmax(dim=2)[:,0:-1]],1)
            tokens_y = torch.cat([encodings_x.argmax(dim=2)[:,0:-1],encodings_y.argmax(dim=2)],1)
            tokens_y[:,0:encodings_x.shape[1]-1] = -1
            logits = self.gpt(tokens)
            loss_gpt = F.cross_entropy(logits.view(-1, logits.size(-1)),tokens_y.view(-1), ignore_index=-1)

            
            ##adesso devo ricostruire la y perche' e quello che voglio come output
            with torch.no_grad():
                encoding_indices = torch.argmax(logits.reshape(-1,self.max_voc_size), dim=1).unsqueeze(1) ##
                encodings = torch.zeros(encoding_indices.shape[0], self.vqvae._vq_vae._num_embeddings, device=self.device)
                encodings.scatter_(1, encoding_indices, 1)
                quantized = torch.matmul(encodings, self.vqvae._vq_vae._embedding.weight).view(data.shape[0],-1,self.d_model) ##B x L x hidden
                quantized = quantized.permute(0, 2, 1).contiguous()
                y_hat = self.vqvae._decoder(quantized,False).squeeze()[:,-self.future_steps:]
            
            l1_loss = nn.L1Loss()(y_hat,batch['y'].squeeze())

            return y_hat, loss_vqvae+loss_gpt+l1_loss
        else:
            return None, loss_vqvae


    def training_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
        _, loss = self(batch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
        _, loss = self(batch)
        return loss


    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None,num_samples=100):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert do_sample==False,logging.info('NOT IMPLEMENTED YET')
        if do_sample:
            
            idx = idx.repeat(num_samples,1,1)
            for _ in range(max_new_tokens):
                tmp = []
                for i in range(num_samples):
                    idx_cond = idx[i,:,:] if idx.size(2) <= self.block_size else idx[i,:, -self.block_size:]
                    logits = self.gpt(idx_cond)
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
                logits = self.gpt(idx_cond)
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

            return idx

    def inference(self, batch:dict)->torch.tensor:

        idx_target = batch['idx_target'][0]
        data = batch['x_num_past'][:,:,idx_target]
        vq_loss, data_recon, perplexity,quantized_x,encodings_x = self.vqvae(data.permute(0,2,1))
        x = encodings_x.argmax(dim=2)
        inp = x[:, :self.sentence_length]
        # let the model sample the rest of the sequence
        cat = self.generate(inp, self.sentence_length, do_sample=False) # non riesco a gestirla qui :-)
        encoding_indices = cat.flatten().unsqueeze(1) ##
        encodings = torch.zeros(encoding_indices.shape[0], self.vqvae._vq_vae._num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.vqvae._vq_vae._embedding.weight).view(x.shape[0],-1,self.d_model) ##B x L x hidden
        quantized = quantized.permute(0, 2, 1).contiguous()
        y_hat = self.vqvae._decoder(quantized,False).squeeze()[:,-self.future_steps:]

        ## BxLxCx3
        return y_hat.unsqueeze(2).unsqueeze(3)
    
