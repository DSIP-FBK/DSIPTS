
from torch import  nn,optim
import torch
from .base import  Base
from .utils import get_device, QuantileLossMO,L1Loss
import math
from typing import List
from .d3vae.model import diffusion_generate, denoise_net,pred_net
from .d3vae.embedding import DataEmbedding

from torch.optim.lr_scheduler import StepLR

def copy_parameters(
    net_source: torch.nn.Module,
    net_dest: torch.nn.Module,
    strict= True,
) -> None:
    """
    Copies parameters from one network to another.
    Parameters
    ----------
    net_source
        Input network.
    net_dest
        Output network.
    strict:
        whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    """

    net_dest.load_state_dict(net_source.state_dict(), strict=strict)


class D3VAE(Base):              
    
    def __init__(self,
                 past_channels,
                 past_steps,
                 future_steps,
                 future_channels,
                 embs,
                 out_channels,
                 quantiles,
                 embedding_dimension=32,
                 scale=0.1,
                 hidden_size=64,
                 num_layers=2,
                 dropout_rate=0.1,
                 diff_steps=200,
                 loss_type='kl',
                 beta_end=0.01,
                 beta_schedule='linear',
                 channel_mult = 2,
                 mult=1,
                 num_preprocess_blocks=1,
                 num_preprocess_cells=3,
                 num_channels_enc=16,
                 arch_instance = 'res_mbconv',
                 num_latent_per_group=6,
                 num_channels_dec=16,
                 groups_per_scale=2,
                 num_postprocess_blocks=1,
                 num_postprocess_cells=2,
                 beta_start=0,
                 
                 freq='h',
                 optim_config=None,
                 scheduler_config=None,
                 )->None:
     
        input_dim = past_channels
        sequence_length = past_steps
        prediction_length = future_steps
        target_dim = out_channels
        ##pytotch lightening stuff
        self.save_hyperparameters(logger=False)
        
        super().__init__()
        
        
        self.gen_net = diffusion_generate(target_dim,embedding_dimension,prediction_length,sequence_length,scale,hidden_size,num_layers,dropout_rate,diff_steps,loss_type,beta_end,beta_schedule, channel_mult,mult,
                 num_preprocess_blocks,num_preprocess_cells,num_channels_enc,arch_instance,num_latent_per_group,num_channels_dec,groups_per_scale,num_postprocess_blocks,num_postprocess_cells).to(self.device)
        
        self.denoise_net = denoise_net(target_dim,embedding_dimension,prediction_length,sequence_length,scale,hidden_size,num_layers,dropout_rate,diff_steps,loss_type,beta_end,beta_schedule, channel_mult,mult,
                 num_preprocess_blocks,num_preprocess_cells,num_channels_enc,arch_instance,num_latent_per_group,num_channels_dec,groups_per_scale,num_postprocess_blocks,num_postprocess_cells,beta_start,input_dim,freq,embs).to(self.device)
        self.diff_step = diff_steps
        self.pred_net = pred_net(target_dim,embedding_dimension,prediction_length,sequence_length,scale,hidden_size,num_layers,dropout_rate,diff_steps,loss_type,beta_end,beta_schedule, channel_mult,mult,
                 num_preprocess_blocks,num_preprocess_cells,num_channels_enc,arch_instance,num_latent_per_group,num_channels_dec,groups_per_scale,num_postprocess_blocks,num_postprocess_cells,beta_start,input_dim,freq,embs).to(self.device)
        #self.embedding = DataEmbedding(input_dim, embedding_dimension, freq,dropout_rate)
        
        self.psi = 0.5
        self.gamma = 0.01
        self.lambda1 = 1.0
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config

  
        self.use_quantiles = False
        self.loss = nn.MSELoss()
        
    def configure_optimizers(self):
        """
        Each model has optim_config and scheduler_config
        
        :meta private:
        """
        optimizer = optim.Adam(self.denoise_net.parameters(),  **self.optim_config)
        self.lr = self.optim_config['lr']
        if self.scheduler_config is not None:
            scheduler = StepLR(optimizer,**self.scheduler_config)
            return [optimizer], [scheduler]
        else:
            return optimizer
        
        
        
    def training_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
        sample, y_noisy,recon,loss2,total_c = self(batch)
        
        mse_loss = self.loss(sample, y_noisy)
        loss1 = - torch.mean(torch.sum(recon, dim=[1, 2, 3]))
        loss = loss1*self.psi + loss2*self.lambda1 + mse_loss - self.gamma*total_c
        return loss
        
    def validation_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
        '''
        copy_parameters(self.denoise_net, self.pred_net)
        batch_x = batch['x_num_past'].to(self.device)
        batch_x_mark = batch['x_cat_past'].to(self.device)
        batch_y = batch['y'].to(self.device)

        
        _, out, _, _ = self.pred_net(batch_x, batch_x_mark)
        mse = self.loss(out.squeeze(1), batch_y)
        '''
        mse = torch.tensor(0.0)
        return mse

        
        
    def forward(self,batch:dict)->torch.tensor:
        
        B = batch['x_num_past'].shape[0]

        
        t = torch.randint(0, self.diff_step, (B,)).long().to(self.device)
        
        batch_x = batch['x_num_past'].to(self.device)
        x_mark = batch['x_cat_past'].to(self.device)
        batch_y = batch['y'].to(self.device)
        
        output, y_noisy, total_c, _, loss2 = self.denoise_net(batch_x, x_mark, batch_y, t)
        recon = output.log_prob(y_noisy)
        sample = output.sample()
        
        
        return sample,y_noisy,recon,loss2,total_c

   
   
   
    

    
    def inference(self, batch:dict)->torch.tensor:
        """Care here, we need to implement it because for predicting the N-step it will use the prediction at step N-1. TODO fix if because I did not implement the
        know continuous variable presence here

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        copy_parameters(self.denoise_net, self.pred_net)

        batch_x = batch['x_num_past'].float().to(self.device)
        batch_x_mark = batch['x_cat_past'].to(self.device)
        _, out, _, _ = self.pred_net(batch_x, batch_x_mark)
        #import pdb
        #pdb.set_trace()
        return torch.permute(out, (0,2,1,3))