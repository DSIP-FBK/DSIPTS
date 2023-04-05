
from torch import nn
import torch
import pytorch_lightning as pl
from .base import  Base
from .utils import  get_device,L1Loss
from typing import List



class Persistent(Base):

    
    def __init__(self, 
                 future_steps:int,
                 past_steps:int,
                 optim_config:dict=None,
                 scheduler_config:dict=None)->None:
        """Persistent model propagatinng  last observed values

        Args:
          
            future_steps (int): number of future lag to predict   
            past_steps (int): number of future lag to predict. Useless but needed for the other stuff

            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None. Usless for this model
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None. Usless for this model
        """
        
    
        super(Persistent, self).__init__()
        self.save_hyperparameters(logger=False)
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.loss = L1Loss()
                               

    def forward(self, batch):
        """It is mandatory to implement this method

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        x =  batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        x_start = x[:,-1,idx_target].unsqueeze(1)
        #this is B,1,C
        
        #[B,L,C,1] remember the outoput size
        res = x_start.repeat(1,self.future_steps,1).unsqueeze(3)
        
        return res
    