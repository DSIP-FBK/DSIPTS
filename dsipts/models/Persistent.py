
from torch import nn
from .base import  Base
from .utils import L1Loss
from ..data_structure.utils import beauty_string
from .utils import  get_scope

class Persistent(Base):
    handle_multivariate = True
    handle_future_covariates = False
    handle_categorical_variables = False
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables)
    beauty_string(description,'info',True)
    
    def __init__(self, 
                 future_steps:int,
                 past_steps:int,
                 loss_type:str=None,#not used but needed
                 persistence_weight:float=0.1,#not used but needed
                 optim_config:dict=None,
                 scheduler_config:dict=None,
                 **kwargs)->None:
        """Persistent model propagatinng  last observed values

        Args:
          
            future_steps (int): number of future lag to predict   
            past_steps (int): number of future lag to predict. Useless but needed for the other stuff

            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None. Usless for this model
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None. Usless for this model
        """
        
    
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.optim = None
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.loss = L1Loss()
        self.fake = nn.Linear(1,1)
        self.use_quantiles = False
        self.loss_type = 'l1'
        self.loss = nn.L1Loss()
        
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
    