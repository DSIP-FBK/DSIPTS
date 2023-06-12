
from torch import optim, nn
import torch
import pickle
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from abc import ABCMeta, abstractmethod
import logging



class Base(pl.LightningModule):
    @abstractmethod
    def __init__(self):
        """
        This is the basic model, each model implemented must overwrite the init method and the forward method. The inference step is optional, by default it uses the forward method but for recurrent 
        network you should implement your own method
        """
        
        super(Base, self).__init__()
        self.save_hyperparameters(logger=False)
        self.count_epoch = 0
    @abstractmethod
    def forward(self, batch:dict)-> torch.tensor:
        """Forlward method used during the training loop

        Args:
            batch (dict): the batch structure. The keys are:
                y : the target variable(s). This is always present
                x_num_past: the numerical past variables. This is always present
                x_num_future: the numerical future variables
                x_cat_past: the categorical past variables
                x_cat_future: the categorical future variables
                idx_target: index of target features in the past array
            

        Returns:
            torch.tensor: output of the mode;
        """
        return None
    
    def inference(self, batch:dict)->torch.tensor:
        """Usually it is ok to return the output of the forward method but sometimes not (e.g. RNN)

        Args:
            batch (dict): batch

        Returns:
            torch.tensor: result
        """
        return self(batch)
        
    def configure_optimizers(self):
        """
        Each model has optim_config and scheduler_config
        
        :meta private:
        """
        #import pdb;pdb.set_trace()
        if self.optim is None:
            optimizer = optim.Adam(self.parameters(),  **self.optim_config)
        else:
            self.optim = eval(self.optim)
            print(self.optim)
            optimizer = self.optim(self.parameters(),  **self.optim_config)
            
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
        y_hat = self(batch)
        loss = self.loss(y_hat, batch['y'].to(self.device))
        #self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
        y_hat = self(batch)

        loss = self.loss(y_hat, batch['y'].to(self.device))
        #self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outs):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
        #print('logging val')
        #import pdb;pdb.set_trace()
        loss = torch.stack(outs).mean()
        self.log("val_loss", loss.item(),sync_dist=True)
        logging.info(f'Epoch: {self.count_epoch}, validation loss: {loss.item():.4f}')

    def training_epoch_end(self, outs):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
        #print('logging train')
        #import pdb;pdb.set_trace()
        loss = sum(outs['loss'] for outs in outs) / len(outs)
        self.log("train_loss", loss.item(),sync_dist=True)
        self.count_epoch+=1
        logging.info(f'Epoch: {self.count_epoch}, train loss: {loss.item():.4f}')

