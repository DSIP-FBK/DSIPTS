
from torch import optim, nn
import torch
import pickle
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from abc import ABCMeta, abstractmethod



class Base(pl.LightningModule):
    @abstractmethod
    def __init__(self):
        super(Base, self).__init__()
        self.save_hyperparameters(logger=False)
        
    @abstractmethod
    def forward(self, batch):
       return None
    
    def inference(self, batch):
        return self(batch)
        
    ##questi metodi li posso sovrascrivere o ereditare :-)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),  **self.optim_config)
        self.lr = self.optim_config['lr']
        if self.scheduler_config is not None:
            scheduler = StepLR(optimizer,**self.scheduler_config)
            return [optimizer], [scheduler]
        else:
            return optimizer


    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss(y_hat, batch['y'].to(self.device))
        #self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss(y_hat, batch['y'].to(self.device))
        #self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outs):
        #print('logging val')
        #import pdb;pdb.set_trace()
        loss = torch.stack(outs).mean()
        self.log("val_loss", loss.item(),sync_dist=False)
        

    def training_epoch_end(self, outs):
        #print('logging train')
        #import pdb;pdb.set_trace()
        loss = sum(outs['loss'] for outs in outs) / len(outs)
        self.log("train_loss", loss.item(),sync_dist=False)
