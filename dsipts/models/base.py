
from torch import optim, nn
import torch
import pickle
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from abc import ABCMeta, abstractmethod
import logging
from .utils import SinkhornDistance


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
        self.initialize = False
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
        
        if self.optim_config is None:
            self.optim_config = {'lr': 5e-05}

        
        if self.optim is None:
            optimizer = optim.Adam(self.parameters(),  **self.optim_config)
            self.initialize = True
        else:
            ##this is strange, pytorch lighening call twice this if autotune is true
            if self.initialize==False:
                self.optim = eval(self.optim)
            print(self.optim)
            optimizer = self.optim(self.parameters(),  **self.optim_config)
            self.initialize = True
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
        return self.compute_loss(batch,y_hat)
    
    def validation_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
        y_hat = self(batch)
        return self.compute_loss(batch,y_hat)


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

    def compute_loss(self,batch,y_hat):
        """
        custom loss calculation
        
        :meta private:
        """

        if self.use_quantiles==False:
            initial_loss = self.loss(y_hat[:,:,:,0], batch['y'])
        else:
            initial_loss = self.loss(y_hat, batch['y'])
        x =  batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        x_start = x[:,-1,idx_target].unsqueeze(1)
        y_persistence = x_start.repeat(1,self.future_steps,1)
        
        if self.loss_type == 'linear_penalization':
            idx = 1 if self.use_quantiles else 0
            persistence_error = self.persistence_weight*(2.0-10.0*torch.clamp( torch.abs((y_persistence-y_hat[:,:,:,idx])/(0.001+torch.abs(y_persistence))),min=0.0,max=0.1))
            loss = torch.mean(torch.abs(y_hat[:,:,:,idx]- batch['y'])*persistence_error)
            #loss = self.persistence_weight*persistence_loss + (1-self.persistence_weight)*mse_loss
            
        elif self.loss_type == 'exponential_penalization':
            idx = 1 if self.use_quantiles else 0
            weights = (1+self.persistence_weight*torch.exp(-torch.abs(y_persistence-y_hat[:,:,:,idx])))
            loss =  torch.mean(torch.abs(y_hat[:,:,:,idx]- batch['y'])*weights)
        
        #elif self.loss_type=='sinkhorn_heavy':
        #    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
        #    if self.use_quantiles==False:
        #        x = y_hat[:,:,:,0]
        #    else:
        #        x = y_hat[:,:,:,1]
        #    loss = sinkhorn.compute(x,batch['y']) +   self.persistence_weight*sinkhorn.compute(x,batch['y'])/(sinkhorn.compute(x,y_persistence)+0.0001)
            
        elif self.loss_type=='sinkhorn':

            sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
            if self.use_quantiles==False:
                x = y_hat[:,:,:,0]
            else:
                x = y_hat[:,:,:,1]
            loss = sinkhorn.compute(x,batch['y'])
            
        elif self.loss_type=='high_order':


            if self.use_quantiles==False:
                x = y_hat[:,:,:,0]
            else:
                x = y_hat[:,:,:,1]
            import pdb
            pdb.set_trace()
            std_real = torch.sqrt(torch.var(batch['y'], dim=2)+ 1e-8)
            std_predict = torch.sqrt(torch.var(x, dim=2)+ 1e-8)
            loss = initial_loss +  self.persistence_weight*torch.mean(std_real-std_predict)
        #elif self.loss_type=='triplet':

        #    triplet = torch.nn.TripletMarginLoss(margin=0.05, p=1)
        #    if self.use_quantiles==False:
        #        x = y_hat[:,:,:,0]
        #    else:
        #        x = y_hat[:,:,:,1]
        #
        #    anchor = x
        #    positive =  batch['y']
        #    negative = y_persistence
        #    loss = triplet(anchor, positive, negative)

        elif self.loss_type=='smape':
            if self.use_quantiles==False:
                x = y_hat[:,:,:,0]
            else:
                x = y_hat[:,:,:,1]
            loss = torch.mean(2*torch.abs(x-batch['y']) / (torch.abs(x)+torch.abs(batch['y'])))
    
        else:
            loss = initial_loss

        return loss