
from torch import optim
import torch
try:
    import lightning.pytorch as pl
except:
    import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from abc import  abstractmethod
from .utils import SinkhornDistance, SoftDTWBatch,PathDTWBatch,pairwise_distances
from ..data_structure.utils import beauty_string
from .samformer.utils import SAM
from .utils import  get_scope
import numpy as np
#from aim import Image
import matplotlib.pyplot as plt
from typing import List, Union
from .utils import QuantileLossMO, CPRS
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RMSprop   

def standardize_momentum(x,order):
    mean = torch.mean(x,1).unsqueeze(1).repeat(1,x.shape[1],1)
    num = torch.pow(x-mean,order).mean(axis=1)
    #den = torch.sqrt(torch.pow(x-mean,2).mean(axis=1)+1e-8)
    #den = torch.pow(den,order)

    return num#/den


def dilate_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
	loss_shape = softdtw_batch(D,gamma)
	
	path_dtw = PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega =  pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = alpha*loss_shape+ (1-alpha)*loss_temporal
	return loss#, loss_shape, loss_temporal


class Base(pl.LightningModule):
    
    ############### SET THE PROPERTIES OF THE ARCHITECTURE##############
    
    handle_multivariate = False
    handle_future_covariates = False
    handle_categorical_variables = False
    handle_quantile_loss = False
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    #####################################################################

    @abstractmethod
    def __init__(self,verbose:bool,
                 past_steps:int,
                 future_steps:int,
                 past_channels:int,
                 future_channels:int,
                 out_channels:int,
                 embs_past:List[int],
                 embs_fut:List[int],
                 n_classes:int=0,
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[int]=[],
                 reduction_mode:str = 'mean',
                 use_classical_positional_encoder:bool=False,
                 emb_dim: int=16,

                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None,
                prediction_channel_indices = None,
                exogenous_channel_indices_cont= None,
                exogenous_channel_indices_cat= None):
        """
        This is the basic model, each model implemented must overwrite the init method and the forward method.
        The inference step is optional, by default it uses the forward method but for recurrent 
        network you should implement your own method
        
        Args:
            verbose (bool): Flag to enable verbose logging.
            past_steps (int): Number of past time steps to consider.
            future_steps (int): Number of future time steps to predict.
            past_channels (int): Number of channels in the past input data.
            future_channels (int): Number of channels in the future input data.
            out_channels (int): Number of output channels.
            embs_past (List[int]): List of embedding dimensions for past data.
            embs_fut (List[int]): List of embedding dimensions for future data.
            n_classes (int, optional): Number of classes for classification. Defaults to 0.
            persistence_weight (float, optional): Weight for persistence in loss calculation. Defaults to 0.0.
            loss_type (str, optional): Type of loss function to use ('l1' or 'mse'). Defaults to 'l1'.
            quantiles (List[int], optional): List of quantiles for quantile loss. Defaults to an empty list.
            reduction_mode (str, optional): Mode for reduction for categorical embedding layer ('mean', 'sum', 'none'). Defaults to 'mean'.
            use_classical_positional_encoder (bool, optional): Flag to use classical positional encoding or using embedding layer also for the positions. Defaults to False.
            emb_dim (int, optional): Dimension of categorical embeddings. Defaults to 16.
            optim (Union[str, None], optional): Optimizer type. Defaults to None.
            optim_config (dict, optional): Configuration for the optimizer. Defaults to None.
            scheduler_config (dict, optional): Configuration for the learning rate scheduler. Defaults to None.
        
        Raises:
            AssertionError: If the number of quantiles is not equal to 3 when quantiles are provided.
            AssertionError: If the number of output channels is not 1 for classification tasks.
        """

   
        beauty_string('V2','block',True)
        super(Base, self).__init__()
        
        
        
        self.save_hyperparameters(logger=False)
        self.count_epoch = 0
        self.initialize = False
        self.train_loss_epoch = -100.0
        self.verbose = verbose
        self.name = self.__class__.__name__
        #self.train_epoch_metrics = 0
        #self.validation_epoch_metrics = 0
        
        self.register_buffer("train_epoch_metrics", torch.tensor(0.0))
        self.register_buffer("validation_epoch_metrics", torch.tensor(0.0))
        self.register_buffer("train_epoch_count", torch.tensor(0))
        self.register_buffer("validation_epoch_count", torch.tensor(0))
        
        self.use_quantiles = True if len(quantiles)>0 else False
        self.quantiles =  quantiles
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.loss_type = loss_type
        self.persistence_weight = persistence_weight 
        self.use_classical_positional_encoder = use_classical_positional_encoder
        self.reduction_mode = reduction_mode
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.embs_past = embs_past
        self.embs_fut = embs_fut
        self.past_channels = past_channels
        self.future_channels = future_channels
        self.emb_dim = emb_dim
        self.out_channels = out_channels
        self.n_classes = n_classes
        if n_classes==0:
            self.is_classification = False
            if len(self.quantiles)>0:
                if self.loss_type=='cprs':
                    self.use_quantiles = True
                    self.mul = len(self.quantiles)
                    self.loss = CPRS(alpha=self.persistence_weight)
                else:
                    assert len(self.quantiles)==3, beauty_string('ONLY 3 quantiles premitted','info',True)
                    self.use_quantiles = True
                    self.mul = len(self.quantiles)
                    self.loss = QuantileLossMO(quantiles)
            else:
                self.use_quantiles = False
                self.mul = 1
                if self.loss_type == 'mse':
                    self.loss = nn.MSELoss()
                else:
                    self.loss = nn.L1Loss()
        else:
            self.is_classification = True
            self.use_quantiles = False
            self.mul = n_classes
            self.loss = torch.nn.CrossEntropyLoss()
            assert self.out_channels==1, "Classification require only one channel"


        self.future_steps = future_steps
        self.return_additional_loss = False
        beauty_string(self.description,'info',True)

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
        
        if self.loss_type=='cprs':
            tmp = self(batch)
            tmp = torch.quantile(tmp, torch.tensor([0.05, 0.5, 0.95]), dim=-1).permute(1,2,3,0)
            return tmp
            #return tmp.mean(axis=-1).unsqueeze(-1)
        
        return self(batch)
        
    def configure_optimizers(self):
        """
        Each model has optim_config and scheduler_config
        
        :meta private:
        """

        self.has_sam_optim = False
        if self.optim_config is None:
            self.optim_config = {'lr': 5e-05}

        
        if self.optim is None:
            optimizer = Adam(self.parameters(),  **self.optim_config)
            self.initialize = True
            
        else:
            if self.initialize is False:
                if self.optim=='SAM':
                    self.has_sam_optim = True
                    self.automatic_optimization = False
                    self.my_step = 0

                else:
                    self.optim = eval(self.optim)
                    self.has_sam_optim = False
                    self.automatic_optimization = True

            beauty_string(self.optim,'',self.verbose)
            if self.has_sam_optim:
                optimizer = SAM(self.parameters(), base_optimizer=Adam, **self.optim_config)
            else:
                optimizer = self.optim(self.parameters(),  **self.optim_config)
            beauty_string(optimizer,'',self.verbose)
            self.initialize = True
        self.lr = self.optim_config['lr']  ##CHECK THISs
        if self.scheduler_config is not None:
            scheduler = StepLR(optimizer,**self.scheduler_config)
            #return [optimizer], [scheduler]
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch", # Adjust to 'step' if your StepLR is per-step
                },
            }
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
        
        #loss = self.compute_loss(batch,y_hat)
        #import pdb
        #pdb.set_trace()

        if self.has_sam_optim:
            
            opt = self.optimizers()
            def closure():
                opt.zero_grad()
                if self.return_additional_loss:
                    y_hat,score = self(batch)
                    loss = self.compute_loss(batch,y_hat) + score
                else:
                    y_hat = self(batch)
                    loss = self.compute_loss(batch,y_hat)
                self.manual_backward(loss)
                return loss

            opt.step(closure)
            if self.return_additional_loss:
                y_hat,score = self(batch)
                loss = self.compute_loss(batch,y_hat)+score
            else:
                y_hat = self(batch)
                loss = self.compute_loss(batch,y_hat)
            
            #opt.first_step(zero_grad=True)

            #y_hat = self(batch)
            #loss = self.compute_loss(batch, y_hat)
            #self.my_step+=1
            #self.manual_backward(loss,retain_graph=True)
            #opt.second_step(zero_grad=True)
            #self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            #self.log("global_step",  self.my_step, on_step=True)  # Correct way to log

   
            #self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment("optimizer")
        else:
            if self.return_additional_loss:
                y_hat,score = self(batch)
                loss = self.compute_loss(batch,y_hat)+score
            else:
                y_hat = self(batch)
                loss = self.compute_loss(batch,y_hat)
            
        self.train_epoch_metrics+=loss.detach()
        self.train_epoch_count +=1
        return loss

    def validation_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff
        
        :meta private:
        """
  
        if self.return_additional_loss:
            y_hat,score = self(batch)
        else:
            y_hat = self(batch)
            score = 0
        #log_this_batch = (batch_idx == 0) and (self.count_epoch % int(max(self.trainer.max_epochs / 100,1)) == 1)

        #if log_this_batch:
            #track the predictions! We can do better than this but maybe it is better to firstly update pytorch-lightening 
        self._val_outputs=[{
                "y": batch['y'].detach().cpu(),
                "y_hat": y_hat.detach().cpu()
            }]               
        self.validation_epoch_metrics+= (self.compute_loss(batch,y_hat)+score).detach()
        self.validation_epoch_count+=1
        return None

    def on_validation_start(self):
        # reset buffer each epoch
        self._val_outputs = []

    def on_validation_epoch_end(self):
        """
        pythotrch lightening stuff
        
        :meta private:
        """   

        if (len(self._val_outputs)>0) & (self.trainer.max_epochs>0):
            ys = torch.cat([o["y"] for o in self._val_outputs])
            y_hats = torch.cat([o["y_hat"] for o in self._val_outputs])
            if self.use_quantiles:
                idx = 1
            else:
                idx = 0
            for i in range(ys.shape[2]):
                real =  ys[0,:,i].cpu().detach().numpy()
    
                pred =  y_hats[0,:,i,idx].cpu().detach().numpy()
                fig, ax = plt.subplots(figsize=(7,5))  
                ax.plot(real,'o-',label='real')
                ax.plot(pred,'o-',label='pred')
                ax.legend()
                ax.set_title(f'Channel {i} first element first batch validation {int(100*self.count_epoch/self.trainer.max_epochs)}%')
                #try:
                #    self.logger.experiment.track(Image(fig), name='cm_training_end')
                #except:
                #    beauty_string('AIM NOT USED','info',self.verbose)
                #    pass ##no aim probably
                #self.log(f"example_{i}", np.stack([real, pred]).T,sync_dist=True)
                plt.close(fig) 
            

        avg = self.validation_epoch_metrics/self.validation_epoch_count

        self.validation_epoch_metrics.zero_()
        self.validation_epoch_count.zero_()
        self.log("val_loss", avg,sync_dist=True)
        beauty_string(f'Epoch: {self.count_epoch} train error: {self.train_loss_epoch:.4f} validation loss: {avg:.4f}','info',self.verbose)

    def on_train_epoch_end(self):

        """
        pythotrch lightening stuff
        
        :meta private:
        """

        avg = self.train_epoch_metrics/self.train_epoch_count
        self.log("train_loss", avg,sync_dist=True)
        self.count_epoch+=1    
        self.train_epoch_metrics.zero_()
        self.train_epoch_count.zero_()
        self.train_loss_epoch = avg

    def compute_loss(self,batch,y_hat):
        """
        custom loss calculation
        
        :meta private:
        """
        if self.loss_type=='cprs':
            return self.loss(y_hat,batch['y'])

        if self.loss_type=='long_lag':

            batch_size,width,n_variables = batch['y'].shape
            tmp = torch.abs(y_hat[:,:,:,0]-batch['y'])*torch.linspace(1,self.persistence_weight,width).view(1,width,1).repeat(batch_size,1,n_variables)
            return tmp.mean()



        if self.use_quantiles is False:
            initial_loss = self.loss(y_hat[:,:,:,0], batch['y'])
        else:
            initial_loss = self.loss(y_hat, batch['y'])
            
        if  self.loss_type in ['mse','l1']:
            return initial_loss
        
        x =  batch['x_num_past'].to(self.device)
        
        idx_target = batch['idx_target'][0]
        
        if idx_target is None:
            beauty_string(f'Can not compute non-standard loss for non autoregressive models, if you want to use custom losses please add check=True wile initialize the time series object','info',self.verbose)
            return initial_loss
        x_start = x[:,-1,idx_target].unsqueeze(1)
        y_persistence = x_start.repeat(1,self.future_steps,1)
        
        ##generally you want to work without quantile loss
        if self.use_quantiles is False:
            x = y_hat[:,:,:,0]
        else:
            x = y_hat[:,:,:,1]

        
        if self.loss_type == 'linear_penalization':
            # persistence_weight in [0,1]: 0 = no penalization, 1 = max.
            # closeness in [0,1]: 1 when x exactly matches the persistence
            # baseline (max distance 2 on a [-1,1]-clamped domain), 0 when it's
            # maximally far from it. weight in [1,1+pw] replaces the old
            # log10-clamp formula, which was flat (no-op) for pw<0.316 and went
            # negative (unbounded, sign-flipped loss) for pw>10.
            pw = min(max(self.persistence_weight, 0.0), 1.0)
            x_c = torch.clamp(x, -1.0, 1.0)
            y_persistence_c = torch.clamp(y_persistence, -1.0, 1.0)
            closeness = torch.clamp(1.0 - torch.abs(y_persistence_c - x_c) / 2.0, 0.0, 1.0)
            weight = 1.0 + pw*closeness
            loss = torch.mean(torch.abs(x - batch['y'])*weight)

        elif self.loss_type == 'mda':
            # mda (mean directional-agreement error) is in [0,2] by construction
            # (mean of a +-1/0 product, subtracted from 1); normalize by its own
            # range instead of the previous arbitrary /10 to land pw in [0,1].
            pw = min(max(self.persistence_weight, 0.0), 1.0)
            mda = (1-torch.mean( torch.sign(torch.diff(x,axis=1))*torch.sign(torch.diff(batch['y'],axis=1))))
            penalty = torch.clamp(mda/2.0, 0.0, 1.0)
            loss = initial_loss + pw*penalty

        elif self.loss_type == 'exponential_penalization':
            # weight in [1,1+pw]: clamping x/y_persistence to [-1,1] keeps the
            # exp() decay meaningful regardless of the data's actual scale.
            pw = min(max(self.persistence_weight, 0.0), 1.0)
            x_c = torch.clamp(x, -1.0, 1.0)
            y_persistence_c = torch.clamp(y_persistence, -1.0, 1.0)
            weights = (1.0+pw*torch.exp(-torch.abs(y_persistence_c-x_c)))
            loss =  torch.mean(torch.abs(x- batch['y'])*weights)


        elif self.loss_type=='sinkhorn':
            sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
            loss = sinkhorn.compute(x,batch['y'])

        elif self.loss_type == 'additive_iv':
            # persistence_weight in [0,1]: 0 = no penalization, 1 = max.
            # Bounded assuming x,y are scaled to [-1,1]: std of a [-1,1]-bounded
            # variable is itself in [0,1], so |x_std-std| is already unit-scale;
            # the clamp guards against predictions drifting outside that range.
            pw = min(max(self.persistence_weight, 0.0), 1.0)
            std = torch.sqrt(torch.var(torch.clamp(batch['y'], -1.0, 1.0), dim=(1))+ 1e-8) ##--> BSxChannel
            x_std = torch.sqrt(torch.var(torch.clamp(x, -1.0, 1.0), dim=(1))+ 1e-8)
            penalty = torch.clamp(torch.abs(x_std-std), 0.0, 1.0)
            loss = torch.mean( torch.abs(x-batch['y']).mean(axis=1).flatten()) + pw*penalty.mean()

        elif self.loss_type == 'multiplicative_iv':
            # Same penalty as additive_iv but scales initial_loss by (1+pw*penalty)
            # in [1,2] instead of adding it, so it never collapses the fit term to 0.
            pw = min(max(self.persistence_weight, 0.0), 1.0)
            std = torch.sqrt(torch.var(torch.clamp(batch['y'], -1.0, 1.0), dim=(1))+ 1e-8) ##--> BSxChannel
            x_std = torch.sqrt(torch.var(torch.clamp(x, -1.0, 1.0), dim=(1))+ 1e-8)
            penalty = torch.clamp(torch.abs(x_std-std), 0.0, 1.0)
            l1 = torch.abs(x-batch['y']).mean(axis=1)
            loss = torch.mean(l1*(1.0+pw*penalty))
        elif self.loss_type=='global_iv':
            pw = min(max(self.persistence_weight, 0.0), 1.0)
            std_real = torch.sqrt(torch.var(torch.clamp(batch['y'], -1.0, 1.0), dim=(0,1))+ 1e-8)
            std_predict = torch.sqrt(torch.var(torch.clamp(x, -1.0, 1.0), dim=(0,1))+ 1e-8)
            penalty = torch.clamp(torch.abs(std_real-std_predict), 0.0, 1.0)
            loss = initial_loss + pw*penalty.mean()

        elif self.loss_type=='smape':
            loss = torch.mean(2*torch.abs(x-batch['y']) / (0.0000001+torch.abs(x)+torch.abs(batch['y'])))
            
        elif self.loss_type=='triplet':
            # Manual, channel-count-independent triplet penalty: push x toward
            # y and away from y_persistence. d_pos/d_neg are per-channel-mean L1
            # distances, each bounded to [0,2] on a [-1,1]-clamped domain, so
            # the margin-relu'd difference is bounded to [0,2] -> /2 lands it in
            # [0,1]. (nn.TripletMarginLoss's default p=1 distance sums over the
            # channel dim instead of averaging, so its scale grew with channel
            # count.)
            pw = min(max(self.persistence_weight, 0.0), 1.0)
            x_c = torch.clamp(x, -1.0, 1.0)
            y_c = torch.clamp(batch['y'], -1.0, 1.0)
            y_persistence_c = torch.clamp(y_persistence, -1.0, 1.0)
            d_pos = torch.abs(x_c - y_c).mean(dim=-1)
            d_neg = torch.abs(x_c - y_persistence_c).mean(dim=-1)
            margin = 0.01
            penalty = torch.clamp((d_pos - d_neg + margin)/2.0, 0.0, 1.0).mean()
            loss = initial_loss + pw*penalty


        elif self.loss_type=='high_order':
            # Each central moment of order i is normalized by its own bound
            # (2**(i+1), from |X-mean|<=2 on a [-1,1]-clamped variable) so
            # order-4 doesn't dominate order-2 the way it did with a shared scale.
            pw = min(max(self.persistence_weight, 0.0), 1.0)
            y_clamped = torch.clamp(batch['y'], -1.0, 1.0)
            x_clamped = torch.clamp(x, -1.0, 1.0)
            penalty = 0.0
            for i in range(2,5):
                mom_real = standardize_momentum(y_clamped,i)
                mom_pred = standardize_momentum(x_clamped,i)
                penalty = penalty + torch.clamp(torch.abs(mom_real-mom_pred)/(2.0**(i+1)), 0.0, 1.0).mean()
            loss = initial_loss + pw*(penalty/3.0)

        elif self.loss_type=='dilated':
            #BxLxCxMUL
            # persistence_weight now mixes base L1 loss vs. the DILATE loss
            # (0=off/pure L1, 1=full DILATE), decoupled from DILATE's internal
            # shape/temporal balance, which is fixed at alpha=0.5.
            pw = min(max(self.persistence_weight, 0.0), 1.0)
            alpha = 0.5
            gamma = 0.01
            dilate = 0
            ##no multichannel here
            for i in range(y_hat.shape[2]):
                dilate += dilate_loss( batch['y'][:,:,i:i+1],x[:,:,i:i+1], alpha, gamma, y_hat.device)
            dilate = dilate / y_hat.shape[2]
            loss = (1.0-pw)*initial_loss + pw*dilate
            
        elif self.loss_type=='huber':
            # persistence_weight in [0,1] mixes base L1 loss vs. a Huber loss
            # with a fixed delta=1.0 (0=off/pure L1, 1=full Huber), decoupled
            # from Huber's own delta. Previously pw *was* delta directly, so
            # the default pw=0.0 silently zeroed the loss entirely (HuberLoss
            # collapses to a constant 0 when delta=0). This also drops the
            # reshape to y_hat.reshape(BS,-1), which compared the full
            # (BS,T,C,MUL) tensor against batch['y'] and broke shape-wise
            # whenever quantiles were enabled (MUL>1); x is already the
            # correctly-sliced (BS,T,C) prediction computed above.
            pw = min(max(self.persistence_weight, 0.0), 1.0)
            huber_fn = torch.nn.HuberLoss(reduction='mean', delta=1.0)
            loss = (1.0-pw)*initial_loss + pw*huber_fn(x, batch['y'])

        else:
            loss = initial_loss
        return loss
