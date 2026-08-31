
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


MDA_TAU        = 0.5    ##soft sign temperature for the mda penalty, in sigma units
REWEIGHT_GAIN  = 4.0    ##contrast of the linear/exponential reweighting family
TRIPLET_MARGIN = 0.01   ##hinge margin for the triplet penalty
CALIBRATION = {}


def central_moment(x,order):
    """Central moment of a given order along the time axis --> BSxChannel

    :meta private:
    """
    return torch.pow(x-torch.mean(x,1,keepdim=True),order).mean(dim=1)


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

        Every non standard loss follows one contract::

            L(pw) = (base + pw*P)/(1+pw)

        where ``pw`` is ``persistence_weight`` clamped to [0,1], ``base`` is the
        reconstruction error in units of the per channel target sigma and ``P`` is a
        dimensionless O(1) penalty. ``pw=0`` gives exactly the standard loss, ``pw=1``
        puts penalty and reconstruction on equal footing (the maximum sane intensity).
        The ``/(1+pw)`` cap is load bearing: without it the reconstruction term
        disappears at pw=1 and penalties with no magnitude anchor diverge.

        ``cprs`` and ``long_lag`` are outside the contract, they interpret
        persistence_weight on their own scale. See
        loss_normalization_wip/PROPOSED_CHANGES.md

        :meta private:
        """
        if self.loss_type=='cprs':
            ##outside the contract: persistence_weight is the CRPS alpha here
            return self.loss(y_hat,batch['y'])

        if self.loss_type=='long_lag':
            ##outside the contract: persistence_weight is the weight of the LAST lag
            batch_size,width,n_variables = batch['y'].shape
            tmp = torch.abs(y_hat[:,:,:,0]-batch['y'])*torch.linspace(1,self.persistence_weight,width,device=y_hat.device).view(1,width,1).repeat(batch_size,1,n_variables)
            return tmp.mean()

        if self.use_quantiles is False:
            initial_loss = self.loss(y_hat[:,:,:,0], batch['y'])
        else:
            initial_loss = self.loss(y_hat, batch['y'])

        if  self.loss_type in ['mse','l1']:
            return initial_loss

        ##non standard losses are point forecast only. QuantileLossMO sums over the
        ##horizon instead of averaging it, so it is 0.5*future_steps bigger than L1:
        ##adding an O(1) penalty to it dilutes the penalty ~32x at future_steps=64.
        ##Refuse the combination explicitly instead of silently diluting it
        if self.use_quantiles:
            beauty_string(f'Non standard loss {self.loss_type} is not supported together with quantiles, falling back to the plain quantile loss. Use quantiles=[] if you want {self.loss_type}','info',self.verbose)
            return initial_loss

        y = batch['y']
        x = y_hat[:,:,:,0]

        if self.loss_type=='smape':
            ##outside the contract: the denominator |x|+|y| goes to zero on zero centred
            ##data, so this is only meaningful on a strictly positive unscaled target
            if (y.min()<0) and (getattr(self,'_smape_warned',False) is False):
                self._smape_warned = True
                beauty_string('smape is undefined for targets that are not strictly positive (the denominator |x|+|y| goes to zero), use it only on a positive unscaled target','info',self.verbose)
            return torch.mean(2*torch.abs(x-y) / (0.0000001+torch.abs(x)+torch.abs(y)))

        ##per channel target scale, detached, == 1 under StandardScaler. This replaces
        ##the old clamp(-1,1): clamp has zero gradient outside its range and 32.6% of
        ##standardized points have |y|>1, so it deleted the gradient on exactly the
        ##extreme events these penalties exist to capture, and it was not scale
        ##equivariant (data x10 blew the cross loss spread from 43.6x to 378.7x)
        s = y.std(dim=(0,1), keepdim=True).detach()
        s = torch.where(s>1e-6, s, torch.ones_like(s))   ##dead channel --> sigma 1
        base = (torch.abs(x-y)/s).mean()

        ##persistence_weight is THE knob: 0 = standard loss, 1 = max penalization.
        ##NB we return `base` and not `initial_loss`: they differ only by the constant
        ##factor sigma (they are identical under StandardScaler) but returning the
        ##un-normalized loss here would make the loss magnitude jump by sigma between
        ##pw=0 and pw>0, which breaks the comparability of a pw sweep under DummyScaler
        pw = float(min(max(self.persistence_weight, 0.0), 1.0))
        if pw == 0.0:
            return base

        ##NB data_structure.py converts an empty idx_target_future to None but NOT an
        ##empty idx_target, so when the targets are not among the past variables this
        ##arrives as an EMPTY array, not as None: the old `is None` guard never fired and
        ##y_persistence came out with 0 channels, crashing on the broadcast below
        idx_target = batch['idx_target'][0] if 'idx_target' in batch else None
        if (idx_target is None) or (len(idx_target)==0):
            beauty_string(f'Can not compute non-standard loss for non autoregressive models, if you want to use custom losses please add check_past=True while initialize the time series object','info',self.verbose)
            return base
        x_past = batch['x_num_past'].to(self.device)
        y_persistence = x_past[:,-1,idx_target].unsqueeze(1).repeat(1,self.future_steps,1)

        if self.loss_type == 'linear_penalization':
            ##up weight the reconstruction error where the prediction hugs persistence.
            ##This family only REWEIGHTS the same L1 term, so its gradient is nearly
            ##collinear with base and it needs an explicit contrast gain to be felt
            c = torch.clamp(1.0-torch.abs(y_persistence-x)/(2.0*s), 0.0, 1.0)
            w = torch.exp(REWEIGHT_GAIN*c)
            P = ((torch.abs(x-y)/s)*w).sum()/(w.sum()+1e-8)

        elif self.loss_type == 'exponential_penalization':
            c = torch.exp(-torch.abs(y_persistence-x)/s)
            w = torch.exp(REWEIGHT_GAIN*c)
            P = ((torch.abs(x-y)/s)*w).sum()/(w.sum()+1e-8)

        elif self.loss_type == 'mda':
            ##torch.sign has zero gradient everywhere, so the old branch trained
            ##NOTHING (pw=0/0.1/0.5/1.0 gave identical metrics to 4 decimals over 60
            ##epochs). tanh is the differentiable replacement
            dx = torch.tanh(torch.diff(x,dim=1)/(MDA_TAU*s))
            dy = torch.tanh(torch.diff(y,dim=1)/(MDA_TAU*s))
            P = ((1.0-dx*dy)/2.0).mean()

        elif self.loss_type=='triplet':
            ##push x towards y and away from persistence. Mean over the channel dim (not
            ##nn.TripletMarginLoss, whose p=1 distance SUMS over it and so grows with the
            ##number of channels)
            d_pos = (torch.abs(x-y)/s).mean(dim=-1)
            d_neg = (torch.abs(x-y_persistence)/s).mean(dim=-1)
            P = torch.relu(d_pos-d_neg+TRIPLET_MARGIN).mean()

        elif self.loss_type=='high_order':
            ##moment of order i is normalized by s**i --> dimensionless with a bounded
            ##derivative. The i-th root also gives sigma units but its derivative blows
            ##up near 0 and made this branch ~20x too strong
            sc = s.squeeze(1)
            P = 0.0
            for i in range(2,5):
                P = P + torch.abs(central_moment(y,i)/sc.pow(i)-central_moment(x,i)/sc.pow(i)).mean()
            P = P/3.0

        elif self.loss_type == 'additive_iv':
            P = (torch.abs(x.std(dim=1)-y.std(dim=1))/s.squeeze(1)).mean()

        elif self.loss_type == 'multiplicative_iv':
            ##same quantity as additive_iv but keeping the multiplicative flavour
            iv = torch.abs(x.std(dim=1)-y.std(dim=1))/s.squeeze(1)
            P = ((torch.abs(x-y)/s).mean(dim=1)*(1.0+iv)).mean()

        elif self.loss_type=='global_iv':
            P = (torch.abs(x.std(dim=(0,1))-y.std(dim=(0,1)))/s.flatten()).mean()

        elif self.loss_type=='sinkhorn':
            ##p=2 cost --> sqrt to get back to sigma units
            sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
            P = torch.sqrt(sinkhorn.compute(x/s,y/s)+1e-8)

        elif self.loss_type=='dilated':
            ##no multichannel here, one dilate_loss per output channel. O(T^2) per
            ##sample, this branch is slow
            x_n = x/s
            y_n = y/s
            dilate = 0
            for i in range(y_hat.shape[2]):
                dilate = dilate + dilate_loss(y_n[:,:,i:i+1],x_n[:,:,i:i+1], 0.5, 0.01, y_hat.device)
            ##dilate_loss accumulates squared distances along a path of length
            ##future_steps, so it is O(T*d^2): /T then sqrt brings it back to sigma units
            P = torch.sqrt(dilate/(y_hat.shape[2]*self.future_steps)+1e-8)

        elif self.loss_type=='huber':
            P = nn.functional.huber_loss(x/s, y/s, delta=1.0)

        elif self.loss_type=='fredf':
            ##FreDF (ICLR 2025): score the residual in the frequency domain instead of
            ##point by point, which removes the label autocorrelation bias of direct
            ##multi step forecasting (the bias that pushes a model towards persistence).
            ##rfft is linear so we transform the residual directly. norm='ortho' keeps
            ##the transform unitary, so the coefficients stay in the same sigma units as
            ##the time domain residual and P is comparable to base without extra scaling.
            ##The explicit sqrt avoids torch.abs()'s NaN gradient at a zero coefficient
            ##transform over the LAST dim on a contiguous tensor: with dim=1 (the time
            ##axis in place) torch.compile/inductor asserts on the strides of _fft_r2c
            ##and the training step dies. We only take a mean afterwards, so moving the
            ##time axis to the end is free
            d = torch.fft.rfft(((x-y)/s).transpose(1,2).contiguous(), dim=-1, norm='ortho')
            P = torch.sqrt(d.real**2 + d.imag**2 + 1e-8).mean()

        else:
            return initial_loss

        P = P*CALIBRATION.get(self.loss_type, 1.0)
        return (base + pw*P)/(1.0+pw)
