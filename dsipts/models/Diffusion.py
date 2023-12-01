import torch
import torch.nn as nn
import numpy as np
from .tft import sub_nn
from .base import  Base
from .utils import  QuantileLossMO
from typing import List, Union
from ..data_structure.utils import beauty_string

class Diffusion(Base):
    def __init__(self, 
                 d_model: int,
                 out_channels: int,
                 past_steps: int,
                 future_steps: int, 
                 past_channels: int,
                 future_channels: int,
                 embs: list[int],

                 learn_var:bool, 
                 cosine_alpha: bool,
                 diffusion_steps: int,
                 beta: float,
                 gamma:float,
                 #for subnet
                 n_layers_RNN: int,
                 d_head: int,
                 n_head: int,
                 dropout_rate: float,
                 activation: str,
                 subnet:int,
                 perc_subnet_learning_for_step:float,

                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[float]=[],
                 optim:Union[str,None]=None,
                 optim_config:Union[dict,None]=None,
                 scheduler_config:Union[dict,None]=None,
                 **kwargs)->None:
        """Denoising Diffusion Probabilistic Model

        Args:
            d_model (int): 
            out_channels (int): number of target variables
            past_steps (int): size of past window
            future_steps (int): size of future window to be predicted
            past_channels (int): number of variables available for the past context
            future_channels (int): number of variables known in the future, available for forecasting
            embs (list[int]): categorical variables dimensions for embeddings
            learn_var (bool): Flag to make the model train the posterior variance (if True) or use the variance of posterior distribution 
            cosine_alpha (bool): Flag for the generation of alphas and betas
            diffusion_steps (int): number of noising steps for the initial sample
            beta (float): starting variable to generate the diffusion perturbations. Ignored if cosine_alpha == True
            gamma (float): trade_off variable to balance loss over noise prediction and NegativeLikelihood/KL_Divergence.
            n_layers_RNN (int): param for subnet
            d_head (int): param for subnet
            n_head (int): param for subnet
            dropout_rate (float): param for subnet
            activation (str): param for subnet
            subnet (int): =1 for attention subnet, =2 for linear subnet. Others can be added(wait for Black Friday for discounts)
            perc_subnet_learning_for_step (float): percentage to choose how many subnet has to be trained for every batch. Decrease this value if the loss blows up.
            persistence_weight (float, optional): Defaults to 0.0.
            loss_type (str, optional): Defaults to 'l1'.
            quantiles (List[float], optional): Only [] accepted. Defaults to [].
            optim (Union[str,None], optional): Defaults to None.
            optim_config (Union[dict,None], optional): Defaults to None.
            scheduler_config (Union[dict,None], optional): Defaults to None.
        """
        
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        self.dropout = dropout_rate
        self.persistence_weight = persistence_weight 
        self.loss_type = loss_type
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config

        #* HANDLING LOSSES 
        # using 'loss_type' and 'quantiles'
        # Here we fix the loss the loss between actual noise and predicted noise
        # Avoid quantile loss!
        assert len(quantiles) ==0
        self.mul = 1
        self.use_quantiles = False
        self.outLinear = nn.Linear(d_model, out_channels)
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()
        
        #* >>>>>>>>>>>>> canonical data parameters
        # dimension of the model, number of variables and sequence length info
        self.d_model = d_model
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.past_channels = past_channels
        self.future_channels = future_channels
        self.output_channels = out_channels

        #* >>>>>>>>>>>>> specific model parameters
        self.learn_var = learn_var # if we want to learn also the variance, instead of using the standard posterior variance of Diffusion NN
        self.T = diffusion_steps # number of noising steps
        self.multinomial_step_weights = np.ones(diffusion_steps) # distribution weigths to avoid less trained subnet 
        self.simultaneous_steps = max(int(diffusion_steps*perc_subnet_learning_for_step), 1) # % of all subnets trained every batch of every epoch
        self.gamma = gamma # trade off for noise loss and distribution loss 

        #* >>>>>>>>>>>>> specific diffusion setup
        self.s = (100*self.T)**(-1)  # offset variable to avoid problems with computations near 0
        # betas and cumulative products of alphas are the main values for the diffusion model
        # according to the flag below we can choose how to generate them
        if cosine_alpha:
            # COSINE ALPHA Computation
            aux_perc = 0.05
            avoid_comp_err_norm = self.T*(1+aux_perc) # enlarging self.T to avoid errors in computations using cos^2
            # the t-th cumulative product of alphas is the 'forgetting' schedule of the inital sample after t diffusion step
            # in this procedure we use the function below to produce all the cumulative products of alphas
            f_cos_t = [(np.cos( (t/avoid_comp_err_norm +self.s*2)/(1+self.s*2) * np.pi/2 ))**2 for t in range(self.T)]
            self.alphas_cumprod = np.append(1-self.s, f_cos_t[1:]/f_cos_t[0]) # scaled cumulative product of alphas f_cos_t[1:]/f_cos_t[0]
            self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) # auxiliar vector to get easily alphaBAT_t-1 
            self.alphas = self.alphas_cumprod * (self.alphas_cumprod_prev)**(-1)
            self.betas = np.append(self.s, 1-self.alphas[1:])
        else:
            # STANDARD ALPHA Computation
            # beta is considered constant in [0,1) for all time steps. Good values near 0.03
            self.betas = np.array([beta]*self.T) 
            self.alphas = 1 - self.betas
            self.alphas_cumprod = np.cumprod(self.alphas)
            self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) # auxiliar vector to get easily alphaBAT_t-1

        # values for posterior distribution, id est the target distribution of each subnet
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = np.append([self.s], self.betas[1:] * (1.0 - self.alphas_cumprod_prev[1:]) / (1.0 - self.alphas_cumprod[1:]))
        self.posterior_log_variance = np.log(self.posterior_variance)

        #* >>>>>>>>>>>>> LAYERS
        # for other numerical variables in the past
        self.aux_past_channels = past_channels - out_channels
        self.linear_aux_past = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_past_channels)])

        # for numerical variables in the future
        self.aux_fut_channels = future_channels
        self.linear_aux_fut = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_fut_channels)])
        
        # embedding categorical for both past and future (ASSUMING BOTH AVAILABLE OR NO ONE)
        self.seq_len = past_steps + future_steps
        self.emb_cat_var = sub_nn.embedding_cat_variables(self.seq_len, future_steps, d_model, embs, self.device)

        # diffusion sub nets, one subnet for each step
        if subnet == 1:
            self.sub_nets = nn.ModuleList([
                SubNet1(self.aux_past_channels, self.aux_fut_channels, learn_var, out_channels, d_model, d_head, n_head, activation, dropout_rate) for _ in range(diffusion_steps)
            ])
        elif subnet == 2:
            self.sub_nets = nn.ModuleList([
                SubNet2(self.aux_past_channels, self.aux_fut_channels, learn_var, past_steps, future_steps, out_channels, d_model, activation, dropout_rate) for _ in range(diffusion_steps)
            ])
        elif subnet ==3 :
            aux_num_available = self.aux_past_channels>0 or self.aux_fut_channels>0  ## check with AM and -->or

            self.sub_nets = nn.ModuleList([
                SubNet3(learn_var, aux_num_available, out_channels, d_model, future_steps, n_layers_RNN, d_head, n_head, dropout_rate) for _ in range(diffusion_steps)
            ])
        else:
            raise ValueError("Wrong number for Subnet. Not yet implemented!.")


    def forward(self, batch:dict)-> float:
        """training process of the diffusion network

        Args:
            batch (dict): variables loaded

        Returns:
            float: total loss about the prediction of the noises over all subnets extracted
        """

        # LOADING TARGET VARIABLES
        y_to_be_pred = batch['y'].to(self.device)

        # LOADING AUTOREGRESSIVE CONTEXT OF TARGET VARIABLES
        num_past = batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        y_past = num_past[:,:,idx_target]

        # LOADING EMBEDDING CATEGORICAL VARIABLES
        emb_cat_past, emb_cat_fut = self.cat_categorical_vars(batch)
        emb_cat_past = torch.mean(emb_cat_past, dim = 2)
        emb_cat_fut = torch.mean(emb_cat_fut, dim = 2)

        ### LOADING PAST AND FUTURE NUMERICAL VARIABLES
        # load in the model auxiliar numerical variables 
        if self.aux_past_channels>0: 
            aux_num_past = self.remove_var(num_past, idx_target, 2) # remove the target index on the second dimension
            assert self.aux_past_channels == aux_num_past.size(2),  beauty_string(f"{self.aux_past_channels} LAYERS FOR PAST VARS AND {aux_num_past.size(2)} VARS",'section',True) # to check if we are using the expected number of variables about past
            
            # past variables
            aux_emb_num_past = torch.Tensor().to(self.device)
            for i, layer in enumerate(self.linear_aux_past):
                aux_emb_past = layer(aux_num_past[:,:,[i]]).unsqueeze(2)
                aux_emb_num_past = torch.cat((aux_emb_num_past, aux_emb_past), dim=2)
            aux_emb_num_past = torch.mean(aux_emb_num_past, dim = 2)
        else: 
            aux_emb_num_past = None
            
        if self.aux_fut_channels>0: # if we have more numerical variables about past
            # AUX means AUXILIARY variables
            # future_variables
            aux_num_fut = batch['x_num_future'].to(self.device)
            assert self.aux_fut_channels == aux_num_fut.size(2), beauty_string(f"{self.aux_fut_channels} LAYERS FOR PAST VARS AND {aux_num_fut.size(2)} VARS",'section',True)  # to check if we are using the expected number of variables about fut
            aux_emb_num_fut = torch.Tensor().to(self.device)
            for j, layer in enumerate(self.linear_aux_fut):
                aux_emb_fut = layer(aux_num_fut[:,:,[j]]).unsqueeze(2)
                aux_emb_num_fut = torch.cat((aux_emb_num_fut, aux_emb_fut), dim=2)
            aux_emb_num_fut = torch.mean(aux_emb_num_fut, dim = 2)
        else:
            aux_emb_num_fut = None

        ### actual DIFFUSION process ----------------------------------------------

        ##* CHOOSE THE t SUBNET
        # We have T subnets: [0, 1, ..., T-1].
        values = list(range(self.T))
        
        ## Probabilistic way to choose the subnet properly
        # avoid exploding step_weights with usages
        self.improving_weight_during_training() 
        # normalizing weights for a more stable subnet training
        t_wei = self.multinomial_step_weights/np.sum(self.multinomial_step_weights)
        # extract times t
        drawn_t = np.random.choice(values, size=self.simultaneous_steps, replace=False, p=t_wei) # type: ignore
        # update weights
        non_draw_val = np.delete(values, drawn_t) # type: ignore
        self.multinomial_step_weights[non_draw_val] += 1

        # init negative loss for the first step
        tot_loss = -1
        for t in drawn_t:
            # LOADING THE SUBNET
            sub_net = self.sub_nets[t]
            # Get y and noise it
            y_noised, true_mean, true_log_var, actual_noise = self.q_sample(y_to_be_pred, t)

            # compute the output from that network using the sample with noises
            # output composed of: noise predicted and vector for variances
            if self.learn_var:
                eps_pred, var_aux_out = sub_net(y_noised, y_past, emb_cat_past, emb_cat_fut, aux_emb_num_past, aux_emb_num_fut)
                pre_var_t = self._extract_into_tensor(np.sqrt(self.betas), t, eps_pred.shape)
                post_var_t = self._extract_into_tensor(np.sqrt(self.posterior_variance), t, eps_pred.shape)
                post_sigma = torch.exp(var_aux_out*torch.log(pre_var_t) + (1-var_aux_out)*torch.log(post_var_t)) # variance, not log_var
            else:
                eps_pred = sub_net(y_noised, y_past, emb_cat_past, emb_cat_fut, aux_emb_num_past, aux_emb_num_fut)
                post_sigma = self._extract_into_tensor(self.posterior_variance, t, eps_pred.shape)

            # posterior mean assuming the predicted noise is the actual one
            out_mean = self._extract_into_tensor(np.sqrt(1/self.alphas), t, eps_pred.shape) * ( y_noised - self._extract_into_tensor(self.betas/np.sqrt(1-self.alphas_cumprod) , t, eps_pred.shape) * eps_pred )
            
            # # At the first timestep return the negative likelihood,
            # # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            if t==0:
                # post_var =  self._extract_into_tensor(self.posterior_variance, t, y_to_be_pred.shape)
                neg_likelihoods = -self.gaussian_likelihood(y_to_be_pred, out_mean, post_sigma) # (values predicted, mean of values predicted, variance)
                distribution_loss = torch.mean(neg_likelihoods)
            else:
                # COMPUTE LOSS between TRUE eps and DRAWN eps_pred
                kl_divergence = self.normal_kl(true_mean, true_log_var, out_mean, torch.log(post_sigma)) # (true mean, true log var, mean of values predicted, log var predicted)
                distribution_loss = torch.mean(kl_divergence)

            # always compute the loss about the straight prediction of the noise
            noise_loss = self.loss(eps_pred, actual_noise)

            noise_loss += self.gamma*distribution_loss # add, scaled according to gamma, the distribution_loss
            
            # update the total loss
            if tot_loss==-1:
                tot_loss = noise_loss
            else:
                tot_loss += noise_loss
        return tot_loss

    def training_step(self, batch, batch_idx):
        # the training loss is already computed in the forward method
        loss_eps = self(batch)
        return loss_eps
    
    
    def inference(self, batch:dict) -> torch.Tensor:
        """Inference process to forecast future y

        Args:
            batch (dict): Keys checked ['x_num_past, 'idx_target', 'x_num_future', 'x_cat_past', 'x_cat_future']

        Returns:
            torch.Tensor: generated sequence [batch_size, future_steps, num_var]
        """
        # LOADING AUTOREGRESSIVE CONTEXT OF TARGET VARIABLES
        num_past = batch['x_num_past'].to(self.device)
        batch_size = num_past.shape[0]
        idx_target = batch['idx_target'][0]
        y_past = num_past[:,:,idx_target]        

        # LOADING EMBEDDING CATEGORICAL VARIABLES
        emb_cat_past, emb_cat_fut = self.cat_categorical_vars(batch)
        emb_cat_past = torch.mean(emb_cat_past, dim = 2)
        emb_cat_fut = torch.mean(emb_cat_fut, dim = 2)

        ### LOADING PAST AND FUTURE NUMERICAL VARIABLES
        # this check is done simultaneously 
        # because in the model we use auxiliar numerical variables 
        # only if we have both them in the past and in the future

        if self.aux_past_channels>0: # if we have more numerical variables about past
            # AUX means AUXILIARY variables
            aux_num_past = self.remove_var(num_past, idx_target, 2) # remove the target index on the second dimension
            assert self.aux_past_channels == aux_num_past.size(2),  beauty_string(f"{self.aux_past_channels} LAYERS FOR PAST VARS AND {aux_num_past.size(2)} VARS",'section',True) # to check if we are using the expected number of variables about past
            
            # past variables
            aux_emb_num_past = torch.Tensor().to(self.device)
            for i, layer in enumerate(self.linear_aux_past):
                aux_emb_past = layer(aux_num_past[:,:,[i]]).unsqueeze(2)
                aux_emb_num_past = torch.cat((aux_emb_num_past, aux_emb_past), dim=2)
            aux_emb_num_past = torch.mean(aux_emb_num_past, dim = 2)
        else:
            aux_emb_num_past = None
        if self.aux_fut_channels>0: 
            # future_variables
            aux_num_fut = batch['x_num_future'].to(self.device)
            assert self.aux_fut_channels == aux_num_fut.size(2), beauty_string(f"{self.aux_fut_channels} LAYERS FOR PAST VARS AND {aux_num_fut.size(2)} VARS",'section',True)  # to check if we are using the expected number of variables about fut
            aux_emb_num_fut = torch.Tensor().to(self.device)
            for j, layer in enumerate(self.linear_aux_fut):
                aux_emb_fut = layer(aux_num_fut[:,:,[j]]).unsqueeze(2)
                aux_emb_num_fut = torch.cat((aux_emb_num_fut, aux_emb_fut), dim=2)
            aux_emb_num_fut = torch.mean(aux_emb_num_fut, dim = 2)
        else:
             aux_emb_num_fut = None

        
        # DIFFUSION INFERENCE 
        y_noised = torch.randn((batch_size, self.future_steps, self.output_channels)).to(self.device)
        # pass the white noise in sub nets
        for t in range(self.T-1, -1, -1): # INVERSE cycle over all subnets, but not the last one
            sub_net = self.sub_nets[t] # load the subnet

            ## CHECK THE NUMBER OF PARAMS
            #   model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            #   params = sum([np.prod(p.size()) for p in model_parameters]) -> 13K
            if self.learn_var:
                eps_pred, var_aux_out = sub_net(y_noised, y_past, emb_cat_past, emb_cat_fut, aux_emb_num_past, aux_emb_num_fut)
                pre_var_t = self._extract_into_tensor(np.sqrt(self.betas), t, eps_pred.shape)
                post_var_t = self._extract_into_tensor(np.sqrt(self.posterior_variance), t, eps_pred.shape)
                post_sigma = torch.exp(var_aux_out*torch.log(pre_var_t) + (1-var_aux_out)*torch.log(post_var_t))
            else:
                eps_pred = sub_net(y_noised, y_past, emb_cat_past, emb_cat_fut, aux_emb_num_past, aux_emb_num_fut)
                post_sigma = self._extract_into_tensor(self.posterior_variance, t, eps_pred.shape)
                
            # Sample x_{t-1} from the model at the given timestep.
            y_noised = self._extract_into_tensor(1/np.sqrt(self.alphas), t, y_noised.shape)*y_noised - self._extract_into_tensor(self.betas/(np.sqrt(self.alphas*self.betas)), t, eps_pred.shape)*eps_pred
            if t>0 :
                noise = torch.rand_like(y_noised).to(self.device)
                y_noised = y_noised + torch.sqrt(post_sigma)*noise
        
        out = y_noised.view(-1, self.future_steps, self.output_channels, 1)
        return out

    # for validation extract the output from the self.inference method
    def validation_step(self, batch, batch_idx):
        out = self.inference(batch)
        loss = self.compute_loss(batch,out)
        return loss

    # function to concat embedded categorical variables
    def cat_categorical_vars(self, batch:dict):
        """Extracting categorical context about past and future

        Args:
            batch (dict): Keys checked -> ['x_cat_past', 'x_cat_future']

        Returns:
            List[torch.Tensor, torch.Tensor]: cat_emb_past, cat_emb_fut
        """
        # GET AVAILABLE CATEGORICAL CONTEXT
        cat_past = batch['x_cat_past'].to(self.device)
        cat_fut = batch['x_cat_future'].to(self.device)
        # CONCAT THEM, according to self.emb_cat_var usage  
        cat_full = torch.cat((cat_past, cat_fut), dim = 1)
        # actual embedding
        emb_cat_full = self.emb_cat_var(cat_full)
        # split past and future categorical embedded variables
        cat_emb_past = emb_cat_full[:,:self.past_steps,:,:]
        cat_emb_fut = emb_cat_full[:,-self.future_steps:,:,:]

        return cat_emb_past, cat_emb_fut

    #function to extract from batch['x_num_past'] all variables except the one autoregressive
    def remove_var(self, tensor: torch.Tensor, indexes_to_exclude: list, dimension: int)-> torch.Tensor:
        """Function to remove variables from tensors in chosen dimension and position 

        Args:
            tensor (torch.Tensor): starting tensor
            indexes_to_exclude (list): index of the chosen dimension we want t oexclude
            dimension (int): dimension of the tensor on which we want to work (not list od dims!!)

        Returns:
            torch.Tensor: new tensor without the chosen variables
        """

        remaining_idx = torch.tensor([i for i in range(tensor.size(dimension)) if i not in indexes_to_exclude]).to(tensor.device)
        # Select the desired sub-tensor
        extracted_subtensors = torch.index_select(tensor, dim=dimension, index=remaining_idx)
        
        return extracted_subtensors
    
    def improving_weight_during_training(self):
        """
        Each time we sample from multinomial we subtract the minimum for more precise sampling, 
        avoiding great learning differences among subnets

        This lead to more stable inference also in early training, mainly for common context embedding.

        For probabilistic reason, weights has to be >0, so we subtract min-1
        """
        self.multinomial_step_weights -= (self.multinomial_step_weights.min()-1)
        return
    
    ### >>>>>>>>>>>>> AUXILIARY MODEL FUNCS
    def q_sample(self, x_start: torch.Tensor, t: int)-> List[torch.Tensor]:
        """Diffuse x_start for t diffusion steps.

        In other words, sample from q(x_t | x_0).

        Also, compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        Posterior mean and variance are the ones to be predicted

        Args:
            x_start (torch.Tensor): values to be predicted
            t (int): diffusion step

        Returns:
            List[torch.Tensor, torch.Tensor, torch.Tensor]: q_sample, posterior mean, posterior log variance and the actual noise
        """
        # noise from normal distribution
        noise = torch.randn_like(x_start)

        # direct diffusion at t-th step
        q_sample = self._extract_into_tensor(np.sqrt(self.alphas_cumprod), t, x_start.shape) * x_start + self._extract_into_tensor(np.sqrt(1 - self.alphas_cumprod), t, x_start.shape) * noise

        # compute meean and variance
        q_mean = self._extract_into_tensor(self.posterior_mean_coef1, t, q_sample.shape) * x_start + self._extract_into_tensor(self.posterior_mean_coef2, t, q_sample.shape) * q_sample
        q_log_var = self._extract_into_tensor( self.posterior_log_variance, t, q_sample.shape )

        # return, the sample, its posterior mean and log_variance
        return [q_sample, q_mean, q_log_var, noise]

    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.

        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"

        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for th.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]

        return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )
    
    def gaussian_likelihood(self, x, mean, var):
        term1 = 1.0 / torch.sqrt(2 * np.pi * var)
        term2 = torch.exp(-0.5 * ((x - mean)**2 / var))
        likelihood = term1 * term2
        return likelihood

    def gaussian_log_likelihood(self, x, mean, var):
        term1 = -0.5 * ((x - mean) / torch.sqrt(var))**2
        term2 = -0.5 * torch.log(2 * torch.tensor(np.pi) * var)
        log_likelihood = term1 + term2
        return log_likelihood

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape 'broadcast_shape' where the shape has K dims.
        """
        ten = torch.tensor(arr[timesteps])
        return ten.expand(broadcast_shape).to(self.device)

### >>>>>>>>>>>>>  SUB NET 1
class SubNet1(nn.Module):
    def __init__(self, aux_past_ch, aux_fut_ch, learn_var:bool, output_channel:int, d_model:int, d_head:int, n_head:int, activation:str, dropout_rate:float) -> None:
        """ -> SUBNET of the DIFFUSION MODEL (DDPM)

        It starts with an autoregressive LSTM Network computation of epsilon, then subtracted to 'y_noised' tensor. This is always possible!
        Now we have an approximation of our 'eps_hat', that at the end will pass in a residual connection with its embedded version 'emb_eps_hat'.

        'emb_eps_hat' will be update with respect to available info about categorical values of our serie:
        Through an ATTENTION Network we compare past categorical with future categorical to update the embedded noise predicted.

        Also, if we have values about auxiliary numerical variables both in past and future, the changes of these variables will be fetched 
        by another ATTENTION Network.

        The goal is ensure valuable computations for 'eps' always, and then updating things if we have enough data.
        Both attentions uses { Q = *_future, K = *_past, V = y_past } using as much as possible context variables for better updates.

        Args:
            learn_var (bool): set if the network has to learn the optim variance of each step
            output_channel (int): number of variables to be predicted 
            future_steps (int): number of step in the future, so the number of timesstep to be predicted
            d_model (int): hidden dimension of the model
            num_layers_RNN (int): number of layers for autoregressive prediction
            d_head (int): number of heads for Attention Networks
            n_head (int): hidden dimension of heads for Attention Networks
            dropout_rate (float): 
        """
        super().__init__()
        self.aux_past_channels = aux_past_ch
        self.aux_fut_channels = aux_fut_ch

        self.learn_var = learn_var
        activation_fun = eval(activation)

        self.y_noised_linear = nn.Linear(output_channel, d_model)
        self.y_past_linear = nn.Linear(output_channel, d_model)

        self.past_sequential = nn.Sequential(
            nn.Linear(d_model*3, d_model*2) if self.aux_past_channels>0 else nn.Linear(d_model*2, d_model*2),
            activation_fun(),
            nn.Linear(d_model*2, d_model)
        )
        
        self.fut_sequential = nn.Sequential(
            nn.Linear(d_model*3, d_model*2) if self.aux_fut_channels>0 else nn.Linear(d_model*2, d_model*2),
            activation_fun(),
            nn.Linear(d_model*2, d_model)
        )

        self.y_sequential = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            activation_fun(),
            nn.Linear(d_model, d_model)
        )

        self.attention = sub_nn.InterpretableMultiHead(d_model, d_head, n_head)

        # if learn_var == True, we want to predict an additional variable for he variance
        # just an intermediate dimension for linears
        hidden_size = int(d_model/3)
        self.eps_out_sequential = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            activation_fun(),
            nn.Linear(hidden_size, output_channel)
        )

        self.var_out_sequential = nn.Sequential(
            nn.Linear(output_channel, hidden_size),
            nn.Linear(hidden_size, d_model),
            activation_fun(),
            nn.Linear(d_model, d_model),
            activation_fun(),
            nn.Linear(d_model, hidden_size),
            nn.Linear(hidden_size, output_channel)
        )


    def forward(self, y_noised:torch.Tensor, y_past:torch.Tensor,
                cat_past:Union[torch.Tensor,None] = None, cat_fut:Union[torch.Tensor,None] = None, 
                num_past:Union[torch.Tensor,None] = None, num_fut:Union[torch.Tensor,None] = None):
        """'DIFFUSION SUBNET
        Args:
            y_noised (torch.Tensor): [B, future_step, num_var]
            y_past (torch.Tensor): [B, past_step, num_var]
            cat_past (torch.Tensor, optional): [B, past_step, d_model]. Defaults to None.
            cat_fut (torch.Tensor, optional): [B, future_step, d_model]. Defaults to None.
            num_past (torch.Tensor, optional): [B, past_step, d_model]. Defaults to None.
            num_fut (torch.Tensor, optional): [B, future_step, d_model]. Defaults to None.

        Returns:
            torch.Tensor: predicted noise [B, future_step, num_var]. According to 'learn_var' param in initialization, the subnet returns another tensor of same size about the variance 
        """
        emb_y_noised = self.y_noised_linear(y_noised.float())
        emb_y_past = self.y_past_linear(y_past)
        
        # LIN FOR PAST
        past = [emb_y_past, cat_past]
        if self.aux_past_channels>0:
            past.append(num_past)
        past_seq_input = torch.cat(past, dim=2) # type: ignore
        past_seq = self.past_sequential(past_seq_input) # -> [B, future_step, d_model]

        # LIN FOR FUT
        fut = [emb_y_noised, cat_fut]
        if self.aux_fut_channels>0:
            fut.append(num_fut)
        fut_seq_input = torch.cat(fut, dim=2) # type: ignore
        fut_seq = self.fut_sequential(fut_seq_input) # -> [B, future_step, d_model]
        # ATTENTION
        attention = self.attention(fut_seq, past_seq, emb_y_past)
        # OUTPUT
        eps_out = self.eps_out_sequential(attention)
        # if LEARN_VAR
        if self.learn_var:
            var_out = eps_out.detach()
            var_out = self.var_out_sequential(var_out)
            return eps_out, var_out

        return eps_out
    
class SubNet2(nn.Module):
    def __init__(self, aux_past_ch, aux_fut_ch, learn_var:bool, past_steps, future_steps, output_channel:int, d_model:int, activation:str, dropout_rate:float):
        super().__init__()
        self.aux_past_channels = aux_past_ch
        self.aux_fut_channels = aux_fut_ch
        self.learn_var = learn_var
        in_size = ( past_steps*(2+bool(aux_past_ch)) + future_steps*(2 + bool(aux_fut_ch)))  * d_model ##Ask AM if correct removing
        out_size = output_channel * future_steps

        activation_fun = eval(activation)

        self.y_noised_linear = nn.Linear(output_channel, d_model)
        self.y_past_linear = nn.Linear(output_channel, d_model)

        hidden_size = int( (output_channel + d_model)/2 )
        self.eps_out_sequential = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, d_model),
            activation_fun(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, hidden_size),
            activation_fun(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_size)
        )
        
        self.var_out_sequential = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            activation_fun(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, y_noised:torch.Tensor, y_past:torch.Tensor,
                cat_past:Union[torch.Tensor,None] = None, cat_fut:Union[torch.Tensor,None] = None, 
                num_past:Union[torch.Tensor,None] = None, num_fut:Union[torch.Tensor,None] = None):
        
        B, fut_step, n_var = y_noised.shape
        emb_y_noised = self.y_noised_linear(y_noised.float()).view(B, -1)
        emb_y_past = self.y_past_linear(y_past).view(B, -1)

        full_concat = torch.cat((emb_y_noised, emb_y_past,cat_past.view(B, -1),cat_fut.view(B, -1)), dim=1)
        for ten in [num_past,num_fut]:
            if ten is not None:
                full_concat = torch.cat((full_concat, ten.view(B, -1)), dim = 1)

        eps_out = self.eps_out_sequential(full_concat).view(B, fut_step, n_var)
        if self.learn_var:
            var_out = self.var_out_sequential(full_concat.detach()).view(B, fut_step, n_var)
            return eps_out, var_out
        return eps_out

class SubNet3(nn.Module):
    def __init__(self, learn_var, flag_aux_num, num_var, d_model, pred_step, num_layers, d_head, n_head, dropout):
        super().__init__()
        self.learn_var = learn_var
        self.flag_aux_num = flag_aux_num
        
        # Autoregressive with RNN (y NOT embedded as inpute)
        self.y_d_model = nn.Linear(num_var, d_model)
        self.rnn = sub_nn.LSTM_Model(num_var, d_model, pred_step, num_layers, dropout)
        self.eps_pred_grn = sub_nn.GRN(d_model, dropout)

        #categorical
        self.cat_MHA = sub_nn.InterpretableMultiHead(d_model, d_head, n_head)
        self.cat_grn = sub_nn.GRN(d_model, dropout)
        self.cat_res_conn = sub_nn.ResidualConnection(d_model, dropout)

        #numerical
        if flag_aux_num:
            self.num_MHA = sub_nn.InterpretableMultiHead(d_model, d_head, n_head)
            self.num_grn = sub_nn.GRN(d_model, dropout)
            self.num_res_conn = sub_nn.ResidualConnection(d_model, dropout)

        # EPS PREDICTION
        self.eps_final_grn = sub_nn.GRN(d_model, dropout)
        self.eps_out_linear = nn.Linear(d_model, num_var)

        if learn_var:
            self.emb_eps_pred = nn.Linear(num_var, d_model)
            self.var_att = sub_nn.InterpretableMultiHead(d_model, d_head, n_head)
            self.var_grn = sub_nn.GRN(d_model, dropout)
            self.var_out = nn.Linear(d_model, num_var)



    def forward(self, y_noised:torch.Tensor, y_past:torch.Tensor, 
                cat_past:torch.Tensor, cat_fut:torch.Tensor, 
                num_past:Union[torch.Tensor,None] = None, num_fut:Union[torch.Tensor,None] = None):

        # Autoregressive
        emb_y_past = self.y_d_model(y_past)
        pred_y_fut = self.rnn(emb_y_past)
        #re-embedding future
        emb_pred_y_fut = self.y_d_model(pred_y_fut)
        emb_y_noised = self.y_d_model(y_noised.float())

        eps_pred = self.eps_pred_grn(emb_pred_y_fut - emb_y_noised)

        # Categorical contribute
        cat_att = self.cat_MHA(cat_fut, cat_past, emb_y_past)
        cat_att = self.cat_grn(cat_att)
        eps_pred = self.cat_res_conn(cat_att, eps_pred, using_norm=False)  ##check with AM

        # Numerical contribute
        if self.flag_aux_num:
            if num_past is None:
                num_past = torch.ones_like(cat_past)
            if num_fut is None:
                num_fut = torch.ones_like(cat_fut)
            num_att = self.num_MHA(num_fut, cat_past, emb_y_past)
            num_att = self.num_grn(num_att)
            eps_pred = self.cat_res_conn(num_att, eps_pred, using_norm=False)

        eps_pred = self.eps_final_grn(eps_pred)
        eps_pred = self.eps_out_linear(eps_pred)

        if self.learn_var:
            emb_eps_pred = self.emb_eps_pred(eps_pred.detach())
            emb_eps_pred = self.var_att(emb_y_noised.detach(), emb_pred_y_fut.detach(), emb_eps_pred)
            emb_var_pred = self.var_grn(emb_eps_pred)
            var_pred = self.var_out(emb_var_pred)
            return eps_pred, var_pred
        return eps_pred
            