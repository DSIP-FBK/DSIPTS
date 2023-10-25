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
                 out_channels:int,
                 past_steps:int,
                 future_steps: int, 
                 past_channels:int,
                 future_channels:int,
                 embs: list[int],

                 learn_var:bool, 
                 cosine_alpha: bool,
                 diffusion_steps: int,
                 beta: float,
                 sigma: float,
                 #for subnet
                 n_layers_RNN:int,
                 d_head:int,
                 n_head:int,
                 dropout_rate: float,

                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[float]=[],
                 optim:Union[str,None]=None,
                 optim_config:dict=None,
                 scheduler_config:dict=None,
                 **kwargs)->None:
        
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        # >>>>>>>>>>>>> TORCH LIGHTNING
        self.persistence_weight = persistence_weight 
        self.loss_type = loss_type
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config

        # OUTPUT HANDLING LOSSES, here used for inference
        # For training, used losses are specified at the end of the file as functions
        # handling quantiles or not
        assert (len(quantiles) ==0) or (len(quantiles)==3), beauty_string('Only 3 quantiles are availables, otherwise set quantiles=[]','block',True)
        if len(quantiles)==0:
            self.mul = 1
            self.use_quantiles = False
            self.outLinear = nn.Linear(d_model, out_channels)
            if self.loss_type == 'mse':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()
        else:
            self.mul = len(quantiles)
            self.use_quantiles = True
            self.outLinear = nn.Linear(d_model, out_channels*len(quantiles))
            self.loss = QuantileLossMO(quantiles)
        
        # >>>>>>>>>>>>> canonical data parameters
        self.d_model = d_model
        self.dropout = dropout_rate
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.past_channels = past_channels
        self.future_channels = future_channels
        self.output_channels = out_channels

        # >>>>>>>>>>>>> specific model parameters
        self.learn_var = learn_var
        self.T = diffusion_steps
        self.multinomial_step_weights = np.ones(diffusion_steps)
        self.simultaneous_steps = int(diffusion_steps/5) # 1/5 of all sabunets trained every batch of every epoch
        self.sigma = sigma

        # >>>>>>>>>>>>> specific diffusion setup
        if cosine_alpha:
            # COSINE_ALPHA Computation
            # offset variables to control betas and alphas
            # assert self.T < 500 # to avoid problems with extremes
            self.s = 0.001
            aux_perc = 0.05
            avoid_comp_err_norm = self.T*(1+aux_perc)
            # alpha is the 'forgetting' schedule
            f_cos_t = [(np.cos( (t/avoid_comp_err_norm +self.s)/(1+self.s) * np.pi/2 ))**2 for t in range(self.T)]
            self.alphas_cumprod = f_cos_t/f_cos_t[0] # scaled cumulative product of alphas 
            self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) # auxiliar vector to get easily alphaBAT_t-1 
            self.alphas = self.alphas_cumprod * (self.alphas_cumprod_prev)**(-1)
            self.betas = np.append(self.s, 1-self.alphas[1:])
        else:
            # STANDARD ALPHA
            # beta is considered constant in [0,1) for all time steps. Good values near 0.03
            self.betas = np.array([beta]*self.T) 
            self.alphas = 1 - self.betas
            self.alphas_cumprod = np.cumprod(self.alphas)
            self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) # auxiliar vector to get easily alphaBAT_t-1 
        # values for posterior distribution
        self.posterior_mean_coef1 = np.append(1, self.betas[1:] * np.sqrt(self.alphas_cumprod_prev[1:]) / (1.0 - self.alphas_cumprod[1:]))
        self.posterior_mean_coef2 = np.append(0, (1.0 - self.alphas_cumprod_prev[1:]) * np.sqrt(self.alphas[1:]) / (1.0 - self.alphas_cumprod[1:]))
        self.posterior_variance = np.append([self.s,self.s], self.betas[2:] * (1.0 - self.alphas_cumprod_prev[2:]) / (1.0 - self.alphas_cumprod[2:]))
        self.posterior_log_variance_clipped = np.log(self.posterior_variance)

        # >>>>>>>>>>>>> LAYERS    
        # the target variables will be embedded inside the subnet, while other context variables
        # for target variable(s)
        self.target_linear = nn.Linear(out_channels, d_model)
        # for other numerical variables in the past
        self.aux_past_channels = past_channels - out_channels # past umerical variables without target one(s)
        self.linear_aux_past = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_past_channels)])
        # for numerical variables in the future
        self.aux_fut_channels = future_channels
        self.linear_aux_fut = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_fut_channels)])
        
        # embedding categorical for both past and future (ASSUMING BOTH AVAILABLE OR NO ONE)
        self.seq_len = past_steps + future_steps
        self.emb_cat_var = sub_nn.embedding_cat_variables(self.seq_len, future_steps, d_model, embs, self.device)

        # layers for autoregressive eps prediction: LSTM
        self.lin_y_past_d_model = nn.Linear(self.output_channels, d_model)
        self.lstm = sub_nn.LSTM_Model(self.output_channels, d_model, future_steps, n_layers_RNN, dropout_rate)

        # diffusion sub nets, one subnet for each step
        self.sub_nets = nn.ModuleList([
            SubNet(learn_var, out_channels, d_model, d_head, n_head, dropout_rate) for _ in range(diffusion_steps)
        ])


    def forward(self, batch:dict) -> torch.Tensor:
        """forward method used to make subnet learn the noise added the the latent variable.
        

        Consequently, in inference the model will subtract the computed noise for each step.

        Args:
            batch (dict): Keys checked ['x_num_past, 'idx_target', 'x_num_future', 'x_cat_past', 'x_cat_future', 'y']

        Returns:
            torch.Tensor: loss to be subtracted element-wise to the input target tensor 
        """

        # LOADING TARGET VARIABLES
        y_to_be_pred = batch['y'].to(self.device)
        batch_size = y_to_be_pred.shape[0]

        # LOADING AUTOREGRESSIVE CONTEXT OF TARGET VARIABLES
        num_past = batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        y_past = num_past[:,:,idx_target]

        # AUTOREGRESSIVE prediction of the future values of y using lstm(y_past) with embeddings
        emb_y_past = self.lin_y_past_d_model(y_past)
        y_noised_hat = self.lstm(emb_y_past)

        # LOADING EMBEDDING CATEGORICAL VARIABLES
        emb_cat_past, emb_cat_fut = self.cat_categorical_vars(batch)
        emb_cat_past = torch.mean(emb_cat_past, dim = 2)
        emb_cat_fut = torch.mean(emb_cat_fut, dim = 2)

        ### LOADING PAST AND FUTURE NUMERICAL VARIABLES
        # this check is done simultaneously 
        # because in the model we use auxiliar numerical variables 
        # only if we have both them in the past and in the future

        if self.aux_past_channels>0 and self.aux_fut_channels>0: # if we have more numerical variables about past
            # AUX means AUXILIARY variables
            aux_num_past = self.remove_var(num_past, idx_target, 2) # remove the target index on the second dimension
            assert self.aux_past_channels == aux_num_past.size(2),  beauty_string(f"{self.aux_past_channels} LAYERS FOR PAST VARS AND {aux_num_past.shape(2)} VARS",'section',True) # to check if we are using the expected number of variables about past
            
            # past variables
            aux_emb_num_past = torch.Tensor().to(self.device)
            for i, layer in enumerate(self.linear_aux_past):
                aux_emb_past = layer(aux_num_past[:,:,[i]]).unsqueeze(2)
                aux_emb_num_past = torch.cat((aux_emb_num_past, aux_emb_past), dim=2)
            aux_emb_num_past = torch.mean(aux_emb_num_past, dim = 2)
            
            # future_variables
            aux_num_fut = batch['x_num_future'].to(self.device)
            assert self.aux_fut_channels == aux_num_fut.size(2), beauty_string(f"{self.aux_fut_channels} LAYERS FOR PAST VARS AND {aux_num_fut.size(2)} VARS",'section',True)  # to check if we are using the expected number of variables about fut
            aux_emb_num_fut = torch.Tensor().to(self.device)
            for j, layer in enumerate(self.linear_aux_fut):
                aux_emb_fut = layer(aux_num_fut[:,:,[j]]).unsqueeze(2)
                aux_emb_num_fut = torch.cat((aux_emb_num_fut, aux_emb_fut), dim=2)
            aux_emb_num_fut = torch.mean(aux_emb_num_fut, dim = 2)
        else:
            aux_emb_num_past, aux_emb_num_fut = None, None

        ### DIFFUSION

        ##* CHOOSE THE t SUBNET
        # extract a t, indicating which network will be used
        # We have T subnets: [0, 1, ..., T-1].
        values = list(range(self.T))
        # avoid exploding step_weights with usages
        self.improving_weight_during_training()
        # normalizing weights
        t_wei = self.multinomial_step_weights/np.sum(self.multinomial_step_weights)
        # extract times t
        drawn_t = np.random.choice(values, size=self.simultaneous_steps, replace=False, p=t_wei)
        # update weights
        non_draw_val = np.delete(values, drawn_t)
        self.multinomial_step_weights[non_draw_val] += 1

        # init negative loss for the first step
        tot_loss = -1
        for t in drawn_t:
            # LOADING THE SUBNET
            sub_net = self.sub_nets[t]
            eps_hat = y_noised - y_noised_hat

            # Get y and noise it
            y_noised, true_mean, true_log_var_clipped = self.q_sample(x_start = y_to_be_pred, t = t)

            # compute the output from that network using the sample with noises
            # output composed of: noise predicted and vector for variances
            if self.learn_var:
                eps_pred, var_pred = sub_net(eps_hat, y_past, emb_cat_past, emb_cat_fut, aux_emb_num_past, aux_emb_num_fut)
                # variance range
                var_range_A = self._extract_into_tensor(np.log(self.betas) , t, eps_pred.shape)
                var_range_B = true_log_var_clipped
                out_log_var = torch.exp(var_pred*var_range_A + (1-var_pred)*var_range_B)
            else:
                eps_pred = sub_net(eps_hat, y_past, emb_cat_past, emb_cat_fut, aux_emb_num_past, aux_emb_num_fut)
                out_log_var = true_log_var_clipped

            out_mean = self._extract_into_tensor(1/self.alphas, t, eps_pred.shape) * ( y_noised - self._extract_into_tensor(self.betas , t, eps_pred.shape) / self._extract_into_tensor(self.betas , t, eps_pred.shape) * eps_pred )
            
            # # At the first timestep return the decoder NLL,
            # # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            if t==0:
                decoder_nll = -self.discretized_gaussian_log_likelihood(y_to_be_pred, means=out_mean, log_scales=0.5 * out_log_var)
                assert decoder_nll.shape == y_to_be_pred.shape
                decoder_nll = torch.mean(decoder_nll) / np.log(2.0)
                loss_output = decoder_nll
            else:
                # COMPUTE LOSS between TRUE eps and DRAWN eps_pred
                kl = self.normal_kl(true_mean, true_log_var_clipped, out_mean, out_log_var)
                kl = torch.mean(kl) / np.log(2.0)
                loss_output = kl
            # loss_output = torch.where((t == 0), decoder_nll, kl)
            
            # update the total loss
            if tot_loss==-1:
                tot_loss = loss_output
            else:
                tot_loss += loss_output
        return tot_loss

    # re-defined to extract directly the loss of the training step
    # training step has its proper loss computed during forward!
    def training_step(self, batch, batch_idx):
        loss_eps = self(batch)
        return loss_eps
    
    
    def inference(self, batch:dict) -> torch.Tensor:
        """Inference process to generate future y

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

        # AUTOREGRESSIVE prediction of the future values of y using lstm(y_past) with embeddings
        emb_y_past = self.lin_y_past_d_model(y_past)
        y_noised_hat = self.lstm(emb_y_past)
        

        # LOADING EMBEDDING CATEGORICAL VARIABLES
        emb_cat_past, emb_cat_fut = self.cat_categorical_vars(batch)
        emb_cat_past = torch.mean(emb_cat_past, dim = 2)
        emb_cat_fut = torch.mean(emb_cat_fut, dim = 2)

        ### LOADING PAST AND FUTURE NUMERICAL VARIABLES
        # this check is done simultaneously 
        # because in the model we use auxiliar numerical variables 
        # only if we have both them in the past and in the future

        if self.aux_past_channels>0 and self.aux_fut_channels>0: # if we have more numerical variables about past
            # AUX means AUXILIARY variables
            aux_num_past = self.remove_var(num_past, idx_target, 2) # remove the target index on the second dimension
            assert self.aux_past_channels == aux_num_past.size(2),  beauty_string(f"{self.aux_past_channels} LAYERS FOR PAST VARS AND {aux_num_past.shape(2)} VARS",'section',True) # to check if we are using the expected number of variables about past
            
            # past variables
            aux_emb_num_past = torch.Tensor().to(self.device)
            for i, layer in enumerate(self.linear_aux_past):
                aux_emb_past = layer(aux_num_past[:,:,[i]]).unsqueeze(2)
                aux_emb_num_past = torch.cat((aux_emb_num_past, aux_emb_past), dim=2)
            aux_emb_num_past = torch.mean(aux_emb_num_past, dim = 2)
            
            # future_variables
            aux_num_fut = batch['x_num_future'].to(self.device)
            assert self.aux_fut_channels == aux_num_fut.size(2), beauty_string(f"{self.aux_fut_channels} LAYERS FOR PAST VARS AND {aux_num_fut.size(2)} VARS",'section',True)  # to check if we are using the expected number of variables about fut
            aux_emb_num_fut = torch.Tensor().to(self.device)
            for j, layer in enumerate(self.linear_aux_fut):
                aux_emb_fut = layer(aux_num_fut[:,:,[j]]).unsqueeze(2)
                aux_emb_num_fut = torch.cat((aux_emb_num_fut, aux_emb_fut), dim=2)
            aux_emb_num_fut = torch.mean(aux_emb_num_fut, dim = 2)
        else:
            aux_emb_num_past, aux_emb_num_fut = None, None

        # DIFFUSION INFERENCE 
        y_noised = torch.randn((batch_size, self.future_steps, self.output_channels)).to(self.device)
        # pass the white noise in sub nets
        for t in range(self.T-1, -1, -1): # INVERSE cycle over all subnets, but not the last one
            sub_net = self.sub_nets[t] # load the subnet
            eps_hat = y_noised - y_noised_hat

            ## CHECK THE NUMBER OF PARAMS
            #   model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            #   params = sum([np.prod(p.size()) for p in model_parameters]) -> 13K
            true_log_var_clipped = self._extract_into_tensor( self.posterior_log_variance_clipped, t, y_noised.shape )
            nonzero_mask = float((t != 0))  # no adding noise when t == 0

            if self.learn_var:
                eps_pred, var_pred = sub_net(eps_hat, y_past, emb_cat_past, emb_cat_fut, aux_emb_num_past, aux_emb_num_fut)
                # variance range if it is learned (constant values, so out of the for cycle)
                var_range_A = self._extract_into_tensor(np.log(self.betas) , t, eps_pred.shape)
                var_range_B = true_log_var_clipped
                out_log_var = torch.exp(var_pred*var_range_A + (1-var_pred)*var_range_B)
            else:
                eps_pred = sub_net(eps_hat, y_past, emb_cat_past, emb_cat_fut, aux_emb_num_past, aux_emb_num_fut)
                out_log_var = true_log_var_clipped

            # compute 
            out_mean = self._extract_into_tensor(1/self.alphas , t, eps_pred.shape) * ( y_noised - self._extract_into_tensor(self.betas , t, eps_pred.shape) / self._extract_into_tensor(self.betas , t, eps_pred.shape) * eps_pred )
            # Sample x_{t-1} from the model at the given timestep.
            noise = torch.rand_like(out_mean).to(self.device)
            y_noised = out_mean + nonzero_mask * torch.exp(0.5 * out_log_var) * noise
        
        out = y_noised.view(-1, self.future_steps, self.output_channels, 1)
        return out

    # for validation extract the output from the self.inference method
    def validation_step(self, batch, batch_idx):
        out = self.inference(batch)
        loss = self.compute_loss(batch,out)
        return loss

    # function to concat embedded categorical variables
    def cat_categorical_vars(self, batch:dict):
        """Extracting 

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
    def q_sample(self, x_start: torch.Tensor, t: int, noise: torch.Tensor=None)-> List[torch.Tensor]:
        """Diffuse x_start for t diffusion steps.

        In other words, sample from q(x_t | x_0).

        Also, compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        Posterior mean and variance are the ones to be predicted

        Args:
            x_start (torch.Tensor): values to be predicted
            t (int): diffusion step
            noise (torch.Tensor, optional): custom noise, if None normal distributed. Defaults to None.

        Returns:
            List[torch.Tensor, torch.Tensor, torch.Tensor]: q_sample, posterior mean and posterior log variance
        """
        
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        q_sample = self._extract_into_tensor(np.sqrt(self.alphas_cumprod), t, x_start.shape) * x_start + self._extract_into_tensor(np.sqrt(1 - self.alphas_cumprod), t, x_start.shape) * noise
        q_mean = self._extract_into_tensor(self.posterior_mean_coef1, t, q_sample.shape) * x_start + self._extract_into_tensor(self.posterior_mean_coef2, t, q_sample.shape) * q_sample
        q_log_var_clipped = self._extract_into_tensor( self.posterior_log_variance_clipped, t, q_sample.shape )
        # return, the sample, its posterior mean and log_variance
        return q_sample, q_mean, q_log_var_clipped

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
    
    def discretized_gaussian_log_likelihood(self, x, *, means, log_scales):
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.

        :param x: the target images. It is assumed that this was uint8 values,
                rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x.shape
        return log_probs
    
    def approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal.
        """
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

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

### >>>>>>>>>>>>>  SUB NET
class SubNet(nn.Module):
    def __init__(self, learn_var:bool, output_channel:int, d_model:int, d_head:int, n_head:int, dropout_rate:float) -> None:
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

        self.lin_eps_d_model = nn.Linear(output_channel, d_model)

        # layers for categorical: ATT + ResConn
        self.cat_attention = sub_nn.InterpretableMultiHead(d_model, d_head, n_head)
        self.cat_res_conn = sub_nn.ResidualConnection(d_model, dropout_rate)

        # layers for numerical: ATT + ResConn
        self.num_attention = sub_nn.InterpretableMultiHead(d_model, d_head, n_head)
        self.num_res_conn = sub_nn.ResidualConnection(d_model, dropout_rate)

        # layers for eps in dim=output_channel: ResConn
        self.lin_eps_out = nn.Linear(d_model, output_channel)
        self.eps_res_conn = sub_nn.ResidualConnection(output_channel, dropout_rate)

        self.learn_var = learn_var
        if self.learn_var:
            self.var_res_conn = sub_nn.ResidualConnection(output_channel, dropout_rate)



    def forward(self, eps_hat:torch.Tensor, y_past:torch.Tensor,
                cat_past:torch.Tensor = None, cat_future:torch.Tensor = None, 
                num_past:torch.Tensor = None, num_future:torch.Tensor = None)-> torch.Tensor:
        """'y_past' is used with 'y_noised' for a first computation. They are always needed.
        Both 'y_past' and 'y_noised' are not embedded. Each subnet will have its own computation for that.  

        'cat_past' and 'cat_future' to check categorical changes between context steps and ones to be predicted.
        Not always needed. Used in pairs when available.   

        'num_past' and 'num_future' to check numerical changes between context steps and ones to be predicted.
        Not always needed. Used in pairs when available.

        Args:
            y_noised (torch.Tensor): [B, future_step, num_var]
            y_noised_hat (torch.Tensor): [B, past_step, num_var]
            cat_past (torch.Tensor, optional): [B, past_step, d_model]. Defaults to None.
            cat_future (torch.Tensor, optional): [B, future_step, d_model]. Defaults to None.
            num_past (torch.Tensor, optional): [B, past_step, d_model]. Defaults to None.
            num_future (torch.Tensor, optional): [B, future_step, d_model]. Defaults to None.

        Returns:
            torch.Tensor: predicted noise [B, future_step, num_var]. According to 'learn_var' param in initialization, the subnet returns another tensor of same size about the variance 
        """
        import pdb
        pdb.set_trace()
        # emb_eps_hat for further computations
        emb_eps_hat = self.lin_eps_d_model(eps_hat.float()) # -> [B, future_step, d_model]

        # emb_eps_hat updated according to changes of CATEGORICAL information
        # needed info about both past and future
        if (cat_past is not None and cat_future is not None): 
            cat_attention = self.cat_attention(cat_future, cat_past, y_past.float()) # -> [B, future_step, d_model]
            emb_eps_hat = self.cat_res_conn(cat_attention, emb_eps_hat.float())

        # emb_eps_hat updated according to changes of NUMERICAL information
        # needed info about both past and future
        if (num_past is not None and num_future is not None): 
            num_attention = self.num_attention(num_future, num_past, y_past.float()) # -> [B, future_step, d_model]
            emb_eps_hat = self.num_res_conn(num_attention, emb_eps_hat.float())
            
        # last residual connection on dimension = actual number of variables to be predicted
        # emb_eps_hat re-mapped to the starting dimension
        aux_eps_hat = self.lin_eps_out(emb_eps_hat.float())
        # Res Conn (last update of eps)
        eps_hat = self.eps_res_conn(eps_hat.float(), aux_eps_hat.float())

        if self.learn_var:
            var_hat = self.var_res_conn(eps_hat.float(), aux_eps_hat.float())
            return eps_hat, var_hat
        else:
            return eps_hat
        
