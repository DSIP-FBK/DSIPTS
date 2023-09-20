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
                 diffusion_steps: int,
                 beta: float,
                 sigma: float,
                 inference_out_from_sub_net:bool,
                 #for subnet
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
    
        self.persistence_weight = persistence_weight 
        self.loss_type = loss_type
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.noise_loss = nn.MSELoss()

        # output, handling quantiles or not
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
        
        # params data
        self.d_model = d_model
        self.dropout = dropout_rate
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.past_channels = past_channels
        self.future_channels = future_channels
        self.output_channels = out_channels

        # params model
        self.T = diffusion_steps
        self.beta = beta
        self.alpha = 1-beta
        self.sigma = sigma

        # layers
        # for target variable(s)
        self.target_linear = nn.Linear(out_channels, d_model)
        # for other numerical variables in the past
        self.aux_past_channels = past_channels - out_channels # past umerical variables without target one(s)
        self.linear_aux_past = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_past_channels)])
        # for numerical variables in the future
        self.aux_fut_channels = future_channels
        self.linear_aux_fut = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.aux_fut_channels)])
        
        # embedding categorical for both past and future
        # ASSUMING BOTH AVAILABLE OR NO ONE
        self.seq_len = past_steps + future_steps
        self.emb_cat_var = sub_nn.embedding_cat_variables(self.seq_len, future_steps, d_model, embs, self.device)

        # diffusion sub nets
        # one subnet for each step
        self.sub_nets = nn.ModuleList([
            SubNet(out_channels, self.mul, d_model, d_head, n_head, dropout_rate) for _ in range(diffusion_steps)
        ])
        # deciding which way to get the output is better (paper not clear) 
        self.inference_out_from_sub_net = inference_out_from_sub_net


    def forward(self, batch:dict) -> torch.Tensor:
        """forward method used to make subnet learn the noise added the the latent variable.

        Consequently, in inference the model will subtract the computed noise for each step.

        Args:
            batch (dict): Keys checked ['x_num_past, 'idx_target', 'x_num_future', 'x_cat_past', 'x_cat_future']

        Returns:
            torch.Tensor: loss to be subtracted element-wise to the input target tensor 
        """

        # past_numerical
        num_past = batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        target_num_past = num_past[:,:,idx_target]
        target_emb_num_past = self.target_linear(target_num_past) # target_variables comunicating with each others

        ### create variable summary_past and summary_fut
        # at the beggining it is composed only by past and future target variable
        summary_past = target_emb_num_past.unsqueeze(2)
        summary_fut = None
        # now we search for others categorical and numerical variables!

        ### PAST NUMERICAL VARIABLES
        # if we have past numerical variables, concatenate them to the summary past 
        if self.aux_past_channels>0: # so we have more numerical variables about past
            # AUX = AUXILIARY variables
            aux_num_past = self.remove_var(num_past, idx_target, 2) # remove the target index on the second dimension
            assert self.aux_past_channels == aux_num_past.size(2),  beauty_string(f"{self.aux_past_channels} LAYERS FOR PAST VARS AND {aux_num_past.shape(2)} VARS",'section',True) # to check if we are using the expected number of variables about past
            aux_emb_num_past = torch.Tensor().to(aux_num_past.device)
            for i, layer in enumerate(self.linear_aux_past):
                aux_emb_past = layer(aux_num_past[:,:,[i]]).unsqueeze(2)
                aux_emb_num_past = torch.cat((aux_emb_num_past, aux_emb_past), dim=2)
            ## update summary about past
            summary_past = torch.cat((summary_past, aux_emb_num_past), dim=2)
        
        ### FUTURE NUMERICAL VARIABLES
        # if we have future numerical variables, concatenate them to the summary fut
        if self.aux_fut_channels>0: # so we have more numerical variables about future
            aux_num_fut = batch['x_num_future'].to(self.device)
            assert self.aux_fut_channels == aux_num_fut.size(2), beauty_string(f"{self.aux_fut_channels} LAYERS FOR PAST VARS AND {aux_num_fut.size(2)} VARS",'section',True)  # to check if we are using the expected number of variables about fut
            aux_emb_num_fut = torch.Tensor().to(aux_num_fut.device)
            for j, layer in enumerate(self.linear_aux_fut):
                aux_emb_fut = layer(aux_num_fut[:,:,[j]]).unsqueeze(2)
                aux_emb_num_fut = torch.cat((aux_emb_num_fut, aux_emb_fut), dim=2)
            ## update summary about future
            if summary_fut is None:
                summary_fut = aux_emb_num_fut
            else:
                summary_fut = torch.cat((summary_fut, aux_emb_num_fut), dim=2)


        ### CATEGORICAL VARIABLES 
        summary_past, summary_fut = self.cat_categorical_vars(batch, summary_past, summary_fut)

        # >>> PAST:
        summary_past = torch.mean(summary_past, dim=2)
        # >>> FUTURE:
        summary_fut = torch.mean(summary_fut, dim=2)

        ### DIFFUSION
        # beta is considered constant in [0,1) for all time steps. Good values near 0.03
        # Also sigma constant.

        ##* CHOOSE THE t SUBNET
        # extract a t, indicating which network will be used
        # We have T subnets: [0, 1, ..., T-1].
        t = np.random.randint(0, self.T) # 0 inclusive, self.T exclusive
        sub_net = self.sub_nets[t]
        # compute alpha_t
        alpha_t = self.alpha**t

        # Get y and noise it
        y_to_be_pred = batch['y'].to(self.device)
        batch_size = y_to_be_pred.shape[0]
        noise = self.generate_noise(batch_size).to(self.device)
        ##* INPUT FOR SUBNET
        y_t_noised = np.sqrt(alpha_t) * y_to_be_pred + np.sqrt(1 - alpha_t) * noise

        # compute the output from that network using the sample with noises
        noise_pred = sub_net(y_t_noised, target_emb_num_past, summary_past, summary_fut)

        # compute the loss with the drawn eps
        loss_noise = torch.sqrt(self.noise_loss(noise_pred, noise))
        return loss_noise

    # re-defined to extract directly the loss of the training step
    def training_step(self, batch, batch_idx):
        loss_eps = self(batch)
        return loss_eps
    
    # for validation extract the output from the self.inference method
    def validation_step(self, batch, batch_idx):
        out = self.inference(batch)
        loss = self.compute_loss(batch,out)
        return loss

    
    def inference(self, batch:dict) -> torch.Tensor:
        """Inference process to generate future y

        Args:
            batch (dict): Keys checked ['x_num_past, 'idx_target', 'x_num_future', 'x_cat_past', 'x_cat_future']

        Returns:
            torch.Tensor: generated sequence [batch_size, future_steps, num_var]
        """

        # PAST TARGET
        num_past = batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        target_num_past = num_past[:,:,idx_target]
        target_emb_num_past = self.target_linear(target_num_past) # target_variables comunicating with each others

        ### create variable summary_past and summary_fut
        # at the beggining it is composed only by past and future target variable
        summary_past = target_emb_num_past.unsqueeze(2)
        summary_fut = None
        # now we search for others categorical and numerical variables!

        ### PAST NUMERICAL VARIABLES
        # if we have past numerical variables, concatenate them to the summary past 
        if self.aux_past_channels>0: # so we have more numerical variables about past
            # AUX = AUXILIARY variables
            aux_num_past = self.remove_var(num_past, idx_target, 2) # remove the target index on the second dimension
            assert self.aux_past_channels == aux_num_past.size(2),  beauty_string(f"{self.aux_past_channels} LAYERS FOR PAST VARS AND {aux_num_past.shape(2)} VARS",'section',True) # to check if we are using the expected number of variables about past
            aux_emb_num_past = torch.Tensor().to(aux_num_past.device)
            for i, layer in enumerate(self.linear_aux_past):
                aux_emb_past = layer(aux_num_past[:,:,[i]]).unsqueeze(2)
                aux_emb_num_past = torch.cat((aux_emb_num_past, aux_emb_past), dim=2)
            ## update summary about past
            summary_past = torch.cat((summary_past, aux_emb_num_past), dim=2)
        
        ### FUTURE NUMERICAL VARIABLES
        # if we have future numerical variables, concatenate them to the summary fut
        if self.aux_fut_channels>0: # so we have more numerical variables about future
            aux_num_fut = batch['x_num_future'].to(self.device)
            assert self.aux_fut_channels == aux_num_fut.size(2), beauty_string(f"{self.aux_fut_channels} LAYERS FOR PAST VARS AND {aux_num_fut.size(2)} VARS",'section',True)  # to check if we are using the expected number of variables about fut
            aux_emb_num_fut = torch.Tensor().to(aux_num_fut.device)
            for j, layer in enumerate(self.linear_aux_fut):
                aux_emb_fut = layer(aux_num_fut[:,:,[j]]).unsqueeze(2)
                aux_emb_num_fut = torch.cat((aux_emb_num_fut, aux_emb_fut), dim=2)
            ## update summary about future
            if summary_fut is None:
                summary_fut = aux_emb_num_fut
            else:
                summary_fut = torch.cat((summary_fut, aux_emb_num_fut), dim=2)

        ### CATEGORICAL VARIABLES 
        summary_past, summary_fut = self.cat_categorical_vars(batch, summary_past, summary_fut)


        #! summary future could be none if only autoregressive
        # IF summary_fut is None change tactic?.
        # >>> PAST:
        summary_past = torch.mean(summary_past, dim=2)
        # >>> FUTURE:
        summary_fut = torch.mean(summary_fut, dim=2) # could be = None

        ### DIFFUSION INFERENCE
        # The process starts from a white noise, that will be modified by all subnets of the model
        B = summary_past.shape[0]
        z_t = self.generate_noise(batch_size = B).to(self.device) # [bayìtch_size, self.future_steps, self.output_channels]

        sqrt_alpha = np.sqrt(self.alpha) # auxiliary term, outside the for because beta now is constant
        # pass the white noise in sub nets
        for t in range(self.T-1, 0, -1): # INVERSE cycle over all subnets, but not the last one
            sub_net = self.sub_nets[t] # load the subnet
            alpha_t = self.alpha**t  # update
            computed_noise = sub_net(z_t, target_emb_num_past, summary_past, summary_fut)  

            z_t_hat = z_t / sqrt_alpha - self.beta * computed_noise / (np.sqrt(1 - alpha_t) * sqrt_alpha)
            noise = self.generate_noise(B).to(self.device)
            # next latent variable, here it is z_{t-1}
            z_t = z_t_hat + self.sigma * noise 

        sub_net = self.sub_nets[0] # last sub_net
        #here z_t = z_1
        if self.inference_out_from_sub_net:
            out = sub_net(z_t, target_emb_num_past, summary_past, summary_fut)
        else:   
            computed_noise_0 = sub_net(z_t, target_emb_num_past, summary_past, summary_fut)
            out = z_t/sqrt_alpha - np.sqrt(self.beta)*computed_noise_0/sqrt_alpha # last diffusion step without re-adding noise, rewritten considering alpha_1 = 1 - beta_1

        out = out.view(-1, self.future_steps, self.output_channels, self.mul)
        return out


    # function to concat embedded categorical variables
    def cat_categorical_vars(self, batch:dict, past_tensor_to_update:torch.Tensor=None, future_tensor_to_update:torch.Tensor=None) -> torch.Tensor:
        """Concatenate to past and future data about embedded categorical variables

        Args:
            batch (dict): Tensor where we extract available data
            past_tensor_to_update (torch.Tensor, optional): Tensor to which vars about past are added. Defaults to None.
            future_tensor_to_update (torch.Tensor, optional): Tensor to which vars about future are added. Defaults to None.

        Returns:
            torch.Tensor: updated tensor with embedded categorical variables 
        """
        if 'x_cat_past' in batch.keys() and 'x_cat_future' in batch.keys(): # if we have both
            # HERE WE ASSUME SAME NUMBER AND KIND OF VARIABLES IN PAST AND FUTURE
            cat_past = batch['x_cat_past'].to(self.device)
            cat_fut = batch['x_cat_future'].to(self.device)
            cat_full = torch.cat((cat_past, cat_fut), dim = 1)
            # EMB CATEGORICAL VARIABLES AND THEN SPLIT IN PAST AND FUTURE
            emb_cat_full = self.emb_cat_var(cat_full)
            cat_emb_past = emb_cat_full[:,:self.past_steps,:,:]
            cat_emb_fut = emb_cat_full[:,-self.future_steps:,:,:]
            
            ## update past
            # if we don't have a tensor to update, init a new one
            if past_tensor_to_update is None:
                update_past = cat_emb_past
            else:
                update_past =  torch.cat((past_tensor_to_update, cat_emb_past), dim=2)

            # update future
            # if we don't have a tensor to update, init a new one
            if future_tensor_to_update is None:
                update_future = cat_emb_fut
            else:
                update_future =  torch.cat((future_tensor_to_update, cat_emb_fut), dim=2)
        
            return update_past, update_future
        # in case we don't have information to concat
        else:
            # logging can be insert
            return past_tensor_to_update, future_tensor_to_update

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
    
    
    ### AUXILIARY MODEL FUNCS

    def perturbing_process(self, x, t_step):
        mean = 0.0
        std_dev = 1.0

        # Generating samples from the normal distribution
        eps_from_normal = torch.normal(mean=mean, std=std_dev, size=x.shape)
        y = x*np.sqrt(self.alpha**t_step) + np.sqrt(1-self.alpha**t_step)*eps_from_normal
        return y
    
    # not used
    def generate_sample(self, batch_size: int, means: float, stds: float)-> torch.Tensor:
        """Generate a 'batch_size' number of samples from a gaussian distribution with mean 'means' and std 'stds'

        Args:
            batch_size (int): -
            means (float): -
            stds (float): -

        Returns:
            torch.Tensor: tensor [batch_size, self.seq_len]
        """
        all_samples = []
        for i in range(batch_size):
            if isinstance(means,float):
                samples = torch.normal(mean=means, std=stds, size=(1, self.seq_len))
            else:
                samples = torch.normal(mean=means[i], std=stds[i], size=(1, self.seq_len))
            all_samples.append(samples)
        all_samples = torch.cat(all_samples, dim = 0)
        return all_samples

    def generate_noise(self, batch_size:int) -> torch.Tensor:
        """generate noise from normal distribution of size (batch_size, self.future_steps)

        Args:
            batch_size (int): number of different sequences generated

        Returns:
            torch.Tensor: concat of all sequences
        """
        all_eps = []
        for i in range(batch_size):
            eps = torch.normal(mean=0., std=1., size=(1, self.future_steps, self.output_channels * self.mul))
            all_eps.append(eps)
        all_eps = torch.cat(all_eps, dim = 0)
        return all_eps

class SubNet(nn.Module):
    def __init__(self, output_channel:int, quantiles:int, d_model:int, d_head:int, n_head:int, dropout_rate:float) -> None:
        """SUB NET of the Diffusion Model

        Due to a reparametrization of the task, the subnet is intended to compute the noise to be subtracted at each step to the input sequence.
        
        In this case we are using a MultiHead Attention + Residual Connection combination.
        Nets recovered from TFT folder.
        
        The Sub Net is equal for each diffusion step:
        - trained one per batch
        - all used sequentially for inference

        Args:
            output_channel (int): number of target variables
            quantiles (int): number of quantiles for each output_channel
            d_model (int): hidden dimension of the Net
            d_head (int): subNet dependent - dimension of each head
            n_head (int): subNet dependent - number of heads
            dropout_rate (float): 
        """
        super().__init__()
        self.target_in_linear = nn.Linear(output_channel*quantiles, d_model)
        self.attention = sub_nn.InterpretableMultiHead(d_model, d_head, n_head)
        self.res_conn = sub_nn.ResidualConnection(d_model, dropout_rate)
        self.target_out_linear = nn.Linear(d_model, output_channel*quantiles)


    def forward(self, y_noised:torch.Tensor, y_past:torch.Tensor, past_summary:torch.Tensor, future_summary:torch.Tensor)-> torch.Tensor:
        emb_y_noised = self.target_in_linear(y_noised)
        attention = self.attention(future_summary, past_summary, y_past)
        res_conn = self.res_conn(attention, emb_y_noised)
        emb_y_noised = emb_y_noised + res_conn
        out = self.target_out_linear(emb_y_noised)
        return out