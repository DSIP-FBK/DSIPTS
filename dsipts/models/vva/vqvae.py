import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations, product
import logging

class VectorQuantizer(nn.Module):
    """
    Inspired from Sonnet implementation of VQ-VAE https://arxiv.org/abs/1711.00937,
    in https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py and
    pytorch implementation of it from zalandoresearch in https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb.

    Implements the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937

    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
            quantized space. Inputs to the modules must be in this format as well.
        num_embeddings: integer, the number of vectors in the quantized space.
            commitment_cost: scalar which controls the weighting of the loss terms
            (see equation 4 in the paper - this variable is Beta).
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, device):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

        self._commitment_cost = commitment_cost
        self._device = device

    def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False):
        """
        Connects the module to some inputs.

        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.

        Returns:
            loss: Tensor containing the loss to optimize.
            quantize: Tensor containing the quantized version of the input.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
            distances
        """

        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(1, 2, 0).contiguous()
        input_shape = inputs.shape
        _, time, batch_size = input_shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Compute distances between encoded audio frames and embedding vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, dtype=torch.float).to(self._device)
        encodings.scatter_(1, encoding_indices, 1)

        # Compute distances between encoding vectors
        if not self.training and compute_distances_if_possible:
            _encoding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in combinations(flat_input, r=2)]
            encoding_distances = torch.tensor(_encoding_distances).to(self._device).view(batch_size, -1)
        else:
            encoding_distances = None

        # Compute distances between embedding vectors
        if not self.training and compute_distances_if_possible:
            _embedding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in combinations(self._embedding.weight, r=2)]
            embedding_distances = torch.tensor(_embedding_distances).to(self._device)
        else:
            embedding_distances = None

        # Sample nearest embedding
        if not self.training and compute_distances_if_possible:
            _frames_vs_embedding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in product(flat_input, self._embedding.weight.detach())]
            frames_vs_embedding_distances = torch.tensor(_frames_vs_embedding_distances).to(self._device).view(batch_size, time, -1)
        else:
            frames_vs_embedding_distances = None

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # TODO: Check if the more readable self._embedding.weight.index_select(dim=1, index=encoding_indices) works better

        concatenated_quantized = self._embedding.weight[torch.argmin(distances, dim=1).detach().cpu()] if not self.training or record_codebook_stats else None

        # Losses
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        commitment_loss = self._commitment_cost * e_latent_loss
        vq_loss = q_latent_loss + commitment_loss

        quantized = inputs + (quantized - inputs).detach() # Trick to prevent backpropagation of quantized
        avg_probs = torch.mean(encodings, dim=0)

        """
        The perplexity a useful value to track during training.
        It indicates how many codes are 'active' on average.
        """
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) # Exponential entropy

        # Convert quantized from BHWC -> BCHW
        return vq_loss, quantized.permute(2, 0, 1).contiguous(), \
            perplexity, encodings.view(batch_size, time, -1), \
            distances.view(batch_size, time, -1), encoding_indices, \
            {'e_latent_loss': e_latent_loss.item(), 'q_latent_loss': q_latent_loss.item(),
            'commitment_loss': commitment_loss.item(), 'vq_loss': vq_loss.item()}, \
            encoding_distances, embedding_distances, frames_vs_embedding_distances, concatenated_quantized

    @property
    def embedding(self):
        return self._embedding


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings.view(input_shape[0],-1,encodings.shape[1])
    
    
class Residual(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_residual_hiddens):
        super(Residual, self).__init__()
        
        relu_1 = nn.ReLU(True)
        conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )


        relu_2 = nn.ReLU(True)
        conv_2 = nn.Conv1d(
            in_channels=num_residual_hiddens,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )


        # All parameters same as specified in the paper
        self._block = nn.Sequential(
            relu_1,
            conv_1,
            relu_2,
            conv_2
        )
    
    def forward(self, x):
        return x + self._block(x)
    
class ResidualStack(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, hidden_channels, num_residual_hiddens)] * self._num_residual_layers)
        
    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_residual_layers=3,padding_middle=1):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=hidden_channels,
                                 kernel_size=3, padding=1)
        
        self._conv_2 = nn.Conv1d(in_channels=hidden_channels,
                                 out_channels=hidden_channels,
                                 kernel_size=3,padding=1)

        self._conv_3 = nn.Conv1d(in_channels=hidden_channels,
                                 out_channels=hidden_channels,
                                 kernel_size=4,
                                 stride=2, padding=padding_middle)
        
        self._conv_4 = nn.Conv1d(in_channels=hidden_channels,
                                 out_channels=hidden_channels,
                                 kernel_size=3,
                                 padding=1)
        
        self._conv_5 = nn.Conv1d(in_channels=hidden_channels,
                                 out_channels=hidden_channels,
                                 kernel_size=3, padding=1)

        self._residual_stack = ResidualStack(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                num_residual_layers=num_residual_layers,
                num_residual_hiddens=hidden_channels//2
        )
    def forward(self, inputs):
        x_conv_1 = F.relu(self._conv_1(inputs))
        x = F.relu(self._conv_2(x_conv_1)) + x_conv_1
        x_conv_3 = F.relu(self._conv_3(x))
        x_conv_4 = F.relu(self._conv_4(x_conv_3)) + x_conv_3
        x_conv_5 = F.relu(self._conv_5(x_conv_4)) + x_conv_4
        x = self._residual_stack(x_conv_5) + x_conv_5
        

        return x
    
    

class Jitter(nn.Module):
    """
    Jitter implementation from [Chorowski et al., 2019].
    During training, each latent vector can replace either one or both of
    its neighbors. As in dropout, this prevents the model from
    relying on consistency across groups of tokens. Additionally,
    this regularization also promotes latent representation stability
    over time: a latent vector extracted at time step t must strive
    to also be useful at time steps t − 1 or t + 1.
    """

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self._probability = probability

    def forward(self, quantized):
        original_quantized = quantized.detach().clone()
        length = original_quantized.size(2)
        for i in range(length):
            """
            Each latent vector is replace with either of its neighbors with a certain probability
            (0.12 from the paper).
            """
            replace = [True, False][np.random.choice([1, 0], p=[self._probability, 1 - self._probability])]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    """
                    "We independently sample whether it is to
                    be replaced with the token right after
                    or before it."
                    """
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])
                quantized[:, :, i] = original_quantized[:, :, neighbor_index]

        return quantized
    
    
class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels,num_residual_layers=3):
        super(Decoder, self).__init__()
        
        
        self._jitter = Jitter(0.125)
        
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=hidden_channels,
                                 kernel_size=3, 
                                 padding=1)
        
        #self._upsample = nn.Upsample(scale_factor=2)
        
        self._residual_stack = ResidualStack(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=hidden_channels//2
        )
        
        self._conv_trans_1 = nn.ConvTranspose1d(in_channels=hidden_channels, 
                                                out_channels=hidden_channels,
                                                kernel_size=3,  padding=0)
        
        self._conv_trans_2 = nn.ConvTranspose1d(
                                                in_channels=hidden_channels, 
                                                out_channels=hidden_channels,
                                                kernel_size=4,padding=0)
        
        self._conv_trans_3 = nn.ConvTranspose1d(in_channels=hidden_channels, 
                                                out_channels=out_channels,
                                                kernel_size=4,  padding=0)

    def forward(self, x,is_training=True):
        if is_training:
             x = self._jitter(x)
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))
        x = F.relu(self._conv_trans_2(x))
        x = self._conv_trans_3(x)
        return x
    
class VQVAE(nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels,num_embeddings, embedding_dim, commitment_cost, decay):
        super(VQVAE, self).__init__()
        
        self._encoder = Encoder(in_channels, hidden_channels)
        self._pre_vq_conv = nn.Conv1d(in_channels=hidden_channels, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            logging.info('CARE NOT TESTED')
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(in_channels=embedding_dim,hidden_channels=hidden_channels,out_channels=out_channels)


    def forward(self, x,is_training=True):
        z = self._encoder(x)

        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, encodings = self._vq_vae(z)
        x_recon = self._decoder(quantized,is_training)

        return loss, x_recon, perplexity,quantized,encodings