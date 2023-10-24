import torch
import torch.nn.init as init
from torch import nn
import numpy as np
from numba import jit
from torch.autograd import Function


def get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables):
    message = f'Can {"NOT" if not handle_multivariate else "" }  handle multivariate output \n'\
                  f'Can {"NOT" if not handle_future_covariates else "" }  handle future covariates\n'\
                  f'Can {"NOT" if not handle_categorical_variables else "" }  handle categorical covariates'
    return message
    



class SinkhornDistance():
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def compute(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).to(x.device)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(x.device)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(x.device)

        u = torch.zeros_like(mu).to(x.device)
        v = torch.zeros_like(nu).to(x.device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost#, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

class QuantileLossMO(nn.Module):
    """Copied from git
    """
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):

        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        tot_loss = 0
        for j in range(preds.shape[2]):
            losses = []
            ##suppose BxLxCxMUL
            for i, q in enumerate(self.quantiles):
                errors = target[:,:,j] - preds[:,:,j, i]
                
                losses.append(torch.max((q-1) * errors,q * errors))

            loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
            tot_loss+=loss
        return loss



class L1Loss(nn.Module):
    """Custom L1Loss
    """
    def __init__(self):
        super().__init__()
        self.f = nn.L1Loss()
    def forward(self, preds, target):
        return self.f(preds[:,:,:,0],target)




class Permute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.permute(input,(0,2,1))
    
def get_activation(activation):
    return eval(activation)


def weight_init_zeros(m):
    
    if isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.constant_(param.data,0.0)
            else:
                init.constant_(param.data,0.0)
    elif isinstance(m, nn.Embedding):
        init.constant_(m.weight,0.0)             
        
    elif isinstance(m, nn.LayerNorm):
        init.zeros_(m.bias)
        init.ones_(m.weight)   
        
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.constant_(param.data,0.0)
            else:
                init.constant_(param.data,0.0)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.constant_(param.data,0.0)
            else:
                init.constant_(param.data,0.0)
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                bias.data[:n // 3].fill_(-1.)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.constant_(param.data,0.0)
            else:
                init.constant_(param.data,0.0)


    else:
        try:
            init.constant_(m.weight.data, 0.0)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        except:
            pass
            
def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                bias.data[:n // 3].fill_(-1.)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
                
    elif isinstance(m, nn.Embedding):
        init.normal_(m.weight, mean=0.0, std=0.02)             
        
    elif isinstance(m, nn.LayerNorm):
        init.zeros_(m.bias)
        init.ones_(m.weight)    
      
          #  if isinstance(module, nn.Linear):
          #      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          #      if module.bias is not None:
          #          torch.nn.init.zeros_(module.bias)




def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))

@jit(nopython = True)
def compute_softdtw(D, gamma):
  N = D.shape[0]
  M = D.shape[1]
  R = np.zeros((N + 2, M + 2)) + 1e8
  R[0, 0] = 0
  for j in range(1, M + 1):
    for i in range(1, N + 1):
      r0 = -R[i - 1, j - 1] / gamma
      r1 = -R[i - 1, j] / gamma
      r2 = -R[i, j - 1] / gamma
      rmax = max(max(r0, r1), r2)
      rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
      softmin = - gamma * (np.log(rsum) + rmax)
      R[i, j] = D[i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  N = D_.shape[0]
  M = D_.shape[1]
  D = np.zeros((N + 2, M + 2))
  E = np.zeros((N + 2, M + 2))
  D[1:N + 1, 1:M + 1] = D_
  E[-1, -1] = 1
  R[:, -1] = -1e8
  R[-1, :] = -1e8
  R[-1, -1] = R[-2, -2]
  for j in range(M, 0, -1):
    for i in range(N, 0, -1):
      a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
      b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
      c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
      a = np.exp(a0)
      b = np.exp(b0)
      c = np.exp(c0)
      E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
  return E[1:N + 1, 1:M + 1]
 

class SoftDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma = 1.0): # D.shape: [batch_size, N , N]
        dev = D.device
        batch_size,N,N = D.shape
        gamma = torch.FloatTensor([gamma]).to(dev)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()

        total_loss = 0
        R = torch.zeros((batch_size, N+2 ,N+2)).to(dev)   
        for k in range(0, batch_size): # loop over all D in the batch    
            Rk = torch.FloatTensor(compute_softdtw(D_[k,:,:], g_)).to(dev)
            R[k:k+1,:,:] = Rk
            total_loss = total_loss + Rk[-2,-2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss / batch_size
  
    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size,N,N = D.shape
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()

        E = torch.zeros((batch_size, N ,N)).to(dev) 
        for k in range(batch_size):         
            Ek = torch.FloatTensor(compute_softdtw_backward(D_[k,:,:], R_[k,:,:], g_)).to(dev)
            E[k:k+1,:,:] = Ek

        return grad_output * E, None





@jit(nopython = True)
def my_max(x, gamma):
    # use the log-sum-exp trick
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z

@jit(nopython = True)
def my_min(x,gamma) :
    min_x, argmax_x = my_max(-x, gamma)
    return - min_x, argmax_x

@jit(nopython = True)
def my_max_hessian_product(p, z, gamma):
    return  ( p * z - p * np.sum(p * z) ) /gamma

@jit(nopython = True)
def my_min_hessian_product(p, z, gamma):
    return - my_max_hessian_product(p, z, gamma)


@jit(nopython = True)
def dtw_grad(theta, gamma):
    m = theta.shape[0]
    n = theta.shape[1]
    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            v, Q[i, j] = my_min(np.array([V[i, j - 1],
                                                V[i - 1, j - 1],
                                                V[i - 1, j]]) , gamma)
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, :] = 0
    E[:, n + 1] = 0
    E[m + 1, n + 1] = 1
    Q[m + 1, n + 1] = 1

    for i in range(m,0,-1):
        for j in range(n,0,-1):
            E[i, j] = Q[i, j + 1, 0] * E[i, j + 1] + \
                      Q[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                      Q[i + 1, j, 2] * E[i + 1, j]
    
    return V[m, n], E[1:m + 1, 1:n + 1], Q, E


@jit(nopython = True)
def dtw_hessian_prod(theta, Z, Q, E, gamma):
    m = Z.shape[0]
    n = Z.shape[1]

    V_dot = np.zeros((m + 1, n + 1))
    V_dot[0, 0] = 0

    Q_dot = np.zeros((m + 2, n + 2, 3))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            V_dot[i, j] = Z[i - 1, j - 1] + \
                          Q[i, j, 0] * V_dot[i, j - 1] + \
                          Q[i, j, 1] * V_dot[i - 1, j - 1] + \
                          Q[i, j, 2] * V_dot[i - 1, j]

            v = np.array([V_dot[i, j - 1], V_dot[i - 1, j - 1], V_dot[i - 1, j]])
            Q_dot[i, j] = my_min_hessian_product(Q[i, j], v, gamma)
    E_dot = np.zeros((m + 2, n + 2))

    for j in range(n,0,-1):
        for i in range(m,0,-1):
            E_dot[i, j] = Q_dot[i, j + 1, 0] * E[i, j + 1] + \
                          Q[i, j + 1, 0] * E_dot[i, j + 1] + \
                          Q_dot[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                          Q[i + 1, j + 1, 1] * E_dot[i + 1, j + 1] + \
                          Q_dot[i + 1, j, 2] * E[i + 1, j] + \
                          Q[i + 1, j, 2] * E_dot[i + 1, j]

    return V_dot[m, n], E_dot[1:m + 1, 1:n + 1]


class PathDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma): # D.shape: [batch_size, N , N]
        batch_size,N,N = D.shape
        device = D.device
        D_cpu = D.detach().cpu().numpy()
        gamma_gpu = torch.FloatTensor([gamma]).to(device)
        
        grad_gpu = torch.zeros((batch_size, N ,N)).to(device)
        Q_gpu = torch.zeros((batch_size, N+2 ,N+2,3)).to(device)
        E_gpu = torch.zeros((batch_size, N+2 ,N+2)).to(device)  
        
        for k in range(0,batch_size): # loop over all D in the batch    
            _, grad_cpu_k, Q_cpu_k, E_cpu_k = dtw_grad(D_cpu[k,:,:], gamma)     
            grad_gpu[k,:,:] = torch.FloatTensor(grad_cpu_k).to(device)
            Q_gpu[k,:,:,:] = torch.FloatTensor(Q_cpu_k).to(device)
            E_gpu[k,:,:] = torch.FloatTensor(E_cpu_k).to(device)
        ctx.save_for_backward(grad_gpu,D, Q_gpu ,E_gpu, gamma_gpu) 
        return torch.mean(grad_gpu, dim=0) 
    
    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        grad_gpu, D_gpu, Q_gpu, E_gpu, gamma = ctx.saved_tensors
        D_cpu = D_gpu.detach().cpu().numpy()
        Q_cpu = Q_gpu.detach().cpu().numpy()
        E_cpu = E_gpu.detach().cpu().numpy()
        gamma = gamma.detach().cpu().numpy()[0]
        Z = grad_output.detach().cpu().numpy()
        
        batch_size,N,N = D_cpu.shape
        Hessian = torch.zeros((batch_size, N ,N)).to(device)
        for k in range(0,batch_size):
            _, hess_k = dtw_hessian_prod(D_cpu[k,:,:], Z, Q_cpu[k,:,:,:], E_cpu[k,:,:], gamma)
            Hessian[k:k+1,:,:] = torch.FloatTensor(hess_k).to(device)

        return  Hessian, None
    
