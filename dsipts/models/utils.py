import torch
import torch.nn.init as init
from torch import nn


ACTIVATIONS = {'selu': nn.SELU,
               'relu': nn.ReLU,
               'prelu': nn.PReLU,   
    }

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



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
    return ACTIVATIONS[activation.lower()]




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