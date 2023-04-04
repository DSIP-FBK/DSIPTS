
from torch import  nn
import torch



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

class Permute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.permute(input,(0,2,1))