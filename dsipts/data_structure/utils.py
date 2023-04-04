from enum import Enum
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from pytorch_lightning import Callback
from typing import Union
import torch


def extend_df(x:Union[pd.Series, np.array],freq:str)-> pd.DataFrame:
    """Utility for generating a full dataset and then merge the real data

    Args:
        x (Union[pd.Series, np.array]): array indicating the time
        freq (str): frequency (in pandas notation) of the resulting dataframe

    Returns:
        pd.DataFrame: a dataframe with the column time ranging from thr minumum of x to the maximum with frequency `freq`
    """

    empty = pd.DataFrame({'time':pd.date_range(x.min(),x.max(),freq=freq)})
    return empty


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback.
    
    :meta private:
    """

    def __init__(self):
        super().__init__()
        self.metrics = {'val_loss':[],'train_loss':[]}

        

    def on_validation_end(self, trainer, pl_module):
        for c in trainer.callback_metrics:
            self.metrics[c].append(trainer.callback_metrics[c].item())

    def on_train_end(self, trainer, pl_module):
        losses = self.metrics
        ##non so perche' le prime due le chiama prima del train
        losses['val_loss'] = losses['val_loss'][2:]
        losses = pd.DataFrame(losses)
        ##accrocchio per quando ci sono piu' gpu!
        losses.to_csv(f'{np.random.randint(10000)}__losses__.csv',index=False)
        print("Saving losses on file because multigpu not working")
       


class MyDataset(Dataset):

    def __init__(self, data:dict,t:np.array,idx_target:Union[np.array,None])->torch.utils.data.Dataset:
        """
            Extension of Dataset class. While training the returned item is a batch containing the standard keys

        Args:
            data (dict): a dictionary. Each key is a np.array containing the data. The keys are:
                y : the target variable(s)
                x_num_past: the numerical past variables
                x_num_future: the numerical future variables
                x_cat_past: the categorical past variables
                x_cat_future: the categorical future variables
                idx_target: index of target features in the past array
            t (np.array): the time array related to the target variables
            idx_target (Union[np.array,None]): you can specify the index in the past data that represent the input features (for differntial analysis or detrending strategies)

        Returns:
            torch.utils.data.Dataset: a torch Dataset to be used in a Dataloader
        """
        self.data = data
        self.t = t
        self.idx_target = np.array(idx_target) if idx_target is not None else None
    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idxs):
        sample = {}
        for k in self.data:
            sample[k] = self.data[k][idxs]
        if self.idx_target is not None:
            sample['idx_target'] = self.idx_target
        return sample

class ActionEnum(Enum):
    """action of categorical variable
    
    :meta private:
    """
    multiplicative: str = 'multiplicative'
    additive: str = 'additive'
