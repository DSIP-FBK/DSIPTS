from enum import Enum
from typing import Union
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

try:
    from lightning.pytorch.callbacks import Callback
except:
    from pytorch_lightning import Callback
import torch
import os
import sys
import logging
from typing import Union

_logger = logging.getLogger('dsipts')
_fallback_handler = None

def _get_logger() -> logging.Logger:
    """Logger 'dsipts': INFO by default, propagates to root.
    If nobody configured logging (e.g. a notebook, no hydra) it prints to stdout by itself;
    as soon as the root logger gets a handler (hydra, logging.basicConfig) the fallback is removed
    so messages are not printed twice. Silence with logging.getLogger('dsipts').setLevel(logging.WARNING)
    """
    global _fallback_handler
    if _logger.level == logging.NOTSET:
        _logger.setLevel(logging.INFO)
    root_configured = len(logging.getLogger().handlers) > 0
    if _fallback_handler is None and not root_configured:
        _fallback_handler = logging.StreamHandler(sys.stdout)
        _fallback_handler.setFormatter(logging.Formatter('%(message)s'))
        _logger.addHandler(_fallback_handler)
        _logger.propagate = False
    elif _fallback_handler is not None and root_configured:
        _logger.removeHandler(_fallback_handler)
        _fallback_handler = None
        _logger.propagate = True
    return _logger

def beauty_string(message:str,type:str,verbose:bool):
    logger = _get_logger()
    size = 150
    if verbose is True:
        if type=='block':
            characters = len(message)
            border = max((100-characters)//2-5,0)
            logger.info('\n')
            logger.info(f"{'#'*size}")
            logger.info(f"{'#'*border}{' '*(size-border*2)}{'#'*border}")
            logger.info(f"{ message:^{size}}")
            logger.info(f"{'#'*border}{' '*(size-border*2)}{'#'*border}")
            logger.info(f"{'#'*size}")
        elif type=='section':
            logger.info('\n')
            logger.info(f"{'#'*size}")
            logger.info(f"{ message:^{size}}")
            logger.info(f"{'#'*size}")
        elif type=='info':
            logger.info(f"{ message:^{size}}")
        else:
            logger.info(message)
    
def extend_time_df(x:pd.DataFrame,freq:Union[str,int],group:Union[str,None]=None,global_minmax:bool=False)-> pd.DataFrame:
    """Utility for generating a full dataset and then merge the real data

    Args:
        x (pd.DataFrame): dataframe containing the column time
        freq (str): frequency (in pandas notation) of the resulting dataframe
        group (string or None): if not None the min max are computed by the group column, default None
        global_minmax (bool): if True the min_max is computed globally for each group. Usually used for stacked model
    Returns:
        pd.DataFrame: a dataframe with the column time ranging from thr minumum of x to the maximum with frequency `freq`
    """

    if group is None:

        if isinstance(freq,int):
            empty = pd.DataFrame({'time':list(range(x.time.min(),x.time.max(),freq))})
        else:
            empty = pd.DataFrame({'time':pd.date_range(x.time.min(),x.time.max(),freq=freq)})

    else:
        
        if global_minmax:
            _min = pd.DataFrame({group:x[group].unique(),'time':x.time.min()})
            _max = pd.DataFrame({group:x[group].unique(),'time':x.time.max()})

        else:
            _min = x.groupby(group).time.min().reset_index()
            _max = x.groupby(group).time.max().reset_index()
        empty = []
        for c in x[group].unique():
            if isinstance(freq,int):
                empty.append(pd.DataFrame({group:c,'time':np.arange(_min.time[_min[group]==c].values[0],_max.time[_max[group]==c].values[0],freq)}))

            else:
                empty.append(pd.DataFrame({group:c,'time':pd.date_range(_min.time[_min[group]==c].values[0],_max.time[_max[group]==c].values[0],freq=freq)}))
            
        empty = pd.concat(empty,ignore_index=True)
    return empty


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback.
    
    :meta private:
    """

    def __init__(self,dirpath):
        super().__init__()
        self.dirpath = dirpath
        self.metrics = {'val_loss':[],'train_loss':[]}

        

    def _align(self, losses):
        ##every validation metric (val_loss, val_select, ...) is trimmed like val_loss always was
        ##non so perche' le prime due le chiama prima del train
        n = len(losses['train_loss'])
        for c in losses:
            if c!='train_loss':
                losses[c] = losses[c][-n:] if n>0 else losses[c][2:]
        return losses

    def on_validation_end(self, trainer, pl_module):
        for c in trainer.callback_metrics:
            self.metrics.setdefault(c,[]).append(trainer.callback_metrics[c].item())
        ##Write csv in a convenient way
        losses = pd.DataFrame(self._align(self.metrics.copy()))
        losses.to_csv(os.path.join(self.dirpath,'loss.csv'),index=False)


    def on_train_end(self, trainer, pl_module):
        ##in place on purpose: TimeSeries.train_model reads mc.metrics afterwards
        losses = self._align(self.metrics)

        #losses['val_loss'] = losses['val_loss'][2:]
        losses = pd.DataFrame(losses)
        ##accrocchio per quando ci sono piu' gpu!
        losses.to_csv(os.path.join(self.dirpath,f'{np.random.randint(10000)}__losses__.csv'),index=False)
        print("Saving losses on file because multigpu not working")
       


class MyDataset(Dataset):

    def __init__(self, data:dict,t:np.array,groups:np.array,idx_target:Union[np.array,None],idx_target_future:Union[np.array,None])->torch.utils.data.Dataset:
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
            idx_target_future (Union[np.array,None]): you can specify the index in the future data that represent the input features (for differntial analysis or detrending strategies)

        Returns:
            torch.utils.data.Dataset: a torch Dataset to be used in a Dataloader
        """

        self.data = data
        self.t = t
        self.groups = groups
        self.idx_target = np.array(idx_target) if idx_target is not None else None
        self.idx_target_future = np.array(idx_target_future) if idx_target_future is not None else None
        self.sampler_weights = data['sampler_weights']
  

    def __len__(self):
        
        return len(self.data['x_num_past'])

    def __getitem__(self, idxs):
        sample = {}
        for k in self.data:
            if k!='sampler_weights':
                sample[k] = self.data[k][idxs]
        if self.idx_target is not None:
            sample['idx_target'] = self.idx_target
        if self.idx_target_future is not None:
            sample['idx_target_future'] = self.idx_target_future
        return sample

class ActionEnum(Enum):
    """action of categorical variable
    
    :meta private:
    """
    multiplicative: str = 'multiplicative'
    additive: str = 'additive'
