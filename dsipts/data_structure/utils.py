from enum import Enum
import pandas,numpy
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from pytorch_lightning import Callback
from typing import Union, List
import torch
import os
import logging
def beauty_string(message:str,type:str):
    
    size = 150

    if type=='block':
        characters = len(message)
        border = max((100-characters)//2-5,0)
        logging.info('\n')
        logging.info(f"{'#'*size}")
        logging.info(f"{'#'*border}{' '*(size-border*2)}{'#'*border}")
        logging.info(f"{ message:^{size}}")
        logging.info(f"{'#'*border}{' '*(size-border*2)}{'#'*border}")
        logging.info(f"{'#'*size}")
    elif type=='section':
        logging.info('\n')
        logging.info(f"{'#'*size}")
        logging.info(f"{ message:^{size}}")
        logging.info(f"{'#'*size}")
    elif type=='info':
        logging.info(f"{ message:^{size}}")
    else:
        logging.info(message)



def extend_time_df(x:pd.DataFrame,freq:str,group:Union[str,None]=None,global_minmax:bool=False)-> pandas.DataFrame:
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

        

    def on_validation_end(self, trainer, pl_module):
        for c in trainer.callback_metrics:
            self.metrics[c].append(trainer.callback_metrics[c].item())
        ##Write csv in a convenient way
        tmp  = self.metrics.copy()
        tmp['val_loss'] = tmp['val_loss'][2:]
        losses = pd.DataFrame(tmp)
        losses.to_csv(os.path.join(self.dirpath,'loss.csv'),index=False)

        
    def on_train_end(self, trainer, pl_module):
        losses = self.metrics
        ##non so perche' le prime due le chiama prima del train
        losses['val_loss'] = losses['val_loss'][2:]
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
            idx_target (Union[np.array,None]): you can specify the index in the future data that represent the input features (for differntial analysis or detrending strategies)

        Returns:
            torch.utils.data.Dataset: a torch Dataset to be used in a Dataloader
        """
        self.data = data
        self.t = t
        self.groups = groups
        self.idx_target = np.array(idx_target) if idx_target is not None else None
        self.idx_target_future = np.array(idx_target_future) if idx_target_future is not None else None

    def __len__(self):
        
        return len(self.data['y'])

    def __getitem__(self, idxs):
        sample = {}
        for k in self.data:
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
