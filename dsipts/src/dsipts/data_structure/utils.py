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
    """PyTorch Lightning metric callback: one row per epoch with train_loss, val_loss, lr and the other validation metrics.

    - The sanity-check validation (run before training) is skipped.
    - Lightning logs the train loss of epoch k after the validation of epoch k, so it is read at the next validation
      (as the previous epoch's) and, for the last epoch, at the end of the training.
    - ``self.metrics`` is a dict of equal-length lists (columns), as TimeSeries.train_model expects; with
      max_epochs=1 it has one row (before, the columns had different lengths and the DataFrame failed).
    - ``lr`` is the learning rate of the first optimizer param group at the validation of the epoch: for a per-step
      scheduler (cosine) the lr after the last step of the epoch, for a per-epoch StepLR the lr used in that epoch
      (it steps after the validation). To check a schedule, see models/schedulers.py.

    :meta private:
    """

    def __init__(self,dirpath):
        super().__init__()
        self.dirpath = dirpath
        self.rows = {}  ##epoch -> {metric: value}
        self.metrics = {'val_loss':[],'train_loss':[]}

    def _columns(self):
        epochs = sorted(self.rows)
        keys = ['val_loss','train_loss'] + sorted({k for r in self.rows.values() for k in r} - {'val_loss','train_loss'})
        return {k: [self.rows[e].get(k, np.nan) for e in epochs] for k in keys}

    def _set(self, epoch, key, value):
        if epoch >= 0:
            self.rows.setdefault(epoch, {})[key] = value

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if trainer.optimizers:
            self._set(epoch, 'lr', float(trainer.optimizers[0].param_groups[0]['lr']))
        for c, v in trainer.callback_metrics.items():
            if c == 'train_loss':
                self._set(epoch - 1, c, v.item())  ##logged at the end of the previous training epoch
            else:
                self._set(epoch, c, v.item())
        self.metrics = self._columns()
        ##Write csv in a convenient way
        pd.DataFrame(self.metrics).to_csv(os.path.join(self.dirpath,'loss.csv'),index=False)

    def on_train_end(self, trainer, pl_module):
        if 'train_loss' in trainer.callback_metrics and self.rows:
            self._set(max(self.rows), 'train_loss', trainer.callback_metrics['train_loss'].item())
        ##TimeSeries.train_model reads mc.metrics afterwards
        self.metrics = self._columns()
        losses = pd.DataFrame(self.metrics)
        losses.to_csv(os.path.join(self.dirpath,'loss.csv'),index=False)  ##now with the last train_loss too
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


class LazyDataset(Dataset):
    """Same samples as `MyDataset`, but the windows are cut on the fly instead of being copied in memory.

    `MyDataset` stores every window (past_steps + future_steps rows), so with overlapping windows each row is copied
    many times (e.g. 3072 + 288 steps with one window per day: ~35 copies). Here every group's arrays are stored once
    (already normalized, float32 / int32) and a sample is only (group, position): `__getitem__` slices the arrays.
    A batch has the same keys, shapes, dtypes and values as with `MyDataset`.

    Not available in lazy mode: `.data` (the stacked arrays, used by the modifiers).

    Args:
        arrays (list): one dict per group with the per-row arrays: 'x_num_past', 'y', 't' and, if present,
            'x_num_future', 'x_cat_past', 'x_cat_future', 'loss_weights'
        labels (list): the group label of each element of `arrays`
        gid (np.array): group (index in `arrays`) of each sample
        pos (np.array): position i of each sample (first future row = i + skip_stacked)
        past_steps (int), future_steps (int), shift (int), skip_stacked (int), keep_entire_seq_while_shifting (bool):
            as in `TimeSeries.create_data_loader`
        sampler_weights (np.array): weight of each sample
        idx_target (Union[np.array,None]), idx_target_future (Union[np.array,None]): as in `MyDataset`
    """

    def __init__(self, arrays: list, labels: list, gid: np.ndarray, pos: np.ndarray, past_steps: int, future_steps: int,
                 shift: int, skip_stacked: int, keep_entire_seq_while_shifting: bool, sampler_weights: np.ndarray,
                 idx_target: Union[np.array, None], idx_target_future: Union[np.array, None]):
        self.arrays = arrays
        self.labels = np.asarray(labels)
        self.gid = np.asarray(gid, dtype=np.int64)
        self.pos = np.asarray(pos, dtype=np.int64)
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.shift = shift
        self.skip_stacked = skip_stacked
        self.keep_entire_seq_while_shifting = keep_entire_seq_while_shifting
        self.sampler_weights = np.asarray(sampler_weights, dtype=np.float32)
        self.idx_target = np.array(idx_target) if idx_target is not None else None
        self.idx_target_future = np.array(idx_target_future) if idx_target_future is not None else None
        ##same key order as MyDataset
        keys = ['y', 'x_num_past', 'x_cat_past', 'x_cat_future', 'x_num_future', 'loss_weights']
        self.keys = [k for k in keys if len(arrays) > 0 and k in arrays[0]]

    def subset(self, sel) -> 'LazyDataset':
        """The samples `sel` (indices or slice), sharing the same arrays (no copy). Used for continual learning."""
        return LazyDataset(self.arrays, self.labels, self.gid[sel], self.pos[sel], self.past_steps, self.future_steps,
                           self.shift, self.skip_stacked, self.keep_entire_seq_while_shifting, self.sampler_weights[sel],
                           self.idx_target, self.idx_target_future)

    def _slice(self, k: str, i: int) -> slice:
        ss = self.skip_stacked
        if k in ('x_num_past', 'x_cat_past'):
            return slice(i - self.past_steps, i)
        if k in ('x_num_future', 'x_cat_future'):
            end = i + self.future_steps + ss if self.keep_entire_seq_while_shifting else i + self.future_steps - self.shift + ss
            return slice(i - self.shift + ss, end)
        return slice(i + ss, i + self.future_steps + ss)  # y, loss_weights, t

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idxs):
        if np.ndim(idxs) == 0:
            a, i = self.arrays[self.gid[idxs]], self.pos[idxs]
            sample = {k: a[k][self._slice(k, i)] for k in self.keys}
        else:  ##batch of indices (MyDataset supports it through numpy indexing)
            items = [self[int(j)] for j in np.arange(len(self))[idxs]]
            sample = {k: np.stack([it[k] for it in items]) for k in self.keys}
        if self.idx_target is not None:
            sample['idx_target'] = self.idx_target
        if self.idx_target_future is not None:
            sample['idx_target_future'] = self.idx_target_future
        return sample

    @property
    def t(self) -> np.ndarray:
        """Time of the target rows of each sample (n_samples x future_steps), as `MyDataset.t`; built on demand."""
        if len(self) == 0:
            return np.empty((0, self.future_steps))
        return np.stack([self.arrays[g]['t'][self._slice('t', i)] for g, i in zip(self.gid, self.pos)])

    @property
    def groups(self) -> np.ndarray:
        """Group label of each sample, as `MyDataset.groups`."""
        return self.labels[self.gid]

    @property
    def data(self):
        raise AttributeError('LazyDataset does not store the stacked windows (.data): use split_for_train(lazy=False)')


class ActionEnum(Enum):
    """action of categorical variable
    
    :meta private:
    """
    multiplicative: str = 'multiplicative'
    additive: str = 'additive'
