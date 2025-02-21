
# %%
#dataset.to_csv('dataset.csv',index=False)

# %%
!pip install lightning

# %%
import pickle 
import pandas as pd
from typing import Union,List, Optional, Dict
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import lightning as L
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import torch
from datetime import timedelta
with open('example.pkl', 'rb') as f:
    dataset,cat_var_past,cat_var_fut,num_var_past = pickle.load(f)
num_var_fut = []
group = None
time_var = 'time'
target = ['signal']

# %%
dataset2 = dataset.copy()
dataset2['group'] = 2
dataset['group'] = 1 
dataset = pd.concat([dataset, dataset2])

# %%
def extend_time_df(x:pd.DataFrame,time:str,freq:Union[str,int],group:Union[List[str],None]=None,global_minmax:bool=False)-> pd.DataFrame:
  

    if group is None:

        if isinstance(freq,int):
            empty = pd.DataFrame({'time':list(range(x[time].min(),x[time].max(),freq))})
        else:
            empty = pd.DataFrame({'time':pd.date_range(x[time].min(),x[time].max(),freq=freq)})

    else:
        
        if global_minmax:
            _min = x[time].min()
            _max = x[time].max()
            _min_max = pd.DataFrame({'min':min, 'max':max})
            for c in group:
               _min_max[c] = _min_max[c]
        else:
            _min = x.groupby(group)[time].min().reset_index().rename(columns={time:'min'})
            _max = x.groupby(group)[time].max().reset_index().rename(columns={time:'max'})
            _min_max = pd.merge(_min,_max)
        empty = []
        for _,row in _min_max.iterrows():
            if isinstance(freq,int):
                tmp = pd.DataFrame({time:np.arange(row['min'],row['max'],freq)})
            else:
                tmp = pd.DataFrame({time:pd.date_range(row['min'],row['max'],freq=freq)})
            for c in group:
               tmp[c] = row[c]
            empty.append(tmp)

        empty = pd.concat(empty,ignore_index=True)
    return empty


# %%
##TODO @Sandeep please try to replicate what is in PandasTSDataSet but for a smore compact example!
class PandasTSDataSet_MINIMAL:


    def __init__(
        self,
        data: pd.DataFrame,
        time: Optional[str] = None,
        target: Optional[Union[str, List[str]]] = None,
        num: Optional[List[Union[str, List[str]]]] = None,
        
    ):

        
        self.time = time
        self.target = _coerce_to_list(target)
        self.num = _coerce_to_list(num)
        self.data = data.copy()

        
        self.feature_cols = self.num 
        #use set ensuring unique columns
        self.data = data[list(set(self.feature_cols + self.target + [self.time]))].copy()
        

        self.label_encoders = {}

        ##ensure to have a coherent dataset
        self.data.drop_duplicates(subset= [time],keep='first',inplace=True,ignore_index=True)

        ##compute minumum frequency
        freq = self.data['time'].diff.min()
      

        if isinstance(freq, timedelta):
            freq = pd.to_timedelta(freq)   
        elif isinstance(freq,  (int, float)):
            freq = int(freq)
        else:
            raise TypeError("time must be integer or datetime")
        

        ##extend dataset
        self.data = extend_time_df(self.data,self.time,freq,None).merge(self.data,how='left').reset_index()

        
        self._groups = {"_single_group": self.data.index}
        self._group_ids = ["_single_group"]

        ## we know on the fly which rows are valid and wich contains nans
        self.data['valid'] = ~pd.isnull(self.data.max(axis=1)) 
        
        self._prepare_metadata()

    def _prepare_metadata(self):
        """Prepare metadata for the dataset."""
        self.metadata = {
            "cols": {
                "y": self.target,
                "x": self.feature_cols,
                "st": [],
            },
            "col_type": {},
            "col_known": {},
            "cat_index":[]
        }
        for c in self.feature_cols:
            if c in self.cat:
                self.metadata['cat_index'].append(i)
                
     
        all_cols = self.target + self.feature_cols 
        for col in all_cols:
            self.metadata["col_type"][col] = "F"

            self.metadata["col_known"][col] = "K" if col in self.feature_cols else "U"
        self.metadata['encoders'] = self.label_encoders
            
    def __len__(self) -> int:
        """Return number of time series in the dataset."""
        return len(self._group_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]: ##NOT USED INDEX!
        """Get time series data for given index."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

     
        data = self.data.sort_values(by=self.time)

        result = {
            "t": data[self.time].values,
            "y": torch.tensor(data[self.target].values,dtype=torch.float32,device=device),
            "x": torch.tensor(data[self.feature_cols].values,dtype=torch.float32,device=device),
            'is_valid': torch.tensor(data.valid.values,dtype=torch.int,device=device),
        }

       


        return result

    def get_metadata(self) -> Dict:
        """Return metadata about the dataset.

        Returns
        -------
        Dict
            Dictionary containing:
            - cols: column names for y, x, and static features
            - col_type: mapping of columns to their types (F/C)
            - col_known: mapping of columns to their future known status (K/U)
        """
        return self.metadata

def _coerce_to_list(obj):
    """Coerce object to list.

    None is coerced to empty list, otherwise list constructor is used.
    Single values are coerced into list
    """
    if obj is None:
        return []
    if isinstance(obj,list):
        return obj
    else:
        return [obj]

# %%

class PandasTSDataSet:


    def __init__(
        self,
        data: pd.DataFrame,
        time: Optional[str] = None,
        target: Optional[Union[str, List[str]]] = None,
        group: Optional[List[str]] = None,
        num: Optional[List[Union[str, List[str]]]] = None,
        cat: Optional[List[Union[str, List[str]]]] = None,
        known: Optional[List[Union[str, List[str]]]] = None,
        unknown: Optional[List[Union[str, List[str]]]] = None,
        static: Optional[List[Union[str, List[str]]]] = None,
        label_encoders: Optional[dict] = None
    ):

        
        self.time = time
        self.target = _coerce_to_list(target)
        self.num = _coerce_to_list(num)
        self.cat = _coerce_to_list(cat)
        self.known = _coerce_to_list(known)
        self.unknown = _coerce_to_list(unknown)
        self.static = _coerce_to_list(static)
        self.data = data.copy()

        self.group = _coerce_to_list(group)

        
        self.feature_cols = self.num + self.cat 
        #use set ensuring unique columns
        self.data = data[list(set(self.static + self.feature_cols + self.target + [self.time] +self.group))].copy()
        
        ##Encoders for categorical since we want to return tensors
        if label_encoders is None:
            label_encoders = {}
            ##Encoders for categorical since we want to return tensors
            for c in self.cat+self.group:
                label_encoders[c] = OrdinalEncoder()
                self.data[c] = label_encoders[c].fit_transform(self.data[c].values.reshape(-1,1)).flatten()
            self.label_encoders = label_encoders
        else:
            for c in self.cat+self.group:
                self.data[c] = label_encoders[c].transform(self.data[c].values.reshape(-1,1)).flatten()
            self.label_encoders = label_encoders

        
        ##ensure to have a coherent dataset
        self.data.drop_duplicates(subset= self.group+[self.time],keep='first',inplace=True,ignore_index=True)

        ##compute minumum frequency
        if self.group is None:
            freq = self.data[self.time].diff.min()
        else:
            freq = self.data.groupby(self.group).time.diff().min()

        if isinstance(freq, timedelta):
            freq = pd.to_timedelta(freq)   
        elif isinstance(freq,  (int, float)):
            freq = int(freq)
        else:
            raise TypeError("time must be integer or datetime")
        

        ##extend dataset
        self.data = extend_time_df(self.data,self.time,freq,self.group).merge(self.data,how='left').reset_index()

        
        ##now we are sure that data is in a coherent form!
        self.lengths  = {}
        if self.group:
            self._groups = self.data.groupby(self.group).groups
            self._group_ids =  list(self._groups.keys())
            for k in self._groups:
                self.lengths[k] = len(self._groups[k])
        else:
            self._groups = {0: self.data.index}
            self._group_ids = [0]
            self.lengths[0] = len( self.data.index)


        
        ## we know on the fly which rows are valid and wich contains nans
        self.data['valid'] = ~pd.isnull(self.data.max(axis=1)) 
        
        self._prepare_metadata()

    def _prepare_metadata(self):
        """Prepare metadata for the dataset."""
        self.metadata = {
            "cols": {
                "y": self.target,
                "x": self.feature_cols,
                "st": self.static,
            },
            "col_type": {},
            "col_known": {},
            "cat_index":[]
        }
        for c in self.feature_cols:
            if c in self.cat:
                self.metadata['cat_index'].append(i)
                
        self.metadata["col_known"]
        all_cols = self.target + self.feature_cols + self.static
        for col in all_cols:
            self.metadata["col_type"][col] = "C" if col in self.cat else "F"

            self.metadata["col_known"][col] = "K" if col in self.known else "U"
        self.metadata['encoders'] = self.label_encoders
            
    def __len__(self) -> int:
        """Return number of time series in the dataset."""
        return len(self._group_ids)

    def get_id_ts_by_idx(self,idx):
        
        tmp = np.cumsum(list(self.lengths.values()))
        idx =  min(np.where(tmp>idx)[0])
        return idx, 0 if idx==0 else tmp[idx-1]
    
    def get_total_len(self):
        l = 0
        for k in  self._groups:
            l+= self.lengths[k]
        return l
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get time series data for given index."""
        group_id = self._group_ids[index]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.group:
            mask = self._groups[group_id]
            data = self.data.loc[mask].sort_values(by=self.time)
        else:
            data = self.data.sort_values(by=self.time)

        result = {
            "t": data[self.time].values,
            "y": torch.tensor(data[self.target].values,dtype=torch.float32,device=device),
            "x": torch.tensor(data[self.feature_cols].values,dtype=torch.float32,device=device),
            "group": torch.tensor([group_id],dtype=torch.float32,device=device),
            "st": torch.tensor(data[self.static].iloc[0].values if self.static else []),
            'is_valid': torch.tensor(data.valid.values,dtype=torch.int,device=device),
        }

       


        return result

    def get_metadata(self) -> Dict:
        """Return metadata about the dataset.

        Returns
        -------
        Dict
            Dictionary containing:
            - cols: column names for y, x, and static features
            - col_type: mapping of columns to their types (F/C)
            - col_known: mapping of columns to their future known status (K/U)
        """
        return self.metadata

def _coerce_to_list(obj):
    """Coerce object to list.

    None is coerced to empty list, otherwise list constructor is used.
    Single values are coerced into list
    """
    if obj is None:
        return []
    if isinstance(obj,list):
        return obj
    else:
        return [obj]

# %%
data_module_metadata = dict(
  perc_train= 0.7,
  perc_valid= 0.1,
    
  range_train= None,
  range_validation= None ,
  range_test= None,
  shift= 0,
  starting_point= None,
  skip_step= 1,
  past_steps=16,
  future_steps= 16,
    precompute=True,
  scaler= 'sklearn.preprocessing.StandardScaler()')

# %%
ds = PandasTSDataSet(dataset,
                     'time', 
                     target, 
                     'group',
                     list(set(num_var_past+num_var_fut)),
                     list(set(cat_var_fut+cat_var_past)),
                     list(set(cat_var_fut+num_var_fut)),
                     list(set(cat_var_past+num_var_past)),
                )
        

# %%
from torch.utils.data import Dataset, DataLoader

# %%
from torch.utils.data.dataloader import default_collate
import torch
def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))
    print(len(batch))
    return default_collate(batch)

## @SANDEEP rework this using some cleaner code you can find here
#https://colab.research.google.com/drive/1FvLlmEOgm3D3JgNFVeAtwPk4cXagJ0CY?usp=sharing#scrollTo=jFuSwWrhTg6y
class MyDataset(Dataset):

    def __init__(self,data,metadata,valid_index)->torch.utils.data.Dataset:
        
        self.metadata = metadata
        self.metadata_dataset = data.metadata
        self.data = data
        self.valid_index = valid_index
        sum = 0
        self.lengths = {}
        #fix this, probably we need to add something (check validity, check past and future)
        for k in self.valid_index:
            sum+=(len(self.valid_index[k])-self.metadata['future_steps'])
            self.lengths[k] = len(self.valid_index[k])
        self.length = sum
    def __len__(self):
        return self.length

    def get_id_ts_by_idx(self,idx):
        
        tmp = np.cumsum(list(self.lengths.values()))

        idx =  min(np.where(tmp>idx)[0])
        return idx, 0 if idx==0 else tmp[idx-1]
    ## TODO: rewrite this using the Aryan code and names! and the D1 C, F and K U feature labeling!!
    def __getitem__(self, idxs): 

        ##crucial point there: correctly identifying which are the indexes that we need!
        sample = {}
        IDX,difference = self.get_id_ts_by_idx(idxs)

        idxs -= difference
        idxs = self.valid_index[IDX][idxs]
        tmp = self.data.__getitem__(IDX) ##use the getitem of the data!
        
        if idxs+self.metadata['future_steps']>self.data.lengths[IDX]:
            return None
        
        for k in tmp.keys():

            if len(tmp[k][idxs-self.metadata['past_steps']:idxs]) + len(tmp[k][idxs:idxs+self.metadata['future_steps']]) == self.metadata['future_steps']+self.metadata['past_steps']:
                if tmp['is_valid'][idxs-self.metadata['past_steps']:idxs+self.metadata['future_steps']].sum()==self.metadata['future_steps']+self.metadata['past_steps']:
                    if '_past' in k:
                        sample[k] = tmp[k][idxs-self.metadata['past_steps']:idxs]
                    elif '_fut' in k or k=='y':
                        sample[k] = tmp[k][idxs:idxs+self.metadata['future_steps']]
                    else:
                        pass
        if len(sample)==0:
            return None
        else:
            return sample

# %%
def compute_ranges( d1_dataset,metadata):
    ##suppose for now we use only percentage! 
    train_ranges = {}
    valid_ranges = {}
    test_ranges = {}
    for k in d1_dataset.lengths:
        ls = d1_dataset.lengths[k] 
        train_ranges[k] = list(range(0,int(metadata['perc_train']*ls)))
        valid_ranges[k] = list(range(int(metadata['perc_train']*ls), int((metadata['perc_valid'] + metadata['perc_train'])*ls)))
        test_ranges[k] = list(range(int((metadata['perc_valid'] + metadata['perc_train'])*ls),ls))
        
    ## in case of datetime ranges we need to use np.where and create a mask
    return train_ranges,valid_ranges,test_ranges
train_ranges,valid_ranges,test_ranges = compute_ranges( ds,data_module_metadata)


# %%
dl =  DataLoader(
            MyDataset(ds,data_module_metadata,train_ranges), ##change respect to the initial vignette, can not use standard dataset
            batch_size=32,
            collate_fn=my_collate
        )

# %%
import numpy as np

for x in dl:
    
    print(x['y'].shape)  #It works! 

# %%


# %%
def compute_ranges( d1_dataset,metadata):
    ##suppose for now we use only percentage! 
    train_ranges = {}
    valid_ranges = {}
    test_ranges = {}
    for k in d1_dataset.lengths:
        ls = d1_dataset.lengths[k] ##the time
        train_ranges[k] = list(range(0,int(metadata['perc_train']*ls)))
        valid_ranges[k] = list(range(int(metadata['perc_train']*ls), int((metadata['perc_valid'] + metadata['perc_train'])*ls)))
        test_ranges[k] = list(range(int(metadata['perc_valid'] + metadata['perc_train'])*ls),ls)
        
    ## in case of ranges we can 
    return train_ranges,valid_ranges,test_ranges

# %%

class DecoderEncoderDataModule(L.LightningDataModule):
    def __init__(self, d1_dataset, batch_size=32, num_workers=4,metadata=None):
        super().__init__()
        # initialize other  params
        self.d1_dataset = d1_dataset
        self.batch_size = batch_size
        self.metadata = metadata  


    def prepare_data(self):
    ##split is performed here
        if self.metadata['precompute']:
            self.precompute = True
            ## @SANDEEP this is what is returned by DSIPTS split_for_train!! 
            ## the only difference is that we need to use the __getitem__ of the d1 layer
            self.train_dataset, self.validation_dataset, self.test_dataset = split_data(self.d1_dataset,self.metadata)
            
            
          
        else:
            self.precompute = False
            ##in this case we need to pass only the data that are referring to the trainin period
            ## but data can be chunked, we need to create a function that transform a D1 object into a D1 object or, even better
            ## something that can be used by the dataset for filtering only valid samples!
            ##for example
            self.train_ranges, self.validation_ranges, self.test_ranges = compute_ranges(self.d1_dataset,self.metadata)
            self.predict_data = None ##??? need to study this
            ##TODO normalization???? maybe we can precompute here some statistics and pass them to the dataset!
    
    def setup(self, stage=None):
        # Get metadata from D1 layer during setup
        #self.metadata = self.d1_dataset.get_metadata() ##NO SEE COMMENT BEFORE!
        ##create dataset here
        if stage == 'fit': 
            if self.precompute:
                pass
            else:
                self.train_dataset = MyDataset(self.d1_dataset,self.train_ranges)
                self.validation_dataset = MyDataset(self.validation_data,self.validation_ranges)
                    
        
        if stage == 'test':
            if self.precompute:
                pass
            else:
                self.test_dataset = MyDataset(self.test_data,self.test_ranges)
            
        if stage == 'predict':
            if self.precompute:
                pass
            else:
                self.predict_datasset = MyDataset(self.predict_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=True,remove_last=True,collate_fn=my_collate)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size,shuffle=True,remove_last=True,collate_fn=my_collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=True,remove_last=True,collate_fn=my_collate)

    def predict_dataloader(self):
        return DataLoader(self.predict_datasset, batch_size=self.batch_size,shuffle=False,remove_last=False,collate_fn=my_collate)



# %%


# %%

# Layer M
class DecoderEncoderModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.metadata = None
        
    def setup(self, stage=None):
        # Get metadata from datamodule during setup
        self.metadata = self.trainer.datamodule.metadata
        
        # Initialize layer T model using metadata
    
    def forward(self, x):
     # forward logic
        pass

# %%



