# %%
from dsipts import Categorical,TimeSeries, RNN
settimana = Categorical('settimanale',1,[1,1,1,1,1,1,1],7,'multiplicative',[0.9,0.8,0.7,0.6,0.5,0.99,0.99])

##montly, additive (here there are only 5 month)
mese = Categorical('mensile',1,[31,28,20,10,33],5,'additive',[10,20,-10,20,0])

##spot categorical variables: in this case it occurs every 100 days and it lasts 7 days adding 10 to the original timeseries
spot = Categorical('spot',100,[7],1,'additive',[10])

##initizate a timeseries object
ts = TimeSeries('prova')
ts.generate_signal(noise_mean=1,categorical_variables=[settimana,mese,spot],length=5000,type=0)
dataset = ts.dataset
dataset['x_num'] = 0.0
dataset.loc[:dataset.shape[0]-10,'x_num']= dataset['x_num'].values[9:]
cat_var_past = ts.cat_var
cat_var_fut = ts.cat_fut
num_var_past = ts.num_var+['x_num']
import pickle 
with open('example.pkl', 'wb') as f:
    pickle.dump([dataset,cat_var_past,cat_var_fut,num_var_past],f)

# %%
#dataset.to_csv('dataset.csv',index=False)

# %%
!pip install lightning

# %%
import pickle
import pandas as pd
from typing import Union, List, Optional, Dict
from sklearn.preprocessing import LabelEncoder  # Used for categorical encoding
from datetime import datetime, timedelta  # For time-related operations
import lightning as L  # PyTorch Lightning
import numpy as np
from sklearn.preprocessing import OrdinalEncoder  # Used for categorical encoding
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


# %% [markdown]
# **Data Loading and Preparation (Example Data)**
# 
# This section loads example time series data and prepares it for use in the `DataSet` classes. It demonstrates the structure of the input data.
# 

# %%
with open('example.pkl', 'rb') as f:
    dataset, cat_var_past, cat_var_fut, num_var_past = pickle.load(f)
num_var_fut = []  # No numerical future variables in this example
group = None       # No grouping in this first example.  Later examples *do* use grouping.
time_var = 'time'
target = ['signal']

# %% [markdown]
# **Example with Grouping**
# 
# The following cell creates a second copy of the dataset and assigns it a different group ID. This is then concatenated with the original `dataset` to showcase how grouping works.
# 

# %%
dataset2 = dataset.copy()

#TODO break the files the into multip
# %% [markdown]
# **`extend_time_df` Function**
# This utility function extends a DataFrame to include all time steps within a specified range and frequency.  This is essential for ensuring that all time series have the same length and that there are no missing time steps.
# 
# 

# %%
def extend_time_df(x:pd.DataFrame,time:str,freq:Union[str,int],group:Union[List[str],None]=None,global_minmax:bool=False)-> pd.DataFrame:
    """Extends a DataFrame to include all time steps within a range.

    Ensures that the DataFrame `x` has rows for all time steps within the
    range defined by the minimum and maximum values of the `time` column,
    with a specified frequency `freq`.  Handles grouping if `group` is provided.

    Args:
        x: The input DataFrame. Must contain a 'time' column and any
           columns specified in 'group'.
        time: The name of the column containing the time index (e.g., 'time').
        freq: The frequency to use for extending the DataFrame.  This can be
            an integer (for integer time steps) or a string representing a
            pandas frequency string (e.g., 'D' for daily, 'H' for hourly).
        group:  Optional list of column names to group by. If provided, the
            DataFrame will be extended separately for each group.
        global_minmax: If True, use the global min/max of the 'time' column
            across all groups. If False, use the min/max within each group.

    Returns:
        pd.DataFrame: An extended DataFrame with all time steps within the
            specified range and frequency.  Missing values introduced by the
            extension are filled with NaNs.
    """
    if group is None:
        # No grouping: compute min and max time directly
        if isinstance(freq, int):
            empty = pd.DataFrame({'time': list(range(x[time].min(), x[time].max(), freq))})
        else:
            empty = pd.DataFrame({'time': pd.date_range(x[time].min(), x[time].max(), freq=freq)})

    else:
        # Grouping: compute min and max time for each group
        if global_minmax:
            # Use global min/max (across all groups)
            _min = x[time].min()
            _max = x[time].max()
            _min_max = pd.DataFrame({'min': [_min], 'max': [_max]}) # Use a list for correct DataFrame creation
            for c in group:
                #  Create dummy columns, will be overwritten
                _min_max[c] = _min_max['min']
        else:
            # Use per-group min/max
            _min = x.groupby(group)[time].min().reset_index().rename(columns={time: 'min'})
            _max = x.groupby(group)[time].max().reset_index().rename(columns={time: 'max'})
            _min_max = pd.merge(_min, _max)

        empty = []
        for _, row in _min_max.iterrows():
            if isinstance(freq, int):
                tmp = pd.DataFrame({time: np.arange(row['min'], row['max'], freq)})
            else:
                tmp = pd.DataFrame({time: pd.date_range(row['min'], row['max'], freq=freq)})
            for c in group:
                tmp[c] = row[c]  # Assign the group values
            empty.append(tmp)

        empty = pd.concat(empty, ignore_index=True)  # Concatenate all group DataFrames
    return empty

# %% [markdown]
# **`_coerce_to_list` Function (Utility)**
# 

# %%
def _coerce_to_list(obj):
    """Coerces input to a list.

    Ensures that the input is always a list.  If the input is `None`,
    it returns an empty list.  If the input is a single value (not already
    a list), it returns a list containing that single value.

    Args:
        obj: The input object.

    Returns:
        list: A list representing the input object.
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    else:
        return [obj]

# %% [markdown]
# **`PandasTSDataSet_MINIMAL` Class (Simplified Example)**
# This class is a simplified version of the `PandasTSDataSet` to illustrate the basic principles without the complexity of handling all the different data types and metadata.  It serves as a stepping stone to the full implementation.
# 
# 

# %%

##TODO @Sandeep please try to replicate what is in PandasTSDataSet but for a smore compact example!

# %%
class PandasTSDataSet_MINIMAL:
    """A simplified version of PandasTSDataSet for demonstration.

    This class demonstrates the core functionality of creating a PyTorch
    Dataset from a pandas DataFrame.  It handles:

    - Extending the DataFrame to ensure regular time intervals.
    - Returning tensors for time, target, and numerical features.
    - Identifying valid data points (non-NaN).

    It does *not* handle:
    - Categorical features.
    - Static features.
    - Future known covariates.
    - Grouping.
    - Metadata handling (beyond a very basic level).

    This simplified version is intended for educational purposes, to
    illustrate the key concepts before diving into the full complexity
    of the `PandasTSDataSet` class.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time: Optional[str] = None,
        target: Optional[Union[str, List[str]]] = None,
        num: Optional[List[Union[str, List[str]]]] = None,

    ):
        """Initializes the PandasTSDataSet_MINIMAL.

        Args:
            data: The input pandas DataFrame.
            time: The name of the time index column.
            target: The name(s) of the target variable column(s).
            num: The name(s) of numerical feature columns.
        """

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
                "st": [],  # No static features in this minimal version
            },
            "col_type": {},
            "col_known": {},
            "cat_index":[] # No categorical features in this minimal version
        }
        # Removed the loop that checks for categorical features, as they are not handled here.

        all_cols = self.target + self.feature_cols
        for col in all_cols:
            self.metadata["col_type"][col] = "F"  # All features are numerical ("F")

            self.metadata["col_known"][col] = "K" if col in self.feature_cols else "U"
        self.metadata['encoders'] = self.label_encoders # No encoders, as they're used for categoricals


    def __len__(self) -> int:
        """Return number of time series in the dataset."""
        return len(self._group_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get time series data for given index.

        Args:
            index: The index of the time series to retrieve (not used directly,
                as we have a single time series in this simplified version).

        Returns:
            A dictionary containing tensors for 't', 'y', 'x', and 'is_valid'.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        data = self.data.sort_values(by=self.time)

        result = {
            "t": data[self.time].values,  # Return as numpy array (no need for tensor here)
            "y": torch.tensor(data[self.target].values,dtype=torch.float32,device=device),
            "x": torch.tensor(data[self.feature_cols].values,dtype=torch.float32,device=device),
            "weights": 
        }
        return result

    def get_metadata(self) -> Dict:
        """Return metadata about the dataset.

        Returns:
            A dictionary containing metadata about the columns.
        """
        return self.metadata



# %% [markdown]
# **`PandasTSDataSet` Class (D1 - pandas Implementation)**

# %%
class PandasTSDataSet(Dataset):
    """PyTorch Dataset for loading time series data from a pandas DataFrame.

    This class implements the D1 (raw data) layer of the `pytorch-forecasting`
    v2 API.  It handles loading time series data from a pandas DataFrame,
    performing basic preprocessing (like ensuring regular time intervals),
    and providing access to the data via the `__getitem__` method.

    Key Features:
    - Handles multiple time series (identified by group IDs).
    - Supports numerical and categorical features.
    - Supports static features (features that don't change over time).
    - Supports known and unknown covariates.
    - Automatically extends the DataFrame to ensure regular time intervals.
    - Returns data in a standardized dictionary format (tensors).
    - Provides metadata about the columns (names, types, known/unknown status).
    """

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
        """Initializes the PandasTSDataSet.

        Args:
            data: The input pandas DataFrame.
            time: The name of the time index column.
            target: The name(s) of the target variable column(s).
            group:  List of column names identifying a time series instance.
            num: The name(s) of numerical feature columns.
            cat: The name(s) of categorical feature columns.
            known:  List of variables known in the future.
            unknown: List of variables unknown in the future
            static: The name(s) of static feature columns (features that don't
                change over time).
            label_encoders: Optional dictionary of pre-fitted label encoders
                for categorical features. If not provided, new encoders will
                be fitted.
        """

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
        for i, c in enumerate(self.feature_cols):
            if c in self.cat:
                self.metadata['cat_index'].append(i) # EDIT: was getting error with i, so updated enumerate in for loop
                

        all_cols = self.target + self.feature_cols + self.static
        for col in all_cols:
            self.metadata["col_type"][col] = "C" if col in self.cat else "F"
            self.metadata["col_known"][col] = "K" if col in self.known else "U"
        self.metadata['encoders'] = self.label_encoders

    def __len__(self) -> int:
        """Return number of time series in the dataset."""
        return len(self._group_ids)

    def get_id_ts_by_idx(self,idx):
        """Gets the time series ID and starting index for a given index.

        Args:
            idx: The overall index.  This is *not* the group index, but
                 the index considering all time series concatenated together.

        Returns:
            tuple: A tuple containing the group ID and the starting index
                of that group in the concatenated data.
        """
        tmp = np.cumsum(list(self.lengths.values()))
        idx_group =  min(np.where(tmp>idx)[0])  # EDIT: Find the group index, as we are getting group
        return idx_group, 0 if idx_group==0 else tmp[idx_group-1] #EDIT: group id, index of the start of the series

    def get_total_len(self):
        """Returns total length of the dataset"""
        l = 0
        for k in  self._groups:
            l+= self.lengths[k]
        return l

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get time series data for given index.

        Args:
            index: The index of the time series to retrieve.

        Returns:
            A dictionary containing tensors for 't', 'y', 'x', 'group', 'st',
            and 'is_valid'.
        """
        group_id = self._group_ids[index]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.group:
            mask = self._groups[group_id]
            data = self.data.loc[mask].sort_values(by=self.time)
        else:
            data = self.data.sort_values(by=self.time)

        result = {
            "t": data[self.time].values,  # Return as numpy array
            "y": torch.tensor(data[self.target].values, dtype=torch.float32, device=device),
            "x": torch.tensor(data[self.feature_cols].values, dtype=torch.float32, device=device),
            "group": torch.tensor([group_id], dtype=torch.int64, device=device),  # group ID, cast to int64
            "st": torch.tensor(data[self.static].iloc[0].values, dtype=torch.float32, device=device) if self.static else torch.tensor([], dtype=torch.float32, device=device), # return empty if no statics
            'is_valid': torch.tensor(data.valid.values,dtype=torch.int,device=device), #indicator of if data is valid
        }

        return result

    def get_metadata(self) -> Dict:
        """Return metadata about the dataset.

        Returns:
        -------
        Dict
            Dictionary containing:
            - cols: column names for y, x, and static features
            - col_type: mapping of columns to their types (F/C)
            - col_known: mapping of columns to their future known status (K/U)
            - encoders: Dict of label encoders used
        """
        return self.metadata

# %% [markdown]
# **Configuration dictionary for the data module.  This holds hyperparameters related to data splitting, preprocessing, and windowing.**

# %%
data_module_metadata = dict(
  perc_train= 0.7,       # Percentage of data for training
  perc_valid= 0.1,       # Percentage of data for validation
  range_train= None,     # Optional: Specific time range for training (not used here)
  range_validation= None,  # Optional: Specific time range for validation (not used here)
  range_test= None,      # Optional: Specific time range for testing (not used here)
  shift= 0,              # Optional: Shift the time series (not used here)
  starting_point= None,  # Optional: Starting point for sampling (not used here)
  skip_step= 1,          # Optional: Step size for skipping data points (not used here)
  past_steps=16,         # Number of past time steps to use as input (encoder length)
  future_steps= 16,      # Number of future time steps to predict (decoder length)
  precompute=True,      # Whether to precompute the train/val/test splits.
  scaler= 'sklearn.preprocessing.StandardScaler()'  # String representation of the scaler (not used directly here)
)


# %% [markdown]
# **Instantiate the PandasTSDataSet (D1 layer).**
# This creates the dataset object that will provide the raw time series data.
# 

# %%
ds = PandasTSDataSet(dataset,
                     'time',  # Name of the time index column
                     target,  # List of target variable names
                     'group', # Name of the column that identifies groups/time series
                     list(set(num_var_past+num_var_fut)),  # List of numerical feature columns
                     list(set(cat_var_fut+cat_var_past)),  # List of categorical feature columns
                     list(set(cat_var_fut+num_var_fut)),  # List of future known covariates
                     list(set(cat_var_past+num_var_past)), # List of past known covariates
                )



# %% [markdown]
# Import necessary PyTorch modules

# %%
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch

# %%
def my_collate(batch):
    """Custom collate function to handle potential None values in batches.

    Filters out `None` values from the batch before collating.  This is
    necessary because the `MyDataset.__getitem__` method can return `None`
    if a valid window cannot be created for a given index.

    Args:
        batch: A list of samples (dictionaries or tensors) from the dataset.

    Returns:
        The collated batch, with `None` values removed.
    """
    batch = list(filter(lambda x : x is not None, batch))
    #print(len(batch))  # Debugging print statement (removed in final version)
    return default_collate(batch)


# %%
## TODO: @SANDEEP rework this using some cleaner code you can find here
#https://colab.research.google.com/drive/1FvLlmEOgm3D3JgNFVeAtwPk4cXagJ0CY?usp=sharing#scrollTo=jFuSwWrhTg6y

# %%
class MyDataset(Dataset):
    """Custom PyTorch Dataset for time series data.

    This class adapts the D1 dataset (`PandasTSDataSet`) for use with a
    PyTorch `DataLoader`.  It handles the creation of input/output windows
    (sequences) for training and validation.  It also handles the case where
    a valid window cannot be created (e.g., due to insufficient past data)
    by returning `None`.

    Args:
        data: The D1 dataset instance (e.g., `PandasTSDataSet`).
        metadata: A dictionary containing metadata about the dataset and
            data processing parameters (e.g., `past_steps`, `future_steps`).
        valid_index:  A dictionary where keys are group IDs and values are
            lists of valid indices for that group.  This defines the
            valid ranges within each time series for creating windows.
    """

    def __init__(self, data, metadata, valid_index) -> torch.utils.data.Dataset:
        """Initializes the MyDataset instance."""

        self.metadata = metadata
        self.metadata_dataset = data.metadata  # Metadata from the D1 dataset
        self.data = data  # The D1 dataset instance
        self.valid_index = valid_index  # Dictionary of valid indices for each group
        sum = 0
        self.lengths = {}
        #fix this, probably we need to add something (check validity, check past and future)
        for k in self.valid_index:
            # Calculate the total number of valid windows that can be created
            # from each group.  Subtract `future_steps` because we need enough
            # data points for both the input window and the target window.
            sum+=(len(self.valid_index[k])-self.metadata['future_steps'])
            self.lengths[k] = len(self.valid_index[k])
        self.length = sum # Sum of lengths accross differnet time series.

    def __len__(self):
        """Returns the total number of valid windows in the dataset."""
        return self.length

    def get_id_ts_by_idx(self,idx):
        """Gets time series id given global index"""
        tmp = np.cumsum(list(self.lengths.values()))
        idx =  min(np.where(tmp>idx)[0])
        return idx, 0 if idx==0 else tmp[idx-1]

    def __getitem__(self, idxs):
        """Retrieves a sample (input/output window) from the dataset.

        Args:
            idxs: The *global* index of the window to retrieve.  This index
                is across all time series in the dataset.

        Returns:
            A dictionary containing the input and output tensors for the window,
            or `None` if a valid window cannot be created at the given index.
        """

        ##crucial point there: correctly identifying which are the indexes that we need!
        sample = {}
        IDX,difference = self.get_id_ts_by_idx(idxs) #find group and local index

        idxs -= difference  # Adjust index to be relative to the start of the time series
        idxs = self.valid_index[IDX][idxs]  # Get the actual index in the data
        tmp = self.data.__getitem__(IDX) ##use the getitem of the data!

        # Check if enough data is available for both input and output windows
        if idxs+self.metadata['future_steps']>self.data.lengths[IDX]:
            return None  # Not enough data, return None

        #Check for the validity of the time series, by checking nans.
        if tmp['is_valid'][idxs-self.metadata['past_steps']:idxs+self.metadata['future_steps']].sum() != self.metadata['future_steps']+self.metadata['past_steps']:
          return None
      
        # Construct sample
        for k in tmp.keys():
            # Separate based on past/future
            # We create past and future samples only for the ones we sliced using idxs
            if '_past' in k:
                sample[k] = tmp[k][idxs-self.metadata['past_steps']:idxs] #past steps based on the past_step provided in metadata.
            elif '_fut' in k or k=='y':
                sample[k] = tmp[k][idxs:idxs+self.metadata['future_steps']]# future steps based on the future_step
            else:
                pass
        return sample


# %%
def compute_ranges(d1_dataset, metadata):
    """Computes valid index ranges for training, validation, and testing.

    This function calculates the start and end indices for the training,
    validation, and testing sets within each time series group, based on the
    provided percentages in `metadata`.

    Args:
        d1_dataset: The D1 dataset instance (e.g., `PandasTSDataSet`).
        metadata: A dictionary containing metadata, including `perc_train`
            and `perc_valid` (percentages for training and validation splits).

    Returns:
        tuple: A tuple containing three dictionaries: `train_ranges`,
            `valid_ranges`, and `test_ranges`.  Each dictionary has group IDs
            as keys and lists of valid indices as values.
    """
    ##suppose for now we use only percentage!
    train_ranges = {}
    valid_ranges = {}
    test_ranges = {}
    for k in d1_dataset.lengths:  # Iterate over each time series group
        ls = d1_dataset.lengths[k]  # Get the length of the current time series
        train_ranges[k] = list(range(0,int(metadata['perc_train']*ls)))  # Training range
        valid_ranges[k] = list(range(int(metadata['perc_train']*ls), int((metadata['perc_valid'] + metadata['perc_train'])*ls)))  # Validation range
        test_ranges[k] = list(range(int((metadata['perc_valid'] + metadata['perc_train'])*ls),ls))  # Testing range

    ## in case of datetime ranges we need to use np.where and create a mask
    return train_ranges,valid_ranges,test_ranges

# Calculate the training, validation, and testing ranges
train_ranges, valid_ranges, test_ranges = compute_ranges(ds, data_module_metadata)



# %% [markdown]
# **Create a DataLoader for the training set.**

# %%
dl =  DataLoader(
            MyDataset(ds, data_module_metadata, train_ranges),  # Create a MyDataset instance
            batch_size=32,  # Set the batch size
            collate_fn=my_collate  # Use the custom collate function
        )



# %%
# Example of iterating through the DataLoader (for debugging/demonstration).
for x in dl:
    print(x['y'].shape)  # Print the shape of the target variable tensor.
    break #only check first batch


# %% [markdown]
# **The following function and class show how one might create a `LightningDataModule` to manage the data loading process, including train/validation/test splits. This builds on top of previous components.**

# %%
class DecoderEncoderDataModule(L.LightningDataModule):
    """LightningDataModule for encoder-decoder models.

    This class manages the data loading and preprocessing for training,
    validation, testing, and prediction.  It handles splitting the data,
    creating `Dataset` instances, and creating `DataLoader` instances.  It
    supports both precomputed splits and on-the-fly splitting.

    Args:
        d1_dataset: The D1 dataset instance (e.g., `PandasTSDataSet`).
        batch_size: The batch size.
        num_workers: The number of worker processes for data loading.
        metadata: A dictionary containing metadata and hyperparameters.
    """
    def __init__(self, d1_dataset, batch_size=32, num_workers=4, metadata=None):
        super().__init__()
        # initialize other  params
        self.d1_dataset = d1_dataset
        self.batch_size = batch_size
        self.metadata = metadata
        self.num_workers = num_workers


    def prepare_data(self):
        """Prepares the data for training/validation/testing.

        This method handles splitting the data into training, validation, and
        testing sets. It supports two modes:

        - `precompute=True`:  Uses a `split_data` function (not defined here,
          but assumed to exist) to precompute the splits. This is useful for
          deterministic splits or when the splitting process is expensive.
        - `precompute=False`:  Calculates the splits on-the-fly using
          `compute_ranges`. This is more flexible and allows for different
          splitting strategies.
        """
        if self.metadata['precompute']:
            self.precompute = True
            ## @SANDEEP this is what is returned by DSIPTS split_for_train!!
            ## the only difference is that we need to use the __getitem__ of the d1 layer

            # The split_data function is assumed to be defined elsewhere and
            # would handle the splitting logic according to however the
            # project defines "splitting" (could be by time, by groups, etc.)
            self.train_dataset, self.validation_dataset, self.test_dataset = split_data(self.d1_dataset,self.metadata) # Placeholder



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
        """Sets up the datasets for each stage (fit, test, predict).

        This method creates the appropriate `Dataset` instances for each stage,
        using either the precomputed splits or the on-the-fly calculated ranges.

        Args:
            stage: The stage ('fit', 'test', or 'predict').
        """
        # Get metadata from D1 layer during setup
        #self.metadata = self.d1_dataset.get_metadata() ##NO SEE COMMENT BEFORE!  We get metadata in __init__
        ##create dataset here
        if stage == 'fit':
            if self.precompute:
                # Use precomputed datasets (implementation of split_data needed)
                pass # Use self.train_dataset and self.validation_dataset as defined in prepare_data
            else:
                # Create datasets using computed ranges
                self.train_dataset = MyDataset(self.d1_dataset, self.metadata, self.train_ranges) # Pass train_ranges
                self.validation_dataset = MyDataset(self.d1_dataset, self.metadata, self.validation_ranges) # Pass valid_ranges

        if stage == 'test':
            if self.precompute:
              pass
                # Use precomputed test dataset
            else:
                self.test_dataset = MyDataset(self.d1_dataset, self.metadata, self.test_ranges) # Pass test_ranges

        if stage == 'predict':
            if self.precompute:
                pass
                # Use precomputed prediction dataset (if applicable)
            else:
                self.predict_datasset = MyDataset(self.d1_dataset, self.metadata, self.predict_data) # Pass predict_data.  Implementation details TBD.

    def train_dataloader(self):
        """Returns the training DataLoader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=my_collate)

    def val_dataloader(self):
        """Returns the validation DataLoader."""
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=my_collate)

    def test_dataloader(self):
        """Returns the test DataLoader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=my_collate)

    def predict_dataloader(self):
        """Returns the prediction DataLoader."""
        return DataLoader(self.predict_datasset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=my_collate)



# %% [markdown]
# **`DecoderEncoderModel` Class (M Layer - Example)**
# This is a placeholder for the Model Layer.

# %%

# Layer M - Example (Placeholder)
class DecoderEncoderModel(L.LightningModule):
    """Placeholder for an encoder-decoder model (M layer).

    This is a minimal example of how a model might be structured using
    PyTorch Lightning.  It shows how to access the metadata from the
    `LightningDataModule`.  A real implementation would include the actual
    network architecture and training logic.
    """
    def __init__(self):
        super().__init__()
        self.metadata = None

    def setup(self, stage=None):
        """Gets metadata from the datamodule during setup."""
        # Get metadata from datamodule during setup
        self.metadata = self.trainer.datamodule.metadata

        # Initialize layer T model using metadata (placeholder)

    def forward(self, x):
        """Forward pass (placeholder)."""
        # forward logic (implementation would go here)
        pass

    def training_step(self, batch, batch_idx):
        """Training step (placeholder)."""
        pass  # Replace with actual training logic

    def validation_step(self, batch, batch_idx):
        """Validation step (placeholder)."""
        pass # Replace with actual validation logic


# %% [markdown]
# ### Layer M

# %%


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




# %%
import numpy as np
import pandas as pd
import zarr
import torch
from torch.utils.data import Dataset
from typing import Optional, Union, List, Dict
import dask.array as da
import dask.dataframe as dd

class ZarrTSDataset(Dataset):
    """PyTorch Dataset for loading time series data from Zarr arrays.
    
    This class implements a dataset layer optimized for larger-than-memory time series data
    using Zarr storage format. It supports chunked data access and lazy loading.
    
    Parameters
    ----------
    data : Union[str, zarr.Group]
        Either path to zarr store or zarr group object
    time : str
        Name of time column
    target : Optional[Union[str, List[str]]]
        Target variable(s) to predict
    group : Optional[List[str]]
        Columns to group by
    num : Optional[List[Union[str, List[str]]]]
        Numerical features
    cat : Optional[List[Union[str, List[str]]]]
        Categorical features
    known : Optional[List[Union[str, List[str]]]]
        Known features (available for future timesteps)
    unknown : Optional[List[Union[str, List[str]]]]
        Unknown features (not available for future timesteps)
    static : Optional[List[Union[str, List[str]]]]
        Static features (constant over time)
    label_encoders : Optional[dict]
        Dictionary of label encoders for categorical variables
    chunk_size : Optional[Dict[str, int]]
        Size of chunks for each dimension, e.g {'time':100, 'group':10}
    """
    
    def __init__(
        self,
        data: Union[str, zarr.Group],
        time: str,
        target: Optional[Union[str, List[str]]] = None,
        group: Optional[List[str]] = None,
        num: Optional[List[Union[str, List[str]]]] = None,
        cat: Optional[List[Union[str, List[str]]]] = None,
        known: Optional[List[Union[str, List[str]]]] = None,
        unknown: Optional[List[Union[str, List[str]]]] = None,
        static: Optional[List[Union[str, List[str]]]] = None,
        label_encoders: Optional[dict] = None,
        chunk_size: Optional[Dict[str, int]] = None,
    ):
        # Initialize zarr store
        self.zarr_store = data if isinstance(data, zarr.Group) else zarr.open(data, mode='r')
        
        # Store parameters
        self.time = time
        self.target = target if isinstance(target, list) else [target] if target else []
        self.group = group if group else []
        self.num = num if num else []
        self.cat = cat if cat else []
        self.known = known if known else []
        self.unknown = unknown if unknown else []
        self.static = static if static else []
        self.label_encoders = label_encoders if label_encoders else {}
        self.chunk_size = chunk_size if chunk_size else {}
        
        # Create dask array from zarr store
        self.dask_array = da.from_zarr(self.zarr_store)
        
        # Convert to dask dataframe
        self.data = dd.from_dask_array(self.dask_array, columns=list(self.zarr_store.array_keys()))
        
        # Compute total length
        self._length = len(self.data)
        
        # Setup chunking
        self._setup_chunks()
        
    def _setup_chunks(self):
        """Setup chunk information for iteration."""
        if not self.chunk_size:
            self.n_chunks = 1
            return
            
        self.chunks_per_dim = {}
        for dim in self.chunk_size:
            total_size = self.data[dim].shape[0]
            chunk_size = self.chunk_size[dim]
            self.chunks_per_dim[dim] = int(np.ceil(total_size / chunk_size))
            
        # Total number of chunks
        self.n_chunks = np.prod(list(self.chunks_per_dim.values()))
        
    def _get_chunk(self, chunk_idx: int) -> pd.DataFrame:
        """Get a chunk of data based on chunk index."""
        if not self.chunk_size:
            return self.data.compute()
            
        # Calculate chunk coordinates
        chunk_coords = np.unravel_index(chunk_idx, tuple(self.chunks_per_dim.values()))
        
        # Create slice for each dimension
        chunk_slices = {}
        for dim_idx, dim in enumerate(self.chunk_size.keys()):
            start = chunk_coords[dim_idx] * self.chunk_size[dim]
            end = min(start + self.chunk_size[dim], self.data[dim].shape[0])
            chunk_slices[dim] = slice(start, end)
            
        # Get chunk data
        chunk_data = self.data.loc[chunk_slices].compute()
        return chunk_data
        
    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        return self._length
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the item to get
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the following tensors:
            - 't': time values
            - 'y': target values
            - 'x': feature values
            - 'g': group values
            - 'st': static feature values
            - 'is_valid': mask indicating valid values
        """
        # Calculate which chunk this index belongs to
        items_per_chunk = np.prod([self.chunk_size[dim] for dim in self.chunk_size])
        chunk_idx = idx // items_per_chunk if self.chunk_size else 0
        local_idx = idx % items_per_chunk if self.chunk_size else idx
        
        # Get chunk data
        chunk_data = self._get_chunk(chunk_idx)
        
        # Get single item from chunk
        data = chunk_data.iloc[local_idx:local_idx+1]
        
        # Convert to tensors
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        result = {
            "t": torch.tensor(data[self.time].values, dtype=torch.float32, device=device),
            "y": torch.tensor(data[self.target].values, dtype=torch.float32, device=device),
            "x": torch.tensor(data[self.num].values, dtype=torch.float32, device=device) if self.num else torch.tensor([], dtype=torch.float32, device=device),
            "g": torch.tensor(data[self.group].values, dtype=torch.int64, device=device) if self.group else torch.tensor([], dtype=torch.int64, device=device),
            "st": torch.tensor(data[self.static].values, dtype=torch.float32, device=device) if self.static else torch.tensor([], dtype=torch.float32, device=device),
            "is_valid": torch.tensor(data.notna().all(axis=1).values, dtype=torch.bool, device=device),
        }
        
        return result
        
    @classmethod
    def from_pandas(cls, df: pd.DataFrame, zarr_path: str, **kwargs) -> "ZarrTSDataset":
        """Create ZarrTSDataset from pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        zarr_path : str
            Path to save zarr store
        **kwargs : dict
            Additional arguments to pass to ZarrTSDataset constructor
            
        Returns
        -------
        ZarrTSDataset
            New dataset instance
        """
        # Create zarr store
        store = zarr.open(zarr_path, mode='w')
        
        # Save each column as a zarr array
        for col in df.columns:
            store.create_dataset(col, data=df[col].values, chunks=True)
            
        return cls(store, **kwargs)
