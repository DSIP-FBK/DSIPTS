"""
Time Series Dataset module for handling data with two layers.

This module provides two main classes:
- TSDataSet (D1 Layer): Handles the raw data and provides a standardized interface
- TSDataProcessor (D2 Layer): Processes the output from D1 for model consumption

Key Features:
- Supports both datetime and numeric time indices
- Handles grouped time series data
- Provides weight handling for data points
- Ensures regular time intervals
- Validates data integrity
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple


def _coerce_to_list(obj):
    """
    Coerce object to list.

    Args:
        obj: Input object to be coerced to list

    Returns:
        List containing the input object(s)
    """
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    return list(obj)


def extend_time_df(df, time_col, freq, max_length=None):
    """
    Extend a dataframe to ensure regular time intervals.

    Args:
        df: Input dataframe containing time series data
        time_col: Column name containing time information
        freq: Frequency to use for extending the dataframe
        max_length: Optional maximum length for the extended dataframe

    Returns:
        DataFrame extended with regular time intervals
    """
    if len(df) == 0:
        return df
    
    # Get the minimum and maximum time values from the dataframe
    min_time = df[time_col].min()
    max_time = df[time_col].max()
    
    # Create a continuous time range based on the time type
    if isinstance(min_time, pd.Timestamp):
        # For datetime type, create a date range
        time_range = pd.date_range(start=min_time, end=max_time, freq=freq)
    else:
        # For numeric time indices, create a numeric range
        time_range = pd.Series(np.arange(min_time, max_time + freq, freq))
    
    # Limit the length if specified
    if max_length and len(time_range) > max_length:
        time_range = time_range[:max_length]
    
    # Create a new dataframe with the continuous time range
    extended_df = pd.DataFrame({time_col: time_range})
    
    return extended_df

'''
class TSDataSet(Dataset):
    """
    D1 Layer - A PyTorch Dataset for time series data.

    This class handles the raw data processing and provides a standardized interface
    for time series data. It supports:
    - Handling regular time intervals
    - Processing numerical features
    - Supporting group-based time series
    - Supporting weights for data points

    Returns from __getitem__:
    - t: Time values 
    - y: Target values
    - x: Feature values
    - group: Group identifier
    - weights: Weight values (if weight column is provided)
    """

    def __init__( #TODO: add cat variables too
        self,
        data: pd.DataFrame,
        time: Optional[str] = None,
        target: Optional[Union[str, List[str]]] = None,
        group: Optional[List[str]] = None, 
        num: Optional[List[Union[str, List[str]]]] = None, #REMOVE if minimal DISCUSS
        weight: Optional[str] = None, #REMOVE if minimal DISCUSS
    ):
        """
        Initialize the TSDataSet.

        Args:
            data: The input pandas DataFrame
            time: The name of the time index column if not specified, assumes the first column as time index
            target: The name(s) of the target variable column(s) if not specified, assumes the last column as target
            group: Columns used to group time series instances
            num: The name(s) of numerical feature columns
            weight: The name of the column containing weights for data points
        """
        # Initialize parameters
        self.data = data.copy()
        
        # Set time column (use the first column if not specified) # REMOVE THIS, as its manddatory field
        if time is None:
            self.time = data.columns[0]
        else:
            self.time = time
            
        # Set target columns (use the last column if not specified) # REMOVE THIS, as its manddatory field
        #self.target = _coerce_to_list(target if target is not None else data.columns[-1])
        
        # Set group columns
        self.group = _coerce_to_list(group)
        
        # Set weight column
        self.weight = weight
        
        # Set numerical feature columns
        if num is None:
            # Use all columns except time, target, group, and weight as numerical features
            exclude_cols = [self.time] + self.target + self.group
            if self.weight:
                exclude_cols.append(self.weight)
            self.num = [col for col in data.columns if col not in exclude_cols]
        else:
            self.num = _coerce_to_list(num)
            
        # Store feature columns (numerical features) TODO: include the target  as well. 
        self.feature_cols = self.num 
        
        # Use all unique columns for processing
        all_cols = list(set(self.feature_cols + self.target + [self.time] + self.group))
        if self.weight:
            all_cols.append(self.weight)
        self.data = data[all_cols].copy()
        
        # Handle duplicate time points by keeping the first occurrence TODO : if group is nan its good, else groups will be in subset and then duplicates. refer to pandas ds
        self.data.drop_duplicates(subset=[self.time], keep='first', inplace=True, ignore_index=True)
        
        # Determine time frequency
        if len(self.data) > 1:
            # Calculate the difference between consecutive time points
            time_diff = self.data[self.time].diff().dropna()
            
            if len(time_diff) > 0:
                # Find the minimum time difference (frequency)
                freq = time_diff.min()
                
                # Handle different time types
                if isinstance(freq, pd.Timedelta):
                    # If time is in timedelta format, convert to pandas timedelta
                    freq = pd.to_timedelta(freq)
                elif isinstance(freq, (int, float)):
                    # If time is numeric, convert to integer
                    freq = int(freq)
                else:
                    # Raise error for unsupported time types
                    raise TypeError("Time must be integer or datetime")
                
                # Extend dataset to ensure regular time intervals
                extended_df = extend_time_df(self.data, self.time, freq)
                
                # Merge extended time range with original data
                self.data = extended_df.merge(self.data, on=self.time, how='left').reset_index(drop=True)
                
        # Setup groups
        if self.group:
            # Create groups based on the specified columns
            self._groups = self.data.groupby(self.group).groups
            self._group_ids = list(self._groups.keys())
            self.lengths = {gid: len(indices) for gid, indices in self._groups.items()}
        else:
            # If no groups specified, treat entire dataset as single group
            self._groups = {"_single_group": self.data.index}
            self._group_ids = ["_single_group"]
            self.lengths = {"_single_group": len(self.data)}
            
        # If weight column is not provided, create a default weight column with all 1s ; TODO if not given we just ignore it. 
        if self.weight is None:
            self.data['_default_weight'] = 1.0
            self.weight = '_default_weight'
            
        # Prepare metadata
        self._prepare_metadata()

    def _prepare_metadata(self):
        """
        Prepare metadata for the dataset.

        The metadata includes information about columns, their types, and known status.
        """
        self.metadata = {
            "cols": {
                "y": self.target,
                "x": self.feature_cols,
                "st": [],  # No static features in this implementation
            },
            "col_type": {},
            "col_known": {},
            "weight": self.weight,
        }
        
        # Set column types and known status 
        all_cols = self.target + self.feature_cols
        for col in all_cols:
            # All features are numerical
            self.metadata["col_type"][col] = "F"
            
            # Features are known, targets are unknown (for future prediction)
            self.metadata["col_known"][col] = "K" if col in self.feature_cols else "U"

    def __len__(self) -> int:
        """
        Return the number of time series in the dataset.

        Returns:
            Number of time series (groups) in the dataset
        """
        return len(self._group_ids)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Get time series data for the given index.

        Args:
            index: The index of the time series to retrieve

        Returns:
            Dictionary containing:
            - t: Time values
            - y: Target values
            - x: Feature values
            - group: Group identifier
            - weights: Weight values
        """
        # Determine device for tensors
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Get group id and data for this index
        group_id = self._group_ids[index]
        
        if self.group:
            # Get data for this group
            mask = self._groups[group_id]
            data = self.data.loc[mask].sort_values(by=self.time)
        else:
            # Use all data if no grouping
            data = self.data.sort_values(by=self.time)
            
        # Create result dictionary
        result = {
            # Time values (as numpy array)
            "t": data[self.time].values,
            
            # Target values (as tensor)
            "y": torch.tensor(data[self.target].values, dtype=torch.float32, device=device),
            
            # Feature values (as tensor)
            "x": torch.tensor(data[self.feature_cols].values, dtype=torch.float32, device=device),
            
            # Group identifier (as hash if not single group)
            "group": torch.tensor([hash(str(group_id))], device=device),
            
            # Weight values
            "weights": torch.tensor(data[self.weight].values, dtype=torch.float32, device=device),
        }
        
        return result

    def get_metadata(self) -> Dict:
        """
        Return metadata about the dataset.

        Returns:
            Dictionary containing metadata about columns and their properties
        """
        return self.metadata
'''
## PICKED AS BASE

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
        #num: Optional[List[Union[str, List[str]]]] = None,
        #cat: Optional[List[Union[str, List[str]]]] = None,
        #known: Optional[List[Union[str, List[str]]]] = None,
        #unknown: Optional[List[Union[str, List[str]]]] = None,
        #static: Optional[List[Union[str, List[str]]]] = None,
        #label_encoders: Optional[dict] = None,
        weights: Optional[str] = None
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



        ## we know on the fly which rows are valid and wich contains nans TODO remove
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
            A dictionary containing tensors for 't', 'y', 'x', 'group', 'st'.
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
'''
class TSDataProcessor(Dataset):
    """
    D2 Layer - PyTorch Dataset for processing time series data for models.

    This class adapts the D1 dataset (TSDataSet) for use with models by creating
    input/output windows (sequences) for training and inference.

    It handles:
    - Creating sliding windows for inputs and targets
    - Validating data points
    - Supporting multiple time series (groups)
    - Handling weights for data points
    """

    def __init__(
        self, 
        data: TSDataSet, 
        metadata: Dict,
        valid_index: Optional[Dict] = None
    ):
        """
        Initialize the TSDataProcessor.

        Args:
            data: The D1 dataset instance (TSDataSet)
            metadata: Dictionary containing processing parameters like:
                - past_steps: Number of past time steps to use
                - future_steps: Number of future time steps to predict
            valid_index: Optional dictionary of valid indices for each group
        """
        # Store input parameters
        self.data = data  # D1 dataset
        self.metadata = metadata  # Processing parameters
        self.metadata_dataset = data.metadata  # Metadata from D1 dataset
        
        # Initialize valid indices if not provided (DISCUSS: OPTIONAL CAN BE REMOVED AS D1 handles this)
        if valid_index is None:
            self.valid_index = self._determine_valid_indices()
        else:
            self.valid_index = valid_index
            
        # Calculate total length and individual lengths for each group
        total_length = 0
        self.lengths = {}
        
        for k in self.valid_index:
            # Calculate valid windows for each group
            # Subtract future_steps to ensure enough data for target window
            group_length = max(0, len(self.valid_index[k]) - self.metadata.get('future_steps', 0))
            total_length += group_length
            self.lengths[k] = len(self.valid_index[k])
            
        self.length = total_length

    def _determine_valid_indices(self) -> Dict: # DISCUSS: REMOVE THIS PART???
        """
        Determine valid indices for each group.

        This method identifies indices in each group where a valid window
        can be created (enough data for past and future steps).

        Returns:
            Dictionary mapping group IDs to lists of valid indices
        """
        valid_index = {}
        past_steps = self.metadata.get('past_steps', 1)
        
        for i, group_id in enumerate(self.data._group_ids):
            # Get group data
            if i < len(self.data):
                group_data = self.data[i]
                
                # Find indices where there are enough data points before and after
                valid_indices = []
                for j in range(past_steps, len(group_data['t'])):
                    # Check if we have non-NaN values in the required window
                    y_window = group_data['y'][j-past_steps:j+1]
                    x_window = group_data['x'][j-past_steps:j+1]
                    
                    # Check if any NaN values in the window
                    if not (torch.isnan(y_window).any() or torch.isnan(x_window).any()):
                        valid_indices.append(j)
                        
                valid_index[i] = valid_indices
                
        return valid_index

    def get_id_ts_by_idx(self, idx: int) -> Tuple[int, int]:
        """
        Get time series ID and offset given global index.

        Args:
            idx: Global index across all time series

        Returns:
            Tuple containing (group_index, offset)
        """
        # Calculate cumulative lengths
        cumulative_lengths = np.cumsum(list(self.lengths.values()))
        
        # Find the group index
        group_idx = np.searchsorted(cumulative_lengths, idx, side='right')
        
        # Calculate the offset within that group
        offset = 0 if group_idx == 0 else cumulative_lengths[group_idx - 1]
        
        return group_idx, offset

    def __len__(self) -> int:
        """
        Return the total number of valid windows in the dataset.

        Returns:
            Number of valid windows across all time series
        """
        return self.length

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get a sample (input/output window) from the dataset.

        Args:
            idx: Global index of the window to retrieve

        Returns:
            Dictionary containing input and output tensors for the window,
            or None if a valid window cannot be created
        """
        # Find group and local index
        group_idx, offset = self.get_id_ts_by_idx(idx)
        
        # Adjust to local index
        local_idx = idx - offset
        
        # Get valid index within the group
        try:
            valid_indices = self.valid_index[group_idx]
            if local_idx >= len(valid_indices):
                return None
                
            data_idx = valid_indices[local_idx]
        except (KeyError, IndexError):
            return None
            
        # Get data from the D1 layer
        raw_data = self.data[group_idx]
        
        # Extract past and future steps parameters
        past_steps = self.metadata.get('past_steps', 1)
        future_steps = self.metadata.get('future_steps', 1)
        
        # Check if enough data is available for both input and output windows
        if data_idx + future_steps > len(raw_data['t']):
            return None
            
        # Check for validity of the time series by checking NaNs
        y_window = raw_data['y'][data_idx - past_steps:data_idx + future_steps]
        x_window = raw_data['x'][data_idx - past_steps:data_idx + future_steps]
        
        if torch.isnan(y_window).any() or torch.isnan(x_window).any():
            return None
            
        # Construct sample
        sample = {}
        
        # Add past data (input window)
        sample['x_past'] = raw_data['x'][data_idx - past_steps:data_idx]
        sample['t_past'] = raw_data['t'][data_idx - past_steps:data_idx]
        
        # Add future data (target window)
        sample['y'] = raw_data['y'][data_idx:data_idx + future_steps]
        sample['t_fut'] = raw_data['t'][data_idx:data_idx + future_steps]
        
        # Add group information
        sample['group'] = raw_data['group']
        
        # Add weights
        sample['weights_past'] = raw_data['weights'][data_idx - past_steps:data_idx]
        sample['weights_fut'] = raw_data['weights'][data_idx:data_idx + future_steps]
        
        return sample
'''

#D2 Layer implementation
