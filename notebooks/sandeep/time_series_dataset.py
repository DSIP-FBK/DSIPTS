"""
Time Series Dataset Module

This module provides classes for two-layer time series data handling:
- MultiSourceTSDataSet (D1 Layer): Handles raw data from multiple CSV files
- TSDataModule (D2 Layer): LightningDataModule for time series data with support for training, validation, and testing.
    - TSDataProcessor (helper of D2 Layer): Processes data for model consumption with sliding windows

Key Features:
- Supports multiple CSV files with different groups
- Handles regular time intervals
- Creates sliding windows for inputs and targets
- Supports validation of data points
- Global indexing across groups and files
- Precomputed indices for valid samples
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import timedelta
from sklearn.preprocessing import OrdinalEncoder


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


def extend_time_df(df, time_col, freq, group_cols=None, max_length=None):
    """
    Extend a dataframe to ensure regular time intervals.
    
    Args:
        df: Input dataframe containing time series data
        time_col: Column name containing time information
        freq: Frequency to use for extending the dataframe
        group_cols: Optional list of columns identifying groups
        max_length: Optional maximum length for the extended dataframe
        
    Returns:
        DataFrame extended with regular time intervals
    """
    if len(df) == 0:
        return df
    
    # Ensure frequency is positive
    if (isinstance(freq, (timedelta, int, float)) and 
        (freq.total_seconds() < 0 if isinstance(freq, timedelta) else freq < 0)):
        freq = abs(freq)
        
    result_df = pd.DataFrame()
    
    # If grouped data, process each group separately
    if group_cols and len(group_cols) > 0:
        for group_name, group_data in df.groupby(group_cols):
            # Get min and max time for this group
            min_time = group_data[time_col].min()
            max_time = group_data[time_col].max()
            
            # Create time range for this group
            if isinstance(min_time, pd.Timestamp):
                time_range = pd.date_range(start=min_time, end=max_time, freq=freq)
            else:
                time_range = pd.Series(np.arange(min_time, max_time + freq, freq))
                
            # Limit the length if specified
            if max_length and len(time_range) > max_length:
                time_range = time_range[:max_length]
                
            # Create extended dataframe for this group
            if isinstance(group_name, tuple):
                # Multiple group columns
                extended_group = pd.DataFrame({time_col: time_range})
                for i, col in enumerate(group_cols):
                    extended_group[col] = group_name[i]
            else:
                # Single group column
                extended_group = pd.DataFrame({
                    time_col: time_range,
                    group_cols[0]: group_name
                })
                
            result_df = pd.concat([result_df, extended_group], ignore_index=True)
    else:
        # No groups, extend the entire dataframe
        min_time = df[time_col].min()
        max_time = df[time_col].max()
        
        if isinstance(min_time, pd.Timestamp):
            time_range = pd.date_range(start=min_time, end=max_time, freq=freq)
        else:
            time_range = pd.Series(np.arange(min_time, max_time + freq, freq))
            
        if max_length and len(time_range) > max_length:
            time_range = time_range[:max_length]
            
        result_df = pd.DataFrame({time_col: time_range})
        
    return result_df


class MultiSourceTSDataSet(Dataset):
    """
    D1 Layer - PyTorch Dataset for loading time series data from multiple CSV files.
    
    This class handles raw data from multiple CSV files, performing basic preprocessing
    and providing access to the data via __getitem__ method.
    
    Key Features:
    - Handles multiple CSV files with different groups
    - Supports numerical and categorical features
    - Automatically extends data to ensure regular time intervals
    - Loads data on-demand to handle large datasets
    - Maintains consistent label encoding across all files
    - Treats groups as file-specific to handle large files efficiently
    """
    
    def __init__(
            self,
            file_paths: List[str],
            time: str,
            target: Union[str, List[str]],
            group: Union[str, List[str]],
            num: Optional[List[str]] = None,
            cat: Optional[List[str]] = None,
            static: Optional[List[str]] = None,
            weights: Optional[str] = None,
            chunk_size: int = 10000
        ):
        """
        Initialize the MultiSourceTSDataSet.
        
        Args:
            file_paths: List of paths to CSV files containing time series data
            time: The name of the time index column
            target: The name(s) of the target variable column(s)
            group: The name(s) of column(s) identifying a time series instance
            num: Optional list of numerical feature columns
            cat: Optional list of categorical feature columns
            static: Optional list of static feature columns
            weights: Optional name of weights column
            chunk_size: Number of rows to process at a time
        """
        # Store configuration parameters
        self.file_paths = file_paths
        self.time = time
        self.target = _coerce_to_list(target)
        self.group = _coerce_to_list(group)
        self.num = _coerce_to_list(num) if num else []
        self.cat = _coerce_to_list(cat) if cat else []
        self.static = _coerce_to_list(static) if static else []
        self.weights = weights
        self.chunk_size = chunk_size
        
        # Initialize feature columns (combination of numerical and categorical features)
        self.feature_cols = self.num + self.cat
        
        # Initialize label encoders for categorical columns
        self.label_encoders = {}
        
        # For compatibility with test code, initialize data attribute
        self.data = None
        
        # Process files to build metadata and encoders
        self._process_files()
        
        # Prepare metadata
        self._prepare_metadata()
    
    def _process_files(self):
        """
        Process each file to extract group information and update encoders.
        
        This method:
        1. Scans through all CSV files in chunks
        2. Identifies unique groups across all files
        3. Updates label encoders for categorical columns
        4. Builds a mapping of where each group's data is located
        5. Calculates the total length of each group
        6. Treats groups as file-specific (same group in different files gets different indices)
        """
        # Initialize data structures
        self.total_length = 0       # Total number of rows across all groups
        self.file_info = []         # Information about each group chunk in each file
        self.group_info = {}        # Maps (file_idx, group_key) to their locations in files
        self.lengths = {}           # Store the length of each group (for compatibility)
        self.file_group_map = []    # Maps global index to (file_idx, group_key) tuples
        
        print("Processing files to build metadata...")
        # Process each file
        for file_idx, file_path in enumerate(self.file_paths):
            print(f"\nProcessing file {file_idx + 1}/{len(self.file_paths)}: {file_path}")
            
            # Track groups in this file
            file_groups = set()
            
            # Read each file in chunks to handle large files
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                # Update label encoders with new categories from the chunk
                self._update_encoders(chunk)
                
                # Find all unique groups in the current chunk
                groups = chunk[self.group].drop_duplicates()
                
                # Process each group in the chunk
                for _, group_row in groups.iterrows():
                    # Create a key from the group columns' values
                    group_key = tuple(group_row[self.group].values)
                    # If there's only one group column, use the value directly instead of a tuple
                    if len(self.group) == 1:
                        group_key = group_key[0]
                    
                    # Create a file-specific group identifier
                    file_group_key = (file_idx, group_key)
                    file_groups.add(file_group_key)
                    
                    # Initialize group entry if not seen before
                    if file_group_key not in self.group_info:
                        self.group_info[file_group_key] = []
                        self.lengths[file_group_key] = 0  # Initialize length counter for this group
                    
                    # Filter data for the current group
                    group_mask = (chunk[self.group] == group_row).all(axis=1)
                    group_data = chunk[group_mask]
                    
                    # Store information about the group chunk
                    info = {
                        'file_idx': file_idx,
                        'file_path': file_path,
                        'group_key': group_key,
                        'file_group_key': file_group_key,
                        'start_time': group_data[self.time].min(),
                        'end_time': group_data[self.time].max(),
                        'row_count': len(group_data)
                    }
                    
                    # Add file info index to group's entry
                    self.group_info[file_group_key].append(len(self.file_info))
                    self.file_info.append(info)
                    
                    # Update total length and group-specific length
                    self.total_length += len(group_data)
                    self.lengths[file_group_key] += len(group_data)
            
            # Add all groups from this file to the global mapping
            for file_group_key in file_groups:
                self.file_group_map.append(file_group_key)
        
        # Store unique file-group combinations for iteration
        self._group_ids = list(self.group_info.keys())
        print(f"\nFound {len(self._group_ids)} unique file-group combinations")
        
        # For backward compatibility, create a mapping of original group keys
        self._original_group_ids = list(set(group_key for _, group_key in self._group_ids))
        print(f"Representing {len(self._original_group_ids)} unique group identifiers")
    
    def _update_encoders(self, data):
        """
        Update label encoders with new categories from the data.
        
        This method ensures consistent encoding across all files by:
        1. Checking each categorical column
        2. Creating new encoders if needed
        3. Updating existing encoders with new categories
        
        Args:
            data: DataFrame chunk to process
        """
        for col in self.cat + self.group:
            if col in data.columns:
                # Get unique values for this column
                unique_values = data[col].dropna().unique().reshape(-1, 1)
                if len(unique_values) > 0:
                    if col not in self.label_encoders:
                        # Create new encoder if it doesn't exist
                        self.label_encoders[col] = OrdinalEncoder()
                        self.label_encoders[col].fit(unique_values)
                    else:
                        # Update existing encoder with new categories
                        current_cats = self.label_encoders[col].categories_[0]
                        # Combine existing and new categories, ensuring uniqueness
                        new_cats = np.unique(np.concatenate([current_cats, unique_values.flatten()]))
                        self.label_encoders[col].categories_ = [new_cats]
    
    def _load_group_data(self, file_group_key):
        """
        Load data for a specific file-group combination.
        
        This method:
        1. Finds all file chunks containing data for the requested group
        2. Loads and filters only the relevant data
        3. Applies encoding to categorical features
        4. Combines all chunks into a single DataFrame
        
        Args:
            file_group_key: Tuple of (file_idx, group_key) to load data for
            
        Returns:
            DataFrame containing all data for the requested file-group combination
        """
        file_idx, group_key = file_group_key
        group_data = []
        
        # Get all file locations for this group
        file_indices = self.group_info[file_group_key]
        
        # Process each file chunk containing this group
        for idx in file_indices:
            info = self.file_info[idx]
            file_path = info['file_path']
            
            # Read the file in chunks
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                # Filter for the group
                if isinstance(group_key, tuple):
                    # For multi-column groups, check all columns
                    mask = np.ones(len(chunk), dtype=bool)
                    for col, val in zip(self.group, group_key):
                        mask &= (chunk[col] == val)
                else:
                    # For single-column groups, simple equality check
                    mask = chunk[self.group[0]] == group_key
                
                # If this chunk contains data for our group
                if mask.any():
                    group_chunk = chunk[mask].copy()
                    
                    # Encode categorical features
                    for col in self.cat:
                        if col in group_chunk.columns:
                            group_chunk[col] = self.label_encoders[col].transform(
                                group_chunk[col].values.reshape(-1, 1)
                            ).flatten()
                    
                    group_data.append(group_chunk)
        
        # Combine all chunks
        if group_data:
            combined_data = pd.concat(group_data, ignore_index=True)
            
            # Sort by time
            combined_data = combined_data.sort_values(self.time)
            
            # Add weight column if not present {NOT NECESSARY TO BE REMOVED}
            if self.weights is None:
                combined_data['_default_weight'] = 1.0
                self.weights = '_default_weight'
            elif self.weights not in combined_data.columns:
                # If weights column is specified but not in data, add it
                combined_data[self.weights] = 1.0
            
            # Mark valid rows (no NaN values in target or feature columns)
            check_cols = self.target + self.feature_cols
            combined_data['valid'] = ~combined_data[check_cols].isna().any(axis=1)
            
            return combined_data
        
        return pd.DataFrame()
    
    def __len__(self):
        """Return the number of file-group combinations in the dataset."""
        return len(self.file_group_map)
    
    def __getitem__(self, idx):
        """
        Get data for a specific file-group combination by index.
        
        This method:
        1. Maps the index to a specific file-group combination
        2. Loads all data for that combination
        3. Converts data to appropriate formats for model consumption
        
        Args:
            idx: Index of the file-group combination to retrieve
            
        Returns:
            Dictionary containing group data with keys:
            - 'x': Feature tensor
            - 'y': Target tensor
            - 't': Time values (as numpy array)
            - 'w': Weight tensor
            - 'v': Valid mask tensor
            - 'group_id': Group identifier
            - 'file_idx': File index
            - 'st': Static features (if available)
        """
        # Get the file-group combination for the requested index
        file_group_key = self.file_group_map[idx]
        file_idx, group_id = file_group_key
        
        # Load data for this file-group combination
        group_data = self._load_group_data(file_group_key)
        
        if len(group_data) == 0:
            raise ValueError(f"No data found for file {file_idx}, group {group_id}")
        
        # For compatibility with test code, store the last loaded group data
        self.data = group_data
        
        # Convert data to appropriate format
        x = torch.tensor(group_data[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(group_data[self.target].values, dtype=torch.float32)
        t = group_data[self.time].values  # Keep time as numpy array
        w = torch.tensor(group_data[self.weights].values, dtype=torch.float32)
        v = torch.tensor(group_data['valid'].values, dtype=torch.bool)
        
        # Prepare static features if available
        st = {}
        if self.static: 
            for col in self.static:
                if col in group_data.columns:
                    # Use the first non-null value for each static feature
                    val = group_data[col].dropna().iloc[0] if not group_data[col].isna().all() else None
                    st[col] = val
        
        return {
            'x': x,  # features
            'y': y,  # targets
            't': t,  # time indices (as numpy array)
            'w': w,  # weights
            'v': v,  # valid mask
            'group_id': group_id,  # group identifier
            'file_idx': file_idx,  # file index
            'st': st  # static features
        }
    
    def _prepare_metadata(self):
        """Prepare metadata about the dataset."""
        self.metadata = {
            "cols": {
                "y": self.target,
                "x": self.feature_cols,
                "st": self.static,
            },
            "col_type": {},
            "col_known": {},
            "weight": self.weights,
            "cat_index": []
        }
        
        # Mark categorical feature indices
        for i, c in enumerate(self.feature_cols):
            if c in self.cat:
                self.metadata['cat_index'].append(i)
        
        # Set column types and known status
        all_cols = self.target + self.feature_cols  + self.static
        for col in all_cols:
            self.metadata["col_type"][col] = "C" if col in self.cat else "F"
            # All columns are considered unknown for future values by default
            self.metadata["col_known"][col] = "U"
        
        # Add encoders to metadata
        self.metadata['encoders'] = self.label_encoders
    
    def get_metadata(self) -> Dict:
        """
        Return metadata about the dataset.
        
        Returns:
            Dictionary containing metadata about columns and their properties
        """
        return self.metadata


class TSDataProcessor(Dataset):
    """
    Works along the D2 Layer - PyTorch Dataset for processing time series data with sliding windows.
    
    This class processes D1 dataset output for model consumption, creating sliding windows 
    for inputs and targets, handling valid/invalid data points, and supporting precomputed 
    indices for efficient batch selection.
    
    Key Features:
    - Creates sliding windows for past inputs and future targets
    - Validates data points (no NaNs allowed)
    - Supports multiple time series (groups)
    - Implements global indexing across groups
    - Supports precomputed indices for valid samples
    """
    
    def __init__(
        self,
        d1_dataset: MultiSourceTSDataSet,
        past_steps: int,
        future_steps: int = 1,
        precompute: bool = True
    ):
        """
        Initialize the TSDataProcessor.
        
        Args:
            d1_dataset: The D1 dataset instance
            past_steps: Number of past time steps to use as input
            future_steps: Number of future time steps to predict
            precompute: Whether to precompute valid indices
        """
        self.d1_dataset = d1_dataset
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.precompute = precompute
        self.metadata = d1_dataset.metadata
        
        # Calculate and store valid indices for efficient access
        if precompute:
            self.valid_indices = self._compute_valid_indices()
            self.mapping = self._create_global_mapping()
            self.length = sum(len(indices) for indices in self.valid_indices.values())
        else:
            # Estimate total length (will check validity at runtime)
            total_len = 0
            for i in range(len(d1_dataset)):
                group_data = d1_dataset[i]
                # will claculate valid sequences considering total number o f time steps -past -future +inclusive counting
                group_len = max(0, len(group_data['t']) - past_steps - future_steps + 1)
                total_len += group_len
            self.length = total_len
    
    def _compute_valid_indices(self) -> Dict[int, List[int]]:
        """
        Compute valid indices for each group where sliding windows can be created.
        
        Returns:
            Dictionary mapping group indices to lists of valid start indices
        """
        valid_indices = {}
        
        # Process each group
        for i in range(len(self.d1_dataset)):
            group_data = self.d1_dataset[i]
            group_valid_indices = []
            
            # Get the validity mask for this group
            valid_mask = group_data['v']
            
            # Find valid windows (with enough past and future steps)
            for j in range(len(valid_mask) - self.past_steps - self.future_steps + 1):
                # check if all points in the window are valid
                if torch.all(valid_mask[j:j + self.past_steps + self.future_steps]):
                    group_valid_indices.append(j)
            
            valid_indices[i] = group_valid_indices
        
        return valid_indices
    
    def _create_global_mapping(self) -> Dict[int, Tuple[int, int]]:
        """
        Create a mapping from global index to (group_index, local_index). This will help treating
        the dataset as a flat sequence while preserving group structure. 
        
        Returns:
            Dictionary mapping global indices to (group_index, local_index) tuples
        """
        mapping = {}
        global_idx = 0
        
        # Create mapping for each valid index
        for group_idx, indices in self.valid_indices.items():
            for local_idx, _ in enumerate(indices):
                mapping[global_idx] = (group_idx, local_idx)
                global_idx += 1
        # now mapping dictionary allows easy lookup of group + local index using global index!
        return mapping
    
    def __len__(self) -> int:
        """Return the total number of valid windows in the dataset."""
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample (input/output window) from the dataset.
        
        Args:
            idx: Global index of the window to retrieve
            
        Returns:
            Dictionary containing input and output tensors for the window
        """
        if self.precompute:
            # Use precomputed mapping
            group_idx, local_idx = self.mapping[idx]
            group_data = self.d1_dataset[group_idx]
            start_idx = self.valid_indices[group_idx][local_idx]
        else:
            # Compute on-the-fly (less efficient)
            running_count = 0
            group_idx = 0
            
            # findd the group this index belongs to
            while group_idx < len(self.d1_dataset):
                group_data = self.d1_dataset[group_idx]
                group_len = max(0, len(group_data['t']) - self.past_steps - self.future_steps + 1)
                
                if running_count + group_len > idx:
                    # this is the group we want
                    local_idx = idx - running_count
                    break
                
                running_count += group_len
                group_idx += 1
            
            # If we couldn't find a valid group, return None
            if group_idx >= len(self.d1_dataset):
                return None
            
            # Find a valid starting point (skipping invalid windows)
            valid_count = 0
            start_idx = 0
            valid_mask = group_data['v']
            
            while start_idx <= len(valid_mask) - self.past_steps - self.future_steps:
                if torch.all(valid_mask[start_idx:start_idx + self.past_steps + self.future_steps]):
                    if valid_count == local_idx:
                        break
                    valid_count += 1
                start_idx += 1
            
            # If we couldn't find a valid window, return None
            if start_idx > len(valid_mask) - self.past_steps - self.future_steps:
                return None
        
        # Extract windows
        end_past = start_idx + self.past_steps
        end_future = end_past + self.future_steps
        
        # Create sample with past and future windows
        sample = {
            # Past window data
            "past_time": group_data['t'][start_idx:end_past],
            "past_target": group_data['y'][start_idx:end_past],
            "past_features": group_data['x'][start_idx:end_past],
            "past_weights": group_data['w'][start_idx:end_past],
            
            # Future window data
            "future_time": group_data['t'][end_past:end_future],
            "future_target": group_data['y'][end_past:end_future],
            "future_features": group_data['x'][end_past:end_future],
            "future_weights": group_data['w'][end_past:end_future],
            
            # Group and static information
            "group": group_data['group_id'],
            "static": group_data.get('st', {}),  # Fallback to empty dict if 'st' key is missing
        }
        
        return sample


class TSDataModule(L.LightningDataModule):
    """
    D2 layer: LightningDataModule for time series data with support for training, validation, and testing.
    It is utilizing the TSDataProcessor for initial processing of the data and fetching valid data instances. 
    This class manages the data loading and preprocessing, supporting both precomputed splits
    and on-the-fly splitting. It handles the creation of DataLoader instances for each stage.
    
    Key Features:
    - Supports train/validation/test splitting
    - Creates DataLoader instances with appropriate batch sizes
    - Handles both precomputed and on-the-fly split modes
    """
    
    def __init__(
        self,
        d1_dataset: MultiSourceTSDataSet,
        past_steps: int,
        future_steps: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_by: str = 'time',  # 'time' or 'group'
        precompute: bool = True
    ):
        """
        Initialize the TSDataModule.
        
        Args:
            d1_dataset: The D1 dataset instance
            past_steps: Number of past time steps for input
            future_steps: Number of future time steps for prediction
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for DataLoaders
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            split_by: How to split the data ('time' or 'group')
            precompute: Whether to precompute valid indices
        """
        super().__init__()
        self.d1_dataset = d1_dataset
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_by = split_by
        self.precompute = precompute
    
    def prepare_data(self):
        """Prepare data for training, validation, and testing."""
        # This method is called only on 1 GPU/TPU
        # Perform global preprocessing if needed
        pass
    
    def setup(self, stage=None):
        """
        Set up datasets for each stage.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Create D2 processor with full dataset
        d2_processor = TSDataProcessor(
            d1_dataset=self.d1_dataset,
            past_steps=self.past_steps,
            future_steps=self.future_steps,
            precompute=self.precompute
        )
        """ example of split implemenetd. 
        Split by Time:
            Train: First 700 data points.
            Val: Next 200 data points.
            Test: Last 100 data points.
        Split by Groups:
            Train: Indices from the first 7 groups.
            Val: Indices from the next 2 groups.
            Test: Indices from the last group.

        """
        # Split indices based on split_by method
        if self.split_by == 'time':
            # Split by time (sequential split)
            train_end = int(len(d2_processor) * self.train_ratio)
            val_end = train_end + int(len(d2_processor) * self.val_ratio)
            
            # Define index ranges for each split
            self.train_indices = list(range(train_end))
            self.val_indices = list(range(train_end, val_end))
            self.test_indices = list(range(val_end, len(d2_processor)))
        else:
            # Split by groups
            n_groups = len(self.d1_dataset)
            train_end = int(n_groups * self.train_ratio)
            val_end = train_end + int(n_groups * self.val_ratio)
            
            # Define group ranges for each split
            train_groups = list(range(train_end))
            val_groups = list(range(train_end, val_end))
            test_groups = list(range(val_end, n_groups))
            
            # Create index lists based on group assignment
            self.train_indices = []
            self.val_indices = []
            self.test_indices = []
            
            for idx in range(len(d2_processor)):
                # For precomputed mode
                if self.precompute:
                    group_idx, _ = d2_processor.mapping[idx]
                else:
                    # Simplified approach for non-precomputed mode
                    # this might need refinement in practice
                    running_count = 0
                    group_idx = 0
                    
                    for i in range(len(self.d1_dataset)):
                        group_data = self.d1_dataset[i]
                        group_len = max(0, len(group_data['t']) - self.past_steps - self.future_steps + 1)
                        
                        if running_count + group_len > idx:
                            group_idx = i
                            break
                        
                        running_count += group_len
                
                # Assign to split based on group
                if group_idx in train_groups:
                    self.train_indices.append(idx)
                elif group_idx in val_groups:
                    self.val_indices.append(idx)
                elif group_idx in test_groups:
                    self.test_indices.append(idx)
        
        # Create subset datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = torch.utils.data.Subset(d2_processor, self.train_indices)
            self.val_dataset = torch.utils.data.Subset(d2_processor, self.val_indices)
        
        if stage == 'test' or stage is None:
            self.test_dataset = torch.utils.data.Subset(d2_processor, self.test_indices)
        
        if stage == 'predict' or stage is None:
            # For prediction, we can use the test set or a custom set
            self.predict_dataset = self.test_dataset
    
    def train_dataloader(self):
        """Get training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Get validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        """Get prediction DataLoader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def create_example_files():
    """Create example CSV files with different groups."""
    # File 1 with groups A and B
    df1 = pd.DataFrame({
        'time': pd.date_range(start='2025-01-01', periods=100, freq='D'),
        'group': ['A'] * 50 + ['B'] * 50,
        'value': np.random.randn(100),
        'category': np.random.choice(['cat1', 'cat2'], size=100)
    })
    df1.to_csv('file1.csv', index=False)  # Save to CSV
    
    # File 2 with groups B and C
    df2 = pd.DataFrame({
        'time': pd.date_range(start='2025-02-01', periods=100, freq='D'),
        'group': ['B'] * 50 + ['C'] * 50,
        'value': np.random.randn(100),
        'category': np.random.choice(['cat2', 'cat3'], size=100)
    })
    df2.to_csv('file2.csv', index=False)  # Save to csv


def test_dataset():
    """Test the MultiSourceTSDataSet and TSDataProcessor with example files."""
    print("Starting test_dataset...\n")
    
    # Create example files
    create_example_files()
    
    # Initialize D1 dataset
    print("\nInitializing D1 Dataset...")
    d1_dataset = MultiSourceTSDataSet(
        file_paths=['file1.csv', 'file2.csv'],
        time='time',
        target='value',
        group='group',
        cat=['category']
    )
    
    print(f"\nD1 Dataset Info:")
    print(f"  Total Groups: {len(d1_dataset)}")
    print(f"  Group IDs: {d1_dataset._group_ids}")
    print(f"  Group lengths: {d1_dataset.lengths}")
    print(f"  Feature columns: {d1_dataset.feature_cols}")
    print(f"  Total data points: {d1_dataset.total_length}")
    
    # Load a sample group to demonstrate functionality
    print("\nLoading sample group data (group 'A')...")
    sample_group = d1_dataset[0]  # Get first group
    print(f"  Group ID: {sample_group['group_id']}")
    print(f"  Time points: {len(sample_group['t'])}")
    print(f"  Features shape: {sample_group['x'].shape}")
    print(f"  Target shape: {sample_group['y'].shape}")
    print("-" * 80)
    
    # Create D2 processor
    print("\nInitializing D2 Processor...")
    d2_processor = TSDataProcessor(
        d1_dataset=d1_dataset,
        past_steps=3,
        future_steps=1,
        precompute=True
    )
    
    print(f"\nD2 Processor Info:")
    print(f"  Total samples: {len(d2_processor)}")
    
    # Get a sample from D2 processor
    print("\nAttempting to get first sample from D2 Processor...")
    sample = d2_processor[0]
    
    print("\nSample structure:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor of shape {v.shape}")
        elif isinstance(v, np.ndarray):
            print(f"  {k}: Array of shape {v.shape}")
        else:
            print(f"  {k}: {v}")

    print("Sample output:")
    print("{")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  '{k}': Tensor of shape {v.shape} with values:{v}")
        elif isinstance(v, np.ndarray):
            print(f"  '{k}': Array of shape {v.shape} with values:{v}")
        else:
            print(f"  '{k}': {v}")
    print("}")
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_dataset()