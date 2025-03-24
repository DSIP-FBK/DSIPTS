"""
Time Series Dataset Module

This module provides classes for two-layer time series data handling:
- MultiSourceTSDataSet (D1 Layer): Handles raw data from multiple CSV files
- TSDataModule (D2 Layer): LightningDataModule for time series data with support for training, validation, and testing.

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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
    - Can load data on-demand or preload based on memory constraints
    - Maintains consistent label encoding across all files
    - Treats groups as file-specific to handle large files efficiently
    - Preserves NaN values for valid index computation in D2 layer
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
            memory_efficient: bool = False,
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
            memory_efficient: If True, load files on demand; if False, preload smaller files
            chunk_size: Number of rows to process at a time for memory-efficient mode
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
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        
        # Initialize feature columns (combination of numerical and categorical features)
        self.feature_cols = self.num + self.cat
        
        # Initialize label encoders for categorical columns
        self.label_encoders = {}
        
        # For compatibility with test code, initialize data attribute
        self.data = None
        
        # Pre-loaded data cache (only used when memory_efficient=False)
        self.data_cache = {}
        
        # Process files to build metadata and encoders
        self._process_files()
        
        # Prepare metadata
        self._prepare_metadata()
        
        # Preload data if memory_efficient is False
        if not self.memory_efficient:
            self._preload_data()
    
    def _process_files(self):
        """
        Process each file to extract group information and update encoders.
        
        This method:
        1. Scans through all CSV files (in chunks if memory_efficient=True)
        2. Identifies unique groups across all files
        3. Updates label encoders for categorical columns
        4. Builds a mapping of where each group's data is located
        5. Calculates the total length of each group
        6. Treats groups as file-specific to handle large files efficiently
        7. Preserves NaN values for valid index computation in D2 layer
        """
        # Initialize data structures
        self.total_length = 0       # Total number of rows across all groups
        self.file_info = []         # Information about each group in each file
        self.group_info = {}        # Maps (file_idx, group_key) to their locations in files
        self.lengths = {}           # Store the length of each group (for compatibility)
        self.file_group_map = []    # Maps global index to (file_idx, group_key) tuples
        self.file_sizes = []        # Store file sizes for memory management
        
        print("Processing files to build metadata...")
        # Process each file
        for file_idx, file_path in enumerate(self.file_paths):
            print(f"\nProcessing file {file_idx + 1}/{len(self.file_paths)}: {file_path}")
            
            # Track groups in this file
            file_groups = set()
            
            # Get file size
            file_size = os.path.getsize(file_path)
            self.file_sizes.append(file_size)
            
            if self.memory_efficient:
                # Process in chunks for memory efficiency
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    self._process_chunk(chunk, file_idx, file_path, file_groups)
            else:
                # Load entire file at once for small files
                chunk = pd.read_csv(file_path)
                self._process_chunk(chunk, file_idx, file_path, file_groups)
            
            # Add all groups from this file to the global mapping
            for file_group_key in file_groups:
                self.file_group_map.append(file_group_key)
        
        # Store unique file-group combinations for iteration
        self._group_ids = list(self.group_info.keys())
        print(f"\nFound {len(self._group_ids)} unique file-group combinations")
        
        # For backward compatibility, create a mapping of original group keys
        self._original_group_ids = list(set(group_key for _, group_key in self._group_ids))
        print(f"Representing {len(self._original_group_ids)} unique group identifiers")
    
    def _process_chunk(self, chunk, file_idx, file_path, file_groups):
        """
        Process a chunk of data from a file.
        
        Args:
            chunk: DataFrame chunk to process
            file_idx: Index of the file being processed
            file_path: Path to the file being processed
            file_groups: Set to track groups in this file
        """
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
            
            # Store information about the group
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
    
    def _preload_data(self):
        """
        preload all data from files for faster access (wee only used when memory_efficient=False).
        """
        print("Preloading data into memory...")
        for file_idx, file_path in enumerate(self.file_paths):
            print(f"Loading file {file_idx + 1}/{len(self.file_paths)}: {file_path}")
            self.data_cache[file_idx] = pd.read_csv(file_path)
            
            # Encode categorical features
            for col in self.cat:
                if col in self.data_cache[file_idx].columns:
                    self.data_cache[file_idx][col] = self.label_encoders[col].transform(
                        self.data_cache[file_idx][col].values.reshape(-1, 1)
                    ).flatten()
        print("Data preloading complete.")
    
    def _load_group_data(self, file_group_key):
        """
        Load data for a specific file-group combination.
        
        This method:
        1. Gets the file containing data for the requested group
        2. Loads from cache or from disk based on memory_efficient setting
        3. Applies encoding to categorical features if needed
        4. Preserves NaN values for valid index computation in D2 layer
        
        Args:
            file_group_key: Tuple of (file_idx, group_key) to load data for
            
        Returns:
            DataFrame containing all data for the requested file-group combination
        """
        file_idx, group_key = file_group_key
        
        # Get file information for this group
        file_indices = self.group_info[file_group_key]
        info = self.file_info[file_indices[0]]
        file_path = info['file_path']
        
        # Different data loading strategies based on memory efficiency setting
        if not self.memory_efficient and file_idx in self.data_cache:
            # Use preloaded data from cache
            file_data = self.data_cache[file_idx]
        else:
            # Load from file
            if self.memory_efficient:
                # We'll accumulate data from chunks
                group_chunks = []
                
                # Read file in chunks
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
                        
                        group_chunks.append(group_chunk)
                
                # Combine all chunks
                if group_chunks:
                    group_data = pd.concat(group_chunks, ignore_index=True)
                else:
                    return pd.DataFrame()
            else:
                # Load entire file at once
                file_data = pd.read_csv(file_path)
                
                # Encode categorical features
                for col in self.cat:
                    if col in file_data.columns:
                        file_data[col] = self.label_encoders[col].transform(
                            file_data[col].values.reshape(-1, 1)
                        ).flatten()
                
                # Filter for the group
                if isinstance(group_key, tuple):
                    # For multi-column groups, check all columns
                    mask = np.ones(len(file_data), dtype=bool)
                    for col, val in zip(self.group, group_key):
                        mask &= (file_data[col] == val)
                else:
                    # For single-column groups, simple equality check
                    mask = file_data[self.group[0]] == group_key
                
                # If this file contains data for our group
                if mask.any():
                    group_data = file_data[mask].copy()
                else:
                    return pd.DataFrame()
        
        # Sort by time
        group_data = group_data.sort_values(self.time)
        
        # Mark valid rows (no NaN values in target or feature columns)
        check_cols = self.target + self.feature_cols
        group_data['valid'] = ~group_data[check_cols].isna().any(axis=1)
        
        # IMPORTANT: We preserve NaN values for valid index computation in D2 layer
        
        return group_data
    
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
            - 'v': Valid mask tensor - will be used in D2 layer #TODO: REMOVE?
            - 'group_id': Group identifier
            - 'st': Static features
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
        v = torch.tensor(group_data['valid'].values, dtype=torch.bool)
        
        # Handle weights properly without modifying the DataFrame
        if self.weights is None or self.weights not in group_data.columns:
            # Create a tensor of ones if weights column is not specified or not present
            w = torch.ones(len(group_data), dtype=torch.float32)
        else:
            # Use the specified weights column
            w = torch.tensor(group_data[self.weights].values, dtype=torch.float32)
        
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
            'v': v,  # valid mask - will be used in D2 layer
            'group_id': group_id,  # group identifier
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


class TSDataModule(L.LightningDataModule):
    """D2 Layer - Processes time series data for model consumption.
    
    This class:
    1. Takes a D1 dataset as input
    2. Creates sliding windows of time series data
    3. Splits data into train/validation/test sets
    4. Manages memory efficiently based on data size and configuration
    5. Provides samples on demand with target and feature values
    6. Integrates with PyTorch Lightning for model training
    """
    
    def __init__(
        self,
        d1_dataset: MultiSourceTSDataSet,
        past_len: int,
        future_len: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
        precompute: bool = True,
        split_config=None,
        split_method: str = 'percentage',
        split_seed: int = 42,
        min_valid_length: int = 1,
        sampler=None
    ):
        """
        Initialize the TSDataModule.
        
        Args:
            d1_dataset: The D1 dataset instance (MultiSourceTSDataSet)
            past_len: Number of past time steps for input
            future_len: Number of future time steps for prediction
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for DataLoaders
            precompute: Whether to precompute valid indices
            split_config: Configuration for data splitting:
                          - For 'percentage' method: (train%, val%, test%)
                          - For 'group' method: (train_groups, val_groups, test_groups)
            split_method: Method for splitting data ('percentage' or 'group')
            split_seed: Random seed for reproducible splits
            min_valid_length: Minimum valid consecutive points for a window
            sampler: Optional custom sampler for the DataLoader
        """
        super().__init__()
        self.d1_dataset = d1_dataset
        self.past_len = past_len
        self.future_len = future_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.precompute = precompute
        self.split_method = split_method
        self.split_seed = split_seed
        self.min_valid_length = min_valid_length
        self.sampler = sampler
        
        # Set feature and target columns
        self.feature_cols = d1_dataset.feature_cols
        self.target_cols = d1_dataset.target
        self.static_cols = d1_dataset.static
        
        # Store reference to D1 dataset metadata
        self.metadata = d1_dataset.metadata
        
        # For tracking loaded data
        self.loaded_groups = {}
        
        # Default split configuration based on method
        if split_config is None:
            if self.split_method == 'percentage':
                self.split_config = (0.7, 0.15, 0.15)  # Default percentages
            elif self.split_method == 'group':
                # Will be set during setup based on available groups
                self.split_config = None
        else:
            self.split_config = split_config
        
        # Datasets and indices
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # These will be initialized during setup
        self.valid_indices = {}
        self.mapping = []
        self.length = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        # Initialize if precompute is enabled
        if precompute:
            self._initialize()
    
    def _initialize(self):
        """Initialize the dataset by computing valid indices and mappings."""
        print("Precomputing valid indices and mappings...")
        self.valid_indices = self._compute_valid_indices()
        self.mapping = self._create_global_mapping()
        self.length = len(self.mapping)
        
        # Create train/validation/test splits if provided
        if self.split_config is not None:
            self.train_indices, self.val_indices, self.test_indices = self._create_splits(self.split_config)
            print(f"Split statistics: Train: {len(self.train_indices)}, Validation: {len(self.val_indices)}, Test: {len(self.test_indices)}")
        else:
            # Default to all indices as training
            self.train_indices = list(range(self.length))
            self.val_indices = []
            self.test_indices = []
    
    def _compute_valid_indices(self):
        """
        Compute valid indices for all groups in the dataset.
        
        A valid index is one where:
        1. There are enough past and future time steps
        2. All required data points are valid (no NaN values)
        
        Returns:
            Dictionary mapping group indices to lists of valid indices
        """
        valid_indices = {}
        
        for i in range(len(self.d1_dataset)):
            # Load group data from D1 dataset
            group_data = self.d1_dataset[i]
            group_id = group_data['group_id']
            
            # Fetch valid mask and find valid indices
            valid_mask = group_data['v'].numpy()
            
            # Get the time series length
            ts_length = len(valid_mask)
            
            # Find valid windows
            # A window is valid if:
            # 1. It's within the time series (has enough past and future points)
            # 2. All points in the window are valid (no NaN values)
            group_valid_indices = []
            
            for t in range(ts_length - (self.past_len + self.future_len) + 1):
                end_past = t + self.past_len
                end_future = end_past + self.future_len
                
                # Check if all points in the window are valid
                past_valid = valid_mask[t:end_past].sum() >= self.min_valid_length
                future_valid = valid_mask[end_past:end_future].sum() >= 1  # At least one future point must be valid
                
                if past_valid and future_valid:
                    group_valid_indices.append(t)
            
            # Store indices if there are any valid ones
            if group_valid_indices:
                valid_indices[i] = group_valid_indices
        
        # Print statistics
        total_valid = sum(len(indices) for indices in valid_indices.values())
        print(f"Found {total_valid} valid windows across {len(valid_indices)} groups")
        
        return valid_indices
    
    def _create_global_mapping(self):
        """
        Create a global mapping from index to (group_idx, local_idx).
        
        This allows O(1) lookup of samples by global index.
        
        Returns:
            List of (group_idx, local_idx) tuples
        """
        mapping = []
        
        # Iterate through all groups with valid indices
        for group_idx, local_indices in self.valid_indices.items():
            # Add each local index with its group index to the mapping
            for local_idx in local_indices:
                mapping.append((group_idx, local_idx))
        
        return mapping
    
    def _create_splits(self, split_config):
        """
        Create train/validation/test splits based on the specified method.
        
        Args:
            split_config: Either a tuple of percentages (train%, val%, test%) or
                          a tuple of lists (train_groups, val_groups, test_groups)
                          
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Set random seed for reproducibility
        np.random.seed(self.split_seed)
        
        if self.split_method == 'percentage':
            # Split by percentage within each group
            train_pct, val_pct, test_pct = split_config
            
            # Validate percentages
            total_pct = train_pct + val_pct + test_pct
            if not np.isclose(total_pct, 1.0):
                raise ValueError(f"Split percentages must sum to 1.0, got {total_pct}")
            
            # Create a mapping from group_idx to list of global indices
            group_to_global_indices = {}
            for global_idx, (group_idx, _) in enumerate(self.mapping):
                if group_idx not in group_to_global_indices:
                    group_to_global_indices[group_idx] = []
                group_to_global_indices[group_idx].append(global_idx)
            
            # Calculate split indices
            train_indices = []
            val_indices = []
            test_indices = []
            
            # Process each group
            for group_idx, global_indices in group_to_global_indices.items():
                # Shuffle indices for this group
                shuffled_indices = np.random.permutation(global_indices)
                n_indices = len(shuffled_indices)
                
                # Calculate split points
                n_train = int(n_indices * train_pct)
                n_val = int(n_indices * val_pct)
                
                # Add to respective splits
                train_indices.extend(shuffled_indices[:n_train])
                val_indices.extend(shuffled_indices[n_train:n_train + n_val])
                test_indices.extend(shuffled_indices[n_train + n_val:])
            
            print(f"Percentage-based split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)} samples")
            
        elif self.split_method == 'group':
            # Split by assigning entire groups to train/val/test
            train_groups, val_groups, test_groups = split_config
            
            # Validate groups
            all_groups = set()
            for group_idx in self.valid_indices.keys():
                group_id = self.d1_dataset[group_idx]['group_id']
                all_groups.add(group_id)
            
            # Ensure all groups are accounted for
            specified_groups = set(train_groups + val_groups + test_groups)
            if specified_groups != all_groups:
                missing = all_groups - specified_groups
                extra = specified_groups - all_groups
                message = []
                if missing:
                    message.append(f"Missing groups: {missing}")
                if extra:
                    message.append(f"Extra groups not in dataset: {extra}")
                raise ValueError("Invalid group specification: " + ", ".join(message))
            
            # Create a mapping from group_idx to group_id and global indices
            group_idx_to_id = {}
            group_idx_to_global_indices = {}
            
            for global_idx, (group_idx, _) in enumerate(self.mapping):
                if group_idx not in group_idx_to_id:
                    group_idx_to_id[group_idx] = self.d1_dataset[group_idx]['group_id']
                    group_idx_to_global_indices[group_idx] = []
                group_idx_to_global_indices[group_idx].append(global_idx)
            
            # Create mappings
            train_indices = []
            val_indices = []
            test_indices = []
            
            # Process each group
            for group_idx, global_indices in group_idx_to_global_indices.items():
                group_id = group_idx_to_id[group_idx]
                
                # Add to respective splits
                if group_id in train_groups:
                    train_indices.extend(global_indices)
                elif group_id in val_groups:
                    val_indices.extend(global_indices)
                elif group_id in test_groups:
                    test_indices.extend(global_indices)
            
            print(f"Group-based split - Train: {len(train_indices)} ({len(train_groups)} groups), "
                  f"Val: {len(val_indices)} ({len(val_groups)} groups), "
                  f"Test: {len(test_indices)} ({len(test_groups)} groups) samples")
        
        else:
            raise ValueError(f"Unknown split method: {self.split_method}. Use 'percentage' or 'group'")
        
        return train_indices, val_indices, test_indices
    
    def __len__(self):
        """Return the number of valid samples in the dataset."""
        if self.length is not None:
            return self.length
        else:
            # Calculate length on first access if not precomputed
            self.valid_indices = self._compute_valid_indices()
            self.mapping = self._create_global_mapping()
            self.length = len(self.mapping)
            return self.length
    
    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        In precompute mode, this uses the precomputed mapping.
        In lazy mode, it may need to compute valid indices on the fly.
        
        Args:
            idx: Global index of the sample to retrieve
            
        Returns:
            Dictionary with past and future data for the sample
        """
        # Ensure mapping is initialized
        if not self.mapping:
            if self.precompute:
                raise RuntimeError("No valid samples found in the dataset")
            else:
                # In lazy mode, compute the mapping if not done yet
                self.valid_indices = self._compute_valid_indices()
                self.mapping = self._create_global_mapping()
                self.length = len(self.mapping)
        
        if idx >= len(self.mapping):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.mapping)}")
            
        group_idx, local_idx = self.mapping[idx]
        
        # Get group data, either from cache or by loading
        if group_idx in self.loaded_groups:
            group_data = self.loaded_groups[group_idx]
        else:
            group_data = self.d1_dataset[group_idx]
            self.loaded_groups[group_idx] = group_data
        
        # Extract time window
        start_past = local_idx
        end_past = start_past + self.past_len
        start_future = end_past
        end_future = start_future + self.future_len
        
        # Get past data
        past_time = group_data['t'][start_past:end_past]
        past_target = group_data['y'][start_past:end_past]
        past_features = group_data['x'][start_past:end_past]
        past_weights = group_data['w'][start_past:end_past]
        
        # Get future data
        future_time = group_data['t'][start_future:end_future]
        future_target = group_data['y'][start_future:end_future]
        future_features = group_data['x'][start_future:end_future]
        future_weights = group_data['w'][start_future:end_future]
        
        # Get group ID and static features
        group_id = group_data['group_id']
        static_features = group_data['st']
        
        # Return sample as a dictionary
        return {
            'past_time': past_time,
            'past_target': past_target,
            'past_features': past_features,
            'past_weights': past_weights,
            'future_time': future_time,
            'future_target': future_target,
            'future_features': future_features,
            'future_weights': future_weights,
            'group': group_id,
            'static': static_features
        }
    
    def setup(self, stage=None):
        """
        Set up the datasets for train, validation, and test.
        
        This method is called by Lightning before training/validation/testing.
        """
        # Initialize if not already done
        if not self.precompute and self.mapping == []:
            self._initialize()
            
        # For group-based splitting, if no config was provided, auto-assign groups
        if self.split_method == 'group' and self.split_config is None:
            # Get all unique groups
            group_ids = set()
            for i in range(len(self.d1_dataset)):
                group_ids.add(self.d1_dataset[i]['group_id'])
            group_ids = sorted(list(group_ids))
            
            # Auto-split: 70% train, 15% val, 15% test
            n_groups = len(group_ids)
            n_train = int(n_groups * 0.7)
            n_val = int(n_groups * 0.15)
            
            train_groups = group_ids[:n_train]
            val_groups = group_ids[n_train:n_train + n_val]
            test_groups = group_ids[n_train + n_val:]
            
            self.split_config = (train_groups, val_groups, test_groups)
            print(f"Auto-assigned groups: {n_train} train, {len(val_groups)} validation, {len(test_groups)} test")
            
            # Create splits with the new config
            self.train_indices, self.val_indices, self.test_indices = self._create_splits(self.split_config)
        
        # Create subset datasets for train/val/test using the indices
        if stage == 'fit' or stage is None:
            self.train_dataset = TimeSeriesSubset(self, self.train_indices)
            self.val_dataset = TimeSeriesSubset(self, self.val_indices)
            
        if stage == 'test' or stage is None:
            self.test_dataset = TimeSeriesSubset(self, self.test_indices)
    
    def train_dataloader(self):
        """Return a DataLoader for training."""
        if self.sampler is not None:
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size,
                sampler=self.sampler(self.train_dataset),
                num_workers=self.num_workers
            )
        else:
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
    
    def val_dataloader(self):
        """Return a DataLoader for validation."""
        if len(self.val_dataset) == 0:
            return None
            
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        """Return a DataLoader for testing."""
        if len(self.test_dataset) == 0:
            return None
            
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class TimeSeriesSubset(Dataset):
    """Subset of a D2 processor dataset that implements the Dataset interface."""
    
    def __init__(self, data_module, indices):
        """
        Initialize the TimeSeriesSubset.
        
        Args:
            data_module: The TSDataModule instance
            indices: List of indices to include in this subset
        """
        self.data_module = data_module
        self.indices = indices
    
    def __len__(self):
        """Return the number of samples in this subset."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a sample from the data module using the mapped index."""
        return self.data_module[self.indices[idx]]


def create_example_files():
    """Create example CSV files with different groups."""
    print("Creating example files with multiple groups...")
    
    # File 1 with groups A and B
    print("  Creating file1.csv with groups A and B")
    df1 = pd.DataFrame({
        'time': pd.date_range(start='2025-01-01', periods=100, freq='D'),
        'group': ['A'] * 50 + ['B'] * 50,
        'value': np.random.randn(100),
        'category': np.random.choice(['cat1', 'cat2'], size=100)
    })
    df1.to_csv('file1.csv', index=False)  # Save to CSV
    
    # File 2 with groups B and C
    print("  Creating file2.csv with groups B and C")
    df2 = pd.DataFrame({
        'time': pd.date_range(start='2025-02-01', periods=100, freq='D'),
        'group': ['B'] * 50 + ['C'] * 50,
        'value': np.random.randn(100),
        'category': np.random.choice(['cat2', 'cat3'], size=100)
    })
    df2.to_csv('file2.csv', index=False)  # Save to csv
    
    # File 3 with groups C, D and E
    print("  Creating file3.csv with groups C, D and E")
    df3 = pd.DataFrame({
        'time': pd.date_range(start='2025-03-01', periods=150, freq='D'),
        'group': ['C'] * 50 + ['D'] * 50 + ['E'] * 50,
        'value': np.random.randn(150),
        'category': np.random.choice(['cat3', 'cat4', 'cat5'], size=150)
    })
    df3.to_csv('file3.csv', index=False)
    
    # File 4 with groups F, G and H
    print("  Creating file4.csv with groups F, G and H")
    df4 = pd.DataFrame({
        'time': pd.date_range(start='2025-04-01', periods=150, freq='D'),
        'group': ['F'] * 50 + ['G'] * 50 + ['H'] * 50,
        'value': np.random.randn(150),
        'category': np.random.choice(['cat4', 'cat5', 'cat6'], size=150)
    })
    df4.to_csv('file4.csv', index=False)
    
    print("  Example files created successfully")


def test_dataset():
    """Test the time series dataset implementation with both memory modes and splitting strategies."""
    print("=== Time Series Dataset Test ===")
    
    # Create example files if they don't exist
    create_example_files()
    
    # Set up paths to example files
    file_paths = [
        "example_data_1.csv",
        "example_data_2.csv",
    ]
    
    # Test with default settings (memory efficient mode OFF)
    print("\n=== Testing with memory_efficient=False (preload all data) ===")
    d1_dataset = MultiSourceTSDataSet(
        file_paths=file_paths,
        feature_cols=["x1", "x2"],
        target=["y1"],
        static=["static1", "static2"],
        group="group",
        memory_efficient=False
    )
    
    print("\nD1 Dataset Stats:")
    print(f"  Number of groups: {len(d1_dataset)}")
    print(f"  Feature columns: {d1_dataset.feature_cols}")
    print(f"  Target columns: {d1_dataset.target}")
    print(f"  Static columns: {d1_dataset.static}")
    print(f"  Memory mode: {'Efficient' if d1_dataset.memory_efficient else 'Preloaded'}")
    
    # Get a sample from D1 dataset
    print("\nSample from D1 dataset (first group):")
    sample = d1_dataset[0]
    
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor of shape {value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    # Test precompute mode with percentage-based splits
    print("\n=== Testing D2 Module with Percentage Splits ===")
    d2_module_pct = TSDataModule(
        d1_dataset=d1_dataset,
        past_len=3,
        future_len=1,
        precompute=True,
        batch_size=8,
        split_config=(0.7, 0.15, 0.15),
        split_method='percentage',
        split_seed=42
    )
    
    print(f"D2 Module Stats (Percentage Split):")
    print(f"  Total samples: {len(d2_module_pct)}")
    print(f"  Past steps: {d2_module_pct.past_len}")
    print(f"  Future steps: {d2_module_pct.future_len}")
    print(f"  Train samples: {len(d2_module_pct.train_indices)}")
    print(f"  Validation samples: {len(d2_module_pct.val_indices)}")
    print(f"  Test samples: {len(d2_module_pct.test_indices)}")
    
    # Get a sample from train split
    if d2_module_pct.train_indices:
        train_idx = d2_module_pct.train_indices[0]
        print(f"\nSample from D2 train split (index {train_idx}):")
        train_sample = d2_module_pct[train_idx]
        print(f"  Past time shape: {train_sample['past_time'].shape}")
        print(f"  Past features shape: {train_sample['past_features'].shape}")
        print(f"  Future target shape: {train_sample['future_target'].shape}")
    
    # Test with group-based splits
    print("\n=== Testing D2 Module with Group-Based Splits ===")
    # Get all unique groups
    group_ids = set()
    for i in range(len(d1_dataset)):
        group_ids.add(d1_dataset[i]['group_id'])
    group_ids = sorted(list(group_ids))
    
    # Split groups manually for demonstration
    n_groups = len(group_ids)
    n_train = max(1, int(n_groups * 0.6))
    n_val = max(1, int(n_groups * 0.2))
    
    train_groups = group_ids[:n_train]
    val_groups = group_ids[n_train:n_train + n_val]
    test_groups = group_ids[n_train + n_val:]
    
    d2_module_group = TSDataModule(
        d1_dataset=d1_dataset,
        past_len=3,
        future_len=1,
        precompute=True,
        batch_size=8,
        split_config=(train_groups, val_groups, test_groups),
        split_method='group',
        split_seed=42
    )
    
    print(f"D2 Module Stats (Group Split):")
    print(f"  Total samples: {len(d2_module_group)}")
    print(f"  Train groups: {train_groups}")
    print(f"  Train samples: {len(d2_module_group.train_indices)}")
    print(f"  Validation groups: {val_groups}")
    print(f"  Validation samples: {len(d2_module_group.val_indices)}")
    print(f"  Test groups: {test_groups}")
    print(f"  Test samples: {len(d2_module_group.test_indices)}")
    
    # Test with memory_efficient=True
    print("\n=== Testing with memory_efficient=True (chunked processing) ===")
    d1_dataset_efficient = MultiSourceTSDataSet(
        file_paths=file_paths,
        feature_cols=["x1", "x2"],
        target=["y1"],
        static=["static1", "static2"],
        group="group",
        memory_efficient=True
    )
    
    # Test PyTorch Lightning Integration with TSDataModule
    print("\n=== Testing PyTorch Lightning Integration ===")
    # Setup the datamodule
    d2_module_pct.setup()
    
    # Get dataloaders
    train_loader = d2_module_pct.train_dataloader()
    val_loader = d2_module_pct.val_dataloader()
    test_loader = d2_module_pct.test_dataloader()
    
    print(f"Train DataLoader: {len(train_loader)} batches of size {d2_module_pct.batch_size}")
    if val_loader:
        print(f"Validation DataLoader: {len(val_loader)} batches of size {d2_module_pct.batch_size}")
    if test_loader:
        print(f"Test DataLoader: {len(test_loader)} batches of size {d2_module_pct.batch_size}")
    
    # Show one batch from the train loader
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1} structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor of shape {value.shape}")
            else:
                print(f"  {key}: {value}")
        # Just show the first batch
        break
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_dataset()