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
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import timedelta
from sklearn.preprocessing import OrdinalEncoder
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    Layer 1 (D1) dataset for multi-source time series data.
    
    This dataset:
    1. Loads time series data from multiple CSV files
    2. Handles categorical encoding and normalization
    3. Efficiently processes data in chunks for memory-efficient operation
    4. Preserves NaN values for D2 layer to handle
    
    It does NOT compute validity of windows or create sliding windows - that is
    the responsibility of the D2 layer (TSDataProcessor).
    """
    
    def __init__(
        self,
        file_paths: List[str],
        group_cols: Union[str, List[str]],
        time_col: str,
        feature_cols: List[str],
        target_cols: List[str],
        static_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        known_cols: Optional[List[str]] = None,
        unknown_cols: Optional[List[str]] = None,
        weights: Optional[str] = None,
        memory_efficient: bool = False,
        chunk_size: int = 10000
    ):
        """
        Initialize the MultiSourceTSDataSet.
        
        Args:
            file_paths: List of paths to CSV files containing time series data
            group_cols: Column(s) that identify unique time series groups
            time_col: Column containing time/date information
            feature_cols: Columns to use as features (X)
            target_cols: Columns to use as targets (y)
            static_cols: Columns with static (non-time-varying) features
            cat_cols: Categorical columns that need encoding
            num_cols: Numerical columns (if None, all non-categorical columns are treated as numerical)
            known_cols: Columns that are known at prediction time (if None, all feature_cols are considered known)
            unknown_cols: Columns that are unknown at prediction time (if None, all target_cols are considered unknown)
            weights: Name of weights column
            memory_efficient: Whether to use memory-efficient mode
            chunk_size: Chunk size for processing data (used in memory-efficient mode)
        """
        # Basic configuration
        self.file_paths = file_paths
        self.time_col = time_col
        self.weights = weights
        
        # Handle group columns (can be single column or multiple)
        if isinstance(group_cols, str):
            self.group_cols = [group_cols]
        else:
            self.group_cols = group_cols
            
        # Feature configuration
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.static_cols = static_cols or []
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        
        # Known/unknown columns configuration
        self.known_cols = known_cols or self.feature_cols.copy()
        self.unknown_cols = unknown_cols or self.target_cols.copy()
        
        # If num_cols not specified, infer from feature_cols and cat_cols
        if not self.num_cols:
            all_cols = self.feature_cols + self.target_cols + self.static_cols
            self.num_cols = [c for c in all_cols if c not in self.cat_cols]
        
        # Internal state
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        
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
        groups = chunk[self.group_cols].drop_duplicates()
        
        # Process each group in the chunk
        for _, group_row in groups.iterrows():
            # Create a key from the group columns' values
            group_key = tuple(group_row[self.group_cols].values)
            # If there's only one group column, use the value directly instead of a tuple
            if len(self.group_cols) == 1:
                group_key = group_key[0]
            
            # Create a file-specific group identifier
            file_group_key = (file_idx, group_key)
            file_groups.add(file_group_key)
            
            # Initialize group entry if not seen before
            if file_group_key not in self.group_info:
                self.group_info[file_group_key] = []
                self.lengths[file_group_key] = 0  # Initialize length counter for this group
            
            # Filter data for the current group
            group_mask = (chunk[self.group_cols] == group_row).all(axis=1)
            group_data = chunk[group_mask]
            
            # Store information about the group
            info = {
                'file_idx': file_idx,
                'file_path': file_path,
                'group_key': group_key,
                'file_group_key': file_group_key,
                'start_time': group_data[self.time_col].min(),
                'end_time': group_data[self.time_col].max(),
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
        Preload all data from files for faster access (wee only used when memory_efficient=False).
        """
        print("Preloading data into memory...")
        for file_idx, file_path in enumerate(self.file_paths):
            print(f"Loading file {file_idx + 1}/{len(self.file_paths)}: {file_path}")
            self.data_cache[file_idx] = pd.read_csv(file_path) #TODO sorting in D1
            
            # Encode categorical features
            for col in self.cat_cols:
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
            if not, use cache storage
            if yes, load from location. 
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
            
            # Filter for the group
            if isinstance(group_key, tuple):
                # For multi-column groups, check all columns
                mask = np.ones(len(file_data), dtype=bool)
                for col, val in zip(self.group_cols, group_key):
                    mask &= (file_data[col] == val)
            else:
                # For single-column groups, simple equality check
                mask = file_data[self.group_cols[0]] == group_key
            
            # If this file contains data for our group
            if mask.any():
                group_data = file_data[mask].copy()
            else:
                return pd.DataFrame()
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
                        for col, val in zip(self.group_cols, group_key):
                            mask &= (chunk[col] == val)
                    else:
                        # For single-column groups, simple equality check
                        mask = chunk[self.group_cols[0]] == group_key
                    
                    # If this chunk contains data for our group
                    if mask.any():
                        group_chunk = chunk[mask].copy()
                        
                        # Encode categorical features
                        for col in self.cat_cols:
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
                for col in self.cat_cols:
                    if col in file_data.columns:
                        file_data[col] = self.label_encoders[col].transform(
                            file_data[col].values.reshape(-1, 1)
                        ).flatten()
                
                # Filter for the group
                if isinstance(group_key, tuple):
                    # For multi-column groups, check all columns
                    mask = np.ones(len(file_data), dtype=bool)
                    for col, val in zip(self.group_cols, group_key):
                        mask &= (file_data[col] == val)
                else:
                    # For single-column groups, simple equality check
                    mask = file_data[self.group_cols[0]] == group_key
                
                # If this file contains data for our group
                if mask.any():
                    group_data = file_data[mask].copy()
                else:
                    return pd.DataFrame()
        
        # Sort by time
        group_data = group_data.sort_values(self.time_col)
        
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
        for col in self.cat_cols + self.group_cols:
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
        y = torch.tensor(group_data[self.target_cols].values, dtype=torch.float32)
        
        # Time values - keep as numpy array for datetime support
        t = group_data[self.time_col].values
        
        # Weights - use ones if not provided
        if self.weights is not None and self.weights in group_data.columns:
            w = torch.tensor(group_data[self.weights].values, dtype=torch.float32)
        else:
            w = torch.ones(len(group_data), dtype=torch.float32)
        
        # Static features - use empty tensor if none provided
        if self.static_cols and len(self.static_cols) > 0:
            # Take first row since static features are constant within a group
            static_values = []
            for col in self.static_cols:
                if col in group_data.columns:
                    # Convert to float if possible, otherwise use 0.0 as placeholder
                    try:
                        val = float(group_data[col].iloc[0])
                    except (ValueError, TypeError):
                        val = 0.0  # Default value for non-numeric static features
                    static_values.append(val)
            
            # Create tensor if we have values, otherwise empty tensor
            if static_values:
                st = torch.tensor(static_values, dtype=torch.float32)
            else:
                st = torch.tensor([], dtype=torch.float32)
        else:
            st = torch.tensor([], dtype=torch.float32)
        
        # Return all data as a dictionary
        return {
            'x': x,  # features
            'y': y,  # targets
            't': t,  # time indices (as numpy array)
            'w': w,  # weights
            'group_id': group_id,  # group identifier
            'st': st  # static features
        }
    
    def _prepare_metadata(self):
        """Prepare metadata about the dataset.""" #TODO: max number of classes in the data. (categoricAL)
        self.metadata = {
            "cols": {
                "y": self.target_cols,
                "x": self.feature_cols,
                "st": self.static_cols,
            },
            "col_type": {},
            "col_known": {},
            "weight": self.weights,
            "cat_index": []
        }
        
        # Mark categorical feature indices
        for i, c in enumerate(self.feature_cols):
            if c in self.cat_cols:
                self.metadata['cat_index'].append(i)
        
        # Set column types and known status #TODO: user should define the known and unknown columns
        all_cols = self.target_cols + self.feature_cols  + self.static_cols
        for col in all_cols:
            self.metadata["col_type"][col] = "C" if col in self.cat_cols else "F"
            # All columns are considered unknown for future values by default
            self.metadata["col_known"][col] = "U"
        
        # Add known/unknown column information
        self.metadata['known_cols'] = self.known_cols
        self.metadata['unknown_cols'] = self.unknown_cols 
        #TODOdistinguish between cat an dnum known and unknown
        
        # Add encoders to metadata
        self.metadata['encoders'] = self.label_encoders
    
    def get_metadata(self) -> Dict:
        """
        Return metadata about the dataset.
        
        Returns:
            Dictionary containing metadata about columns and their properties
        """
        return self.metadata


class TSDataModule(pl.LightningDataModule):
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
        min_valid_length: Optional[int] = None,
        split_method: str = 'percentage',
        split_config: Optional[tuple] = None,
        num_workers: int = 0,
        sampler: Optional[Sampler] = None,
        memory_efficient: bool = False,
        known_cols: Optional[List[str]] = None,
        unknown_cols: Optional[List[str]] = None
    ):
        """
        Initialize the TSDataModule.
        
        Args:
            d1_dataset: The D1 dataset instance (MultiSourceTSDataSet)
            past_len: Number of past time steps for input
            future_len: Number of future time steps for prediction
            batch_size: Batch size for DataLoaders
            min_valid_length: Minimum number of valid points required in a window
            split_method: Method for splitting data ('percentage' or 'group')
            split_config: Configuration for the split
            num_workers: Number of workers for DataLoader
            sampler: Optional custom sampler for the DataLoader
            memory_efficient: Whether to use memory-efficient mode
            known_cols: Columns that are known at prediction time (overrides D1 dataset settings)
            unknown_cols: Columns that are unknown at prediction time (overrides D1 dataset settings)
        """
        super().__init__()
        self.d1_dataset = d1_dataset
        self.past_len = past_len
        self.future_len = future_len
        self.batch_size = batch_size
        self.min_valid_length = min_valid_length or past_len
        self.split_method = split_method
        self.split_config = split_config
        self.num_workers = num_workers
        self.sampler = sampler
        self.memory_efficient = memory_efficient
        
        # Fixed number for max cached groups
        self.max_cached_groups = 50
        
        # Store reference to D1 dataset metadata
        self.metadata = d1_dataset.metadata.copy()
        
        # Set feature and target columns from D1 dataset
        self.feature_cols = d1_dataset.feature_cols
        self.target_cols = d1_dataset.target_cols
        self.static_cols = d1_dataset.static_cols
        
        # Override known/unknown columns if provided
        self.known_cols = known_cols if known_cols is not None else d1_dataset.known_cols
        self.unknown_cols = unknown_cols if unknown_cols is not None else d1_dataset.unknown_cols
        
        # Update metadata with known/unknown columns
        self.metadata['known_cols'] = self.known_cols
        self.metadata['unknown_cols'] = self.unknown_cols
        
        # For tracking loaded data with simple FIFO caching
        self.loaded_groups = {}
        self.group_load_order = []
        
        # Default split configuration based on method
        if split_config is None:
            if split_method == 'percentage':
                self.split_config = (0.7, 0.15, 0.15)  # Default: 70% train, 15% val, 15% test
            else:
                raise ValueError("For 'group' split method, split_config must be provided")
                
        # Initialize the module
        self._initialize()
        
        # Add max classes information to metadata
        self._add_max_classes_to_metadata()
    
    def _initialize(self):
        """Initialize the dataset by computing valid indices and mappings."""
        print("Precomputing valid indices and mappings...")
        self.valid_indices = self._compute_valid_indices()
        self.mapping = self._create_global_mapping()
        self.length = len(self.mapping)
        
        # Create splits if not already done
        if not hasattr(self, 'train_indices') or not self.train_indices:
            if self.split_config is not None:
                print(f"Creating {self.split_method} splits with config: {self.split_config}")
                
                # Create splits with the new config
                self.train_indices, self.val_indices, self.test_indices = self._create_splits(self.split_config)
                print(f"Split statistics: Train: {len(self.train_indices)}, Validation: {len(self.val_indices)}, Test: {len(self.test_indices)}")
            else:
                # Default to all indices as training
                self.train_indices = list(range(self.length))
                self.val_indices = []
                self.test_indices = []
        
        # Create subset datasets for train/val/test using the indices
        if hasattr(self, 'train_indices') and self.train_indices:
            self.train_dataset = TimeSeriesSubset(self, self.train_indices)
        if hasattr(self, 'val_indices') and self.val_indices:
            self.val_dataset = TimeSeriesSubset(self, self.val_indices)
        if hasattr(self, 'test_indices') and self.test_indices:
            self.test_dataset = TimeSeriesSubset(self, self.test_indices)
    
    def _compute_valid_indices(self):
        """
        Compute valid indices for all groups in the dataset.
        
        This method ensures that windows are valid based on:
        1. Having enough valid points in the past window
        2. Having at least one valid point in the future window
        
        Optimizations:
        - Vectorized operations for NaN checks
        - Early termination for invalid regions
        - Efficient masking for bulk validation
        
        Returns:
            Dictionary mapping group indices to lists of valid indices
        """
        valid_indices = {} 
        device = torch.device('cpu')  # Use CPU for consistency across environments
        
        for i in range(len(self.d1_dataset)):
            # Load group data from D1 dataset
            group_data = self.d1_dataset[i]
            
            # Fetch feature and target tensors
            features = group_data.get('x', torch.tensor([]))
            targets = group_data.get('y', torch.tensor([]))
            
            # Get the time series length
            ts_length = len(features)
            
            # Skip if time series is too short
            if ts_length < (self.past_len + self.future_len):
                valid_indices[i] = []
                continue
            
            # Pre-compute all possible window start indices
            all_indices = list(range(ts_length - (self.past_len + self.future_len) + 1))
            
            # Create validity masks for features and targets
            feature_is_valid = ~torch.isnan(features) if features.numel() > 0 else torch.ones((ts_length, 1), dtype=torch.bool)
            target_is_valid = ~torch.isnan(targets) if targets.numel() > 0 else torch.ones((ts_length, 1), dtype=torch.bool)
            
            # If features or targets are multi-dimensional, check if any dimension has NaN
            if feature_is_valid.dim() > 1:
                feature_is_valid = feature_is_valid.all(dim=1)
            if target_is_valid.dim() > 1:
                target_is_valid = target_is_valid.all(dim=1)
            
            # Combined mask for both features and targets
            combined_mask = feature_is_valid & target_is_valid
            
            # Vectorized approach to find valid windows
            group_valid_indices = []
            
            # Process each potential window
            t = 0
            while t < len(all_indices):
                idx = all_indices[t]
                end_past = idx + self.past_len
                end_future = end_past + self.future_len
                
                # Check past validity - count valid points in past window
                past_valid_count = combined_mask[idx:end_past].sum().item()
                past_valid = past_valid_count >= self.min_valid_length
                
                # Early termination - if past isn't valid, skip this window
                if not past_valid:
                    # Find the next valid point after the current position
                    next_valid_idx = None
                    for j in range(idx + 1, min(ts_length, idx + self.past_len * 2)):
                        if combined_mask[j]:
                            next_valid_idx = j
                            break
                    
                    if next_valid_idx is not None:
                        # Skip to a position where this valid point would be in the window
                        # but not at the end (to maximize valid points in window)
                        skip_to = max(t + 1, next_valid_idx - self.past_len + 1)
                        t = skip_to
                    else:
                        # No valid points ahead, skip to the end
                        t = len(all_indices)
                    continue
                
                # Check future validity - at least one point must be valid
                future_valid = combined_mask[end_past:end_future].any().item()
                
                if past_valid and future_valid:
                    group_valid_indices.append(idx)
                
                t += 1
            
            valid_indices[i] = group_valid_indices
            
        # Print statistics
        total_valid = sum(len(indices) for indices in valid_indices.values())
        groups_with_valid = sum(1 for indices in valid_indices.values() if len(indices) > 0)
        print(f"Found {total_valid} valid windows across {groups_with_valid} groups")
            
        return valid_indices
    
    def _create_global_mapping(self):
        """
        Create a global mapping from index to (group_idx, local_idx).
        
        This allows O(1) lookup of samples by global index.
        
        Returns:
            List of (group_idx, local_idx) tuples
        """
        mapping = []
        
        # Track statistics for reporting
        total_valid = 0
        groups_with_valid = 0
        
        for group_idx, local_indices in self.valid_indices.items():
            if local_indices:  # Only add groups with valid indices
                for start_idx in local_indices:
                    mapping.append((group_idx, start_idx))
                total_valid += len(local_indices)
                groups_with_valid += 1
        
        print(f"Created global mapping with {len(mapping)} windows from {groups_with_valid} groups")
        return mapping
    
    def _get_group_data(self, group_idx):
        """
        Get data for a specific group, using cache if available.
        
        This method:
        1. Checks if the group data is already in the cache
        2. If not, loads it from the D1 dataset
        
        Args:
            group_idx: Index of the group to retrieve
            
        Returns:
            Dictionary containing the group data
        """
        # Check if the group is already loaded
        if group_idx in self.loaded_groups:
            return self.loaded_groups[group_idx]
            
        # Load the group data from D1 dataset
        group_data = self.d1_dataset[group_idx]
        
        # Simple FIFO caching
        if len(self.loaded_groups) >= self.max_cached_groups:
            # Remove oldest group if cache is full
            if self.group_load_order:
                oldest_group = self.group_load_order.pop(0)
                if oldest_group in self.loaded_groups:
                    del self.loaded_groups[oldest_group]
        
        # Add to cache and update load order
        self.loaded_groups[group_idx] = group_data
        self.group_load_order.append(group_idx)
            
        return group_data
        
    def _add_max_classes_to_metadata(self):
        """
        Add information about maximum number of classes for categorical features.
        This is useful for model architecture decisions.
        """
        if 'max_classes' not in self.metadata:
            self.metadata['max_classes'] = {}
            
        # Get label encoders from D1 dataset
        for col, encoder in self.d1_dataset.label_encoders.items():
            # Number of classes is the number of unique values seen by the encoder
            num_classes = len(encoder.classes_)
            self.metadata['max_classes'][col] = num_classes
            
        # Ensure known/unknown columns are properly reflected in metadata
        self.metadata['known_cols'] = self.known_cols
        self.metadata['unknown_cols'] = self.unknown_cols
    
    def _create_splits(self, split_config):
        """
        Create train/validation/test splits based on the specified configuration.
        
        Args:
            split_config: Configuration for splits:
                          - For 'percentage' method: (train%, val%, test%)
                          - For 'group' method: (train_groups, val_groups, test_groups)
                          
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        if self.split_method == 'percentage':
            # Percentage-based split (temporal or random)
            train_pct, val_pct, test_pct = split_config
            total_samples = len(self.mapping)
            
            # Normalize percentages if they don't sum to 1
            total_pct = train_pct + val_pct + test_pct
            if abs(total_pct - 1.0) > 1e-6:
                train_pct /= total_pct
                val_pct /= total_pct
                test_pct /= total_pct
            
            # Calculate number of samples for each split
            train_size = int(total_samples * train_pct)
            val_size = int(total_samples * val_pct)
            test_size = total_samples - train_size - val_size
            
            # For temporal splits, sort by time
            all_indices = list(range(total_samples))
            
            # Try to sort by time if available #TODO: sort in D1 by group and then time. 
            try:
                # Collect time values for each window
                time_values = []
                for idx in all_indices:
                    group_idx, local_idx = self.mapping[idx]
                    group_data = self.d1_dataset[group_idx]
                    # Use the first time point of each window
                    time_point = group_data['t'][local_idx]
                    # Convert to numeric value for sorting
                    if isinstance(time_point, (str, np.datetime64)):
                        # Convert to timestamp (seconds since epoch)
                        time_numeric = pd.to_datetime(time_point).timestamp()
                    else:
                        time_numeric = float(time_point)
                    time_values.append(time_numeric)
                
                # Sort indices by time
                sorted_indices = [idx for _, idx in sorted(zip(time_values, all_indices))]
                
                # Split into train/val/test
                train_indices = sorted_indices[:train_size]
                val_indices = sorted_indices[train_size:train_size + val_size]
                test_indices = sorted_indices[train_size + val_size:]
                
                print(f"Temporal percentage-based split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)} samples")
                
            except (TypeError, ValueError) as e:
                # Fallback to random split if time-based sorting fails
                print(f"Warning: Could not sort by time ({str(e)}), using random split instead")
                random.shuffle(all_indices)
                train_indices = all_indices[:train_size]
                val_indices = all_indices[train_size:train_size + val_size]
                test_indices = all_indices[train_size + val_size:]
                
                print(f"Random percentage-based split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)} samples")
                
        elif self.split_method == 'group':
            # Group-based split
            train_groups, val_groups, test_groups = split_config
            
            # Convert to sets for faster lookup
            train_groups_set = set(train_groups)
            val_groups_set = set(val_groups)
            test_groups_set = set(test_groups)
            
            # Assign indices to splits based on group membership
            train_indices = []
            val_indices = []
            test_indices = []
            
            for idx, (group_idx, _) in enumerate(self.mapping):
                group_id = self.d1_dataset[group_idx]['group_id']
                
                if group_id in train_groups_set:
                    train_indices.append(idx)
                elif group_id in val_groups_set:
                    val_indices.append(idx)
                elif group_id in test_groups_set:
                    test_indices.append(idx)
                else:
                    # Default to train if not specified
                    train_indices.append(idx)
            
            print(f"Group-based split - Train: {len(train_indices)} (from {len(train_groups)} groups), "
                  f"Val: {len(val_indices)} (from {len(val_groups)} groups), "
                  f"Test: {len(test_indices)} (from {len(test_groups)} groups) samples")
            
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")
            
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
        Get a time series window by global index.
        
        This method:
        1. Maps the global index to a specific group and local index
        2. Extracts the window from the group data
        3. Returns the window in a format suitable for model training
        
        Args:
            idx: Global index of the window to retrieve
            
        Returns:
            Dictionary containing:
            - past_features: Tensor of past features
            - past_time: Array of past time points
            - future_targets: Tensor of future targets
            - future_time: Array of future time points
            - group_id: Group identifier
            - static: Static features tensor
        """
        # Map global index to group and local index
        group_idx, local_idx = self.mapping[idx]
        
        # Get the group data
        group_data = self._get_group_data(group_idx)
        
        # Get the start and end indices for the window
        start_idx = local_idx
        past_end_idx = start_idx + self.past_len
        future_end_idx = past_end_idx + self.future_len
        
        # Extract past and future windows
        past_features = group_data['x'][start_idx:past_end_idx]
        past_time = group_data['t'][start_idx:past_end_idx]
        future_targets = group_data['y'][past_end_idx:future_end_idx]
        future_time = group_data['t'][past_end_idx:future_end_idx]
        
        # Get static features
        static = group_data.get('st', torch.tensor([]))
        
        # Return the window as a dictionary
        return {
            'past_features': past_features,
            'past_time': past_time,
            'future_targets': future_targets,
            'future_time': future_time,
            'group_id': group_data['group_id'],
            'static': static
        }
    
    def setup(self, stage=None):
        """
        Prepare data for the given stage.
        
        Args:
            stage: Either 'fit' or 'test'
        """
        # If we haven't precomputed valid indices yet, do it now
        if not hasattr(self, 'valid_indices') or self.valid_indices is None:
            print("Computing valid indices...")
            self.valid_indices = self._compute_valid_indices()
            self.mapping = self._create_global_mapping()
            self.length = len(self.mapping)
        
        # Create splits if not already done
        if not hasattr(self, 'train_indices') or not self.train_indices:
            if self.split_config is not None:
                print(f"Creating {self.split_method} splits with config: {self.split_config}")
                
                # Create splits with the new config
                self.train_indices, self.val_indices, self.test_indices = self._create_splits(self.split_config)
                print(f"Split statistics: Train: {len(self.train_indices)}, Validation: {len(self.val_indices)}, Test: {len(self.test_indices)}")
            else:
                # Default to all indices as training
                self.train_indices = list(range(self.length))
                self.val_indices = []
                self.test_indices = []
        
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
                num_workers=self.num_workers,
                collate_fn=custom_collate_fn
            )
        else:
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=custom_collate_fn
            )
    
    def val_dataloader(self):
        """Return a DataLoader for validation."""
        if len(self.val_dataset) == 0:
            return None
            
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn
        )
    
    def test_dataloader(self):
        """Return a DataLoader for testing."""
        if len(self.test_dataset) == 0:
            return None
            
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn
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


def custom_collate_fn(batch):
    """
    Custom collate function for the DataLoader to handle mixed data types.
    Handles static features that may be objects or other non-tensor types.
    """
    elem = batch[0]
    result = {}
    
    # Process each key in the batch
    for key in elem:
        if key in ['st', 'group_id', 't', 'v']:  # Special handling for non-tensor data
            # Store as lists
            result[key] = [sample[key] for sample in batch]
        else:  # Default handling for tensors
            # For tensors, we can stack them
            try:
                result[key] = torch.stack([sample[key] for sample in batch])
            except:
                # If stacking fails, just store as a list
                result[key] = [sample[key] for sample in batch]
    
    return result


def generate_example_csv(output_dir, num_groups=5, samples_per_group=100, num_features=3, num_targets=1):
    """
    Generate example CSV files for testing the time series dataset.
    
    Args:
        output_dir: Directory to save CSV files
        num_groups: Number of groups to generate
        samples_per_group: Number of samples per group
        num_features: Number of features to generate
        num_targets: Number of targets to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate two CSV files
    for file_idx in range(2):
        data = []
        
        # Generate data for each group
        for group_idx in range(num_groups):
            # Determine which file gets which groups
            if (group_idx % 2 == 0 and file_idx == 0) or (group_idx % 2 == 1 and file_idx == 1):
                # Generate time series for this group
                for t in range(samples_per_group):
                    row = {
                        'group': f'group_{group_idx}',
                        'time': t,
                    }
                    
                    # Generate features with some NaNs
                    for f in range(num_features):
                        # Add some randomness and occasional NaNs
                        if np.random.random() < 0.05:  # 5% chance of NaN
                            row[f'feature_{f}'] = np.nan
                        else:
                            row[f'feature_{f}'] = np.sin(t/10 + group_idx) + np.random.normal(0, 0.1)
                    
                    # Generate targets with some NaNs
                    for tgt in range(num_targets):
                        if np.random.random() < 0.05:  # 5% chance of NaN
                            row[f'target_{tgt}'] = np.nan
                        else:
                            # Target is a function of features with noise
                            row[f'target_{tgt}'] = np.cos(t/10 + group_idx) + np.random.normal(0, 0.1)
                    
                    # Add a categorical feature
                    row['cat_feature'] = f'cat_{np.random.randint(0, 3)}'
                    
                    data.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(output_dir, f'example_data_{file_idx}.csv'), index=False)
        print(f"Generated {len(df)} rows in file {file_idx}")

# Example usage
if __name__ == "__main__":
    # Generate example data
    example_dir = "example_data"
    generate_example_csv(example_dir)
    
    # Create D1 dataset with explicit known/unknown columns
    file_paths = [os.path.join(example_dir, f'example_data_{i}.csv') for i in range(2)]
    
    # Define column sets
    feature_cols = [f'feature_{i}' for i in range(3)]
    target_cols = [f'target_{0}']
    
    # Explicitly define which features are known at prediction time
    # In this example, we're saying feature_0 and feature_1 are known,
    # but feature_2 is unknown (perhaps it's only available during training)
    known_cols = [f'feature_{i}' for i in range(2)]  # Only first two features are known
    unknown_cols = [f'feature_2'] + target_cols  # feature_2 and target are unknown
    
    d1_dataset = MultiSourceTSDataSet(
        file_paths=file_paths,
        group_cols='group',
        time_col='time',
        feature_cols=feature_cols,
        target_cols=target_cols,
        cat_cols=['cat_feature'],
        known_cols=known_cols,
        unknown_cols=unknown_cols,
        memory_efficient=False,
        chunk_size=1000
    )
    
    print(f"D1 Dataset Stats:")
    print(f"Number of groups: {len(d1_dataset)}")
    print(f"Feature columns: {d1_dataset.feature_cols}")
    print(f"Known columns: {d1_dataset.known_cols}")
    print(f"Unknown columns: {d1_dataset.unknown_cols}")
    print(f"Metadata: {d1_dataset.metadata}")
    
    # Create D2 module with percentage split
    # Here we're overriding the known/unknown columns from D1
    d2_module = TSDataModule(
        d1_dataset=d1_dataset,
        past_len=10,
        future_len=5,
        batch_size=32,
        min_valid_length=8,
        split_method='percentage',
        split_config=(0.7, 0.15, 0.15),
        memory_efficient=False,
        num_workers=0,
        sampler=None,
        # Override known/unknown columns from D1
        known_cols=['feature_0'],  # Now only feature_0 is known
        unknown_cols=['feature_1', 'feature_2', 'target_0']  # feature_1 is now unknown too
    )
    
    print(f"D2 Module Stats (Percentage Split):")
    print(f"Total windows: {d2_module.length}")
    print(f"Train windows: {len(d2_module.train_indices)}")
    print(f"Val windows: {len(d2_module.val_indices)}")
    print(f"Test windows: {len(d2_module.test_indices)}")
    print(f"Known columns: {d2_module.known_cols}")
    print(f"Unknown columns: {d2_module.unknown_cols}")
    print(f"Metadata: {d2_module.metadata}")
    
    # Test DataLoaders
    train_loader = d2_module.train_dataloader()
    val_loader = d2_module.val_dataloader()
    test_loader = d2_module.test_dataloader()
    
    # Fetch a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Past features shape: {batch['past_features'].shape}")
    print(f"Future targets shape: {batch['future_targets'].shape}")
    
    # Create D2 module with group split
    # First, get all group indices
    all_groups = list(range(len(d1_dataset)))
    
    # Split groups into train/val/test
    np.random.shuffle(all_groups)
    train_groups = all_groups[:int(0.7 * len(all_groups))]
    val_groups = all_groups[int(0.7 * len(all_groups)):int(0.85 * len(all_groups))]
    test_groups = all_groups[int(0.85 * len(all_groups)):]
    
    # Using the D1 dataset's known/unknown columns (not overriding)
    d2_module_group = TSDataModule(
        d1_dataset=d1_dataset,
        past_len=10,
        future_len=5,
        batch_size=32,
        min_valid_length=8,
        split_method='group',
        split_config=(train_groups, val_groups, test_groups),
        memory_efficient=False,
        num_workers=0,
        sampler=None
        # Not specifying known_cols/unknown_cols, so it will use D1's settings
    )
    
    print(f"D2 Module Stats (Group Split):")
    print(f"Total windows: {d2_module_group.length}")
    print(f"Train windows: {len(d2_module_group.train_indices)}")
    print(f"Val windows: {len(d2_module_group.val_indices)}")
    print(f"Test windows: {len(d2_module_group.test_indices)}")
    print(f"Known columns: {d2_module_group.known_cols}")
    print(f"Unknown columns: {d2_module_group.unknown_cols}")