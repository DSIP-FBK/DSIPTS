"""
Time Series D1 Layer Module

This module provides the D1 layer for time series data handling:
- MultiSourceTSDataSet: Handles raw data from multiple CSV files

Key Features:
- Supports multiple CSV files with different groups
- Handles regular time intervals
- Efficiently processes data in chunks for memory-efficient operation
- Handles categorical encoding and normalization
- Preserves NaN values for D2 layer to handle
"""

import os
import torch
from torch.utils.data import Dataset
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


def extend_time_df(df, time_col, freq, group_cols=None, max_length=None, fill_value=None):
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
        Preload all data from files into memory for faster access.
        """
        print("Preloading data into memory...")
        for file_idx, file_path in enumerate(self.file_paths):
            print(f"Loading file {file_idx + 1}/{len(self.file_paths)}: {file_path}")
            # Load the data
            file_data = pd.read_csv(file_path)
            
            # Encode categorical features
            for col in self.cat_cols:
                if col in file_data.columns:
                    file_data[col] = self.label_encoders[col].transform(
                        file_data[col].values.reshape(-1, 1)
                    ).flatten()
            
            # Sort by group columns and then by time
            sort_cols = self.group_cols + [self.time_col]
            file_data = file_data.sort_values(sort_cols)
            
            # Store in cache
            self.data_cache[file_idx] = file_data
            
        print(f"Preloaded {len(self.file_paths)} files")
        
        # Estimate memory usage
        if self.data_cache:
            first_file = next(iter(self.data_cache.values()))
            # Calculate approximate size in MB
            size_mb = first_file.memory_usage(deep=True).sum() / (1024 * 1024)
            total_mb = sum(df.memory_usage(deep=True).sum() for df in self.data_cache.values()) / (1024 * 1024)
            print(f"Estimated memory usage: {total_mb:.2f} MB")
        
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
        """Prepare metadata about the dataset."""
        self.metadata = {
            "cols": {
                "y": self.target_cols,
                "x": self.feature_cols,
                "st": self.static_cols
            },
            "col_type": {},
            "col_known": {},
            "weight": self.weights,
            "cat_index": []
        }
        
        # Add column indices
        for i, c in enumerate(self.feature_cols):
            if c in self.cat_cols:
                self.metadata['cat_index'].append(i)
        
        # Set column types and known status
        all_cols = self.target_cols + self.feature_cols + self.static_cols
        for col in all_cols:
            self.metadata["col_type"][col] = "C" if col in self.cat_cols else "F"
            # All columns are considered unknown for future values by default
            self.metadata["col_known"][col] = "U"
        
        # Add known/unknown column information
        self.metadata['known_cols'] = self.known_cols
        self.metadata['unknown_cols'] = self.unknown_cols
        
        # Distinguish between categorical and numerical known/unknown columns
        self.metadata['known_cat_cols'] = [col for col in self.known_cols if col in self.cat_cols]
        self.metadata['known_num_cols'] = [col for col in self.known_cols if col not in self.cat_cols]
        self.metadata['unknown_cat_cols'] = [col for col in self.unknown_cols if col in self.cat_cols]
        self.metadata['unknown_num_cols'] = [col for col in self.unknown_cols if col not in self.cat_cols]
        
        # Add max classes information for categorical columns
        self.metadata['max_classes'] = {}
        for col, encoder in self.label_encoders.items():
            # Number of classes is the number of unique values seen by the encoder
            num_classes = len(encoder.categories_[0])
            self.metadata['max_classes'][col] = num_classes
        
        # Add encoders to metadata
        self.metadata['encoders'] = self.label_encoders
    
    def get_metadata(self) -> Dict:
        """
        Return metadata about the dataset.
        
        Returns:
            Dictionary containing metadata about columns and their properties
        """
        return self.metadata
    
    def _load_group_data(self, file_group_key):
        """
        Load data for a specific file-group combination.
        
        This method:
        1. Gets the file containing data for the requested group
        2. Loads from cache or from disk based on memory_efficient setting
        3. Applies encoding to categorical features if needed
        4. Sorts data by time column
        5. Regularizes time series using extend_time_df
        6. Preserves NaN values for valid index computation in D2 layer
        
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
            # Use preloaded data from cache (already sorted by group and time)
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
        
        # Sort by time (since we're dealing with a single group, no need to sort by group)
        group_data = group_data.sort_values(self.time_col)
        
        # Regularize time series (fill gaps with NaN values)
        # First, try to detect the frequency
        try:
            # Convert time column to datetime if it's not numeric
            time_col_data = group_data[self.time_col]
            if not pd.api.types.is_numeric_dtype(time_col_data):
                time_col_data = pd.to_datetime(time_col_data)
                
            # Calculate differences between consecutive time points
            if len(time_col_data) > 1:
                if pd.api.types.is_datetime64_dtype(time_col_data):
                    # For datetime, calculate timedeltas
                    time_diffs = time_col_data.diff().dropna()
                    # Get the most common difference (mode)
                    freq = time_diffs.mode().iloc[0]
                else:
                    # For numeric time, calculate differences
                    time_diffs = np.diff(time_col_data)
                    # Get the most common difference (mode)
                    freq = pd.Series(time_diffs).mode().iloc[0]
                    
                # Use extend_time_df to regularize the time series
                group_data = extend_time_df(
                    df=group_data,
                    time_col=self.time_col,
                    freq=freq,
                    fill_value=np.nan
                )
        except Exception as e:
            # If regularization fails, log the error but continue with original data
            print(f"Warning: Time series regularization failed for group {group_key}: {str(e)}")
        
        return group_data
