import numpy as np
import pandas as pd
import zarr
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, List, Dict, Tuple, Any
import os
import glob
from datetime import datetime, timedelta
from sklearn.preprocessing import OrdinalEncoder
import dask.dataframe as dd
import dask.array as da
import itertools
import logging

class MultiFileZarrTSDataset(Dataset):
    """PyTorch Dataset for loading time series data from multiple Zarr stores.
    
    This class handles large datasets split across multiple files, with different
    groups in each file. It processes data in chunks and maintains a global label
    encoder that can handle all categories across files.
    
    Parameters
    ----------
    file_paths : List[str]
        List of paths to zarr stores or directories containing zarr stores
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
        Dictionary of pre-fitted label encoders for categorical variables
    chunk_size : Optional[Dict[str, int]]
        Size of chunks for each dimension, e.g {'time':100, 'group':10}
    """
    
    def __init__(
        self,
        file_paths: List[str],
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
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing MultiFileZarrTSDataset")
        
        # Validate file paths
        if not file_paths:
            self.logger.error("No file paths provided")
            raise ValueError("file_paths cannot be empty")
        
        # Store parameters
        self.file_paths = file_paths
        self.time = time
        self.target = _coerce_to_list(target)
        self.group = _coerce_to_list(group)
        self.num = _coerce_to_list(num)
        self.cat = _coerce_to_list(cat)
        self.known = _coerce_to_list(known)
        self.unknown = _coerce_to_list(unknown)
        self.static = _coerce_to_list(static)
        self.chunk_size = chunk_size if chunk_size else {}
        
        # Initialize zarr stores
        self.zarr_stores = []
        self.file_info = []
        
        # Initialize label encoders
        if label_encoders is None:
            self.label_encoders = {}
            for c in self.cat + self.group:
                self.label_encoders[c] = OrdinalEncoder()
        else:
            self.label_encoders = label_encoders
        
        # Process each file
        self._process_files()
        
        # Setup chunking
        self._setup_chunks()
        
    def _process_files(self):
        """Process each file, extract group information, and update label encoders."""
        self.logger.info("Processing files")
        self.group_info = {}  # Maps group values to file indices
        self.total_length = 0
        
        for file_idx, file_path in enumerate(self.file_paths):
            self.logger.info(f"Processing file {file_idx + 1}/{len(self.file_paths)}: {file_path}")
            
            # Open zarr store
            try:
                zarr_store = zarr.open(file_path, mode='r')
                self.zarr_stores.append(zarr_store)
            except Exception as e:
                self.logger.error(f"Failed to open Zarr store at {file_path}: {e}")
                raise
            
            # Create a small sample to analyze groups and update encoders
            sample_data = self._get_sample_from_store(zarr_store)
            
            # Extract group information
            if self.group:
                groups_in_file = sample_data[self.group].drop_duplicates()
                
                # For each group in this file
                for _, group_row in groups_in_file.iterrows():
                    group_key = tuple(group_row[self.group].values)
                    
                    # Store mapping from group to file
                    if group_key not in self.group_info:
                        self.group_info[group_key] = []
                    self.group_info[group_key].append(file_idx)
                    
                    # Get time range for this group in this file
                    group_mask = True
                    for g_idx, g_col in enumerate(self.group):
                        group_mask = group_mask & (sample_data[g_col] == group_key[g_idx])
                    
                    group_data = sample_data[group_mask]
                    min_time = group_data[self.time].min()
                    max_time = group_data[self.time].max()
                    
                    # Store file info
                    self.file_info.append({
                        'file_idx': file_idx,
                        'group_key': group_key,
                        'min_time': min_time,
                        'max_time': max_time,
                        'length': len(group_data)
                    })
                    
                    self.total_length += len(group_data)
            else:
                # No groups, just store file info
                min_time = sample_data[self.time].min()
                max_time = sample_data[self.time].max()
                
                self.file_info.append({
                    'file_idx': file_idx,
                    'group_key': None,
                    'min_time': min_time,
                    'max_time': max_time,
                    'length': len(sample_data)
                })
                
                self.total_length += len(sample_data)
            
            # Update label encoders with categories from this file
            self._update_encoders(sample_data)
        
        self.logger.info(f"Finished processing {len(self.file_paths)} files")
    
    def _get_sample_from_store(self, zarr_store, sample_size=1000):
        """Get a sample from a zarr store for analysis."""
        # Create a dictionary to hold the data
        data_dict = {}
        
        # Get all array keys from the store
        array_keys = list(zarr_store.array_keys())
        
        # Get a sample from each array
        for key in array_keys:
            array = zarr_store[key]
            # Take first sample_size elements or all if fewer
            sample = array[:min(sample_size, array.shape[0])]
            data_dict[key] = sample
        
        # Convert to pandas DataFrame
        return pd.DataFrame(data_dict)
    
    def _update_encoders(self, data):
        """Update label encoders with new categories from the data."""
        for col in self.cat + self.group:
            if col in data.columns:
                # Get unique values
                unique_values = data[col].dropna().unique().reshape(-1, 1)
                
                if len(unique_values) > 0:
                    # Check if encoder exists
                    if col not in self.label_encoders:
                        self.label_encoders[col] = OrdinalEncoder()
                        self.label_encoders[col].fit(unique_values)
                    else:
                        # Get current categories
                        current_categories = self.label_encoders[col].categories_[0]
                        
                        # Check if there are new categories
                        new_categories = np.setdiff1d(unique_values.flatten(), current_categories)
                        
                        if len(new_categories) > 0:
                            # Create new encoder with combined categories
                            combined_categories = np.concatenate([current_categories, new_categories])
                            new_encoder = OrdinalEncoder(categories=[combined_categories])
                            new_encoder.fit(combined_categories.reshape(-1, 1))
                            self.label_encoders[col] = new_encoder
    
    def _setup_chunks(self):
        """Setup chunk information for iteration."""
        if not self.chunk_size:
            self.n_chunks = len(self.file_info)
            return
        
        # Calculate total chunks based on file_info and chunk_size
        self.n_chunks = 0
        self.chunk_map = []  # Maps chunk_idx to (file_info_idx, local_chunk_idx)
        
        for info_idx, info in enumerate(self.file_info):
            # Calculate chunks for this file and group
            chunks_in_file = 1
            for dim, size in self.chunk_size.items():
                if dim == self.time:
                    # Calculate chunks based on time range
                    time_range = info['max_time'] - info['min_time']
                    if isinstance(time_range, timedelta):
                        # For datetime
                        chunks_in_time = int(np.ceil(time_range.total_seconds() / 
                                                    pd.Timedelta(seconds=size).total_seconds()))
                    else:
                        # For numeric
                        chunks_in_time = int(np.ceil(time_range / size))
                    chunks_in_file *= max(1, chunks_in_time)
            
            # Store mapping
            for local_chunk_idx in range(chunks_in_file):
                self.chunk_map.append((info_idx, local_chunk_idx))
            
            self.n_chunks += chunks_in_file
    
    def _get_chunk(self, chunk_idx):
        """Get a chunk of data based on chunk index."""
        if chunk_idx >= len(self.chunk_map):
            raise IndexError(f"Chunk index {chunk_idx} out of range")
        
        info_idx, local_chunk_idx = self.chunk_map[chunk_idx]
        file_info = self.file_info[info_idx]
        file_idx = file_info['file_idx']
        zarr_store = self.zarr_stores[file_idx]
        
        # Create a dictionary to hold the data
        data_dict = {}
        
        # Get all array keys from the store
        array_keys = list(zarr_store.array_keys())
        
        # Calculate chunk boundaries
        if self.chunk_size and self.time in self.chunk_size:
            time_chunk_size = self.chunk_size[self.time]
            start_time = file_info['min_time'] + local_chunk_idx * time_chunk_size
            end_time = min(start_time + time_chunk_size, file_info['max_time'])
            
            # Create dask arrays for each column
            for key in array_keys:
                data_dict[key] = da.from_zarr(zarr_store[key])
            
            # Convert to dask dataframe
            ddf = dd.from_dask_array(
                da.stack([data_dict[key] for key in array_keys], axis=1),
                columns=array_keys
            )
            
            # Filter by time range
            filtered_ddf = ddf[(ddf[self.time] >= start_time) & (ddf[self.time] < end_time)]
            
            # Filter by group if applicable
            if self.group and file_info['group_key'] is not None:
                for g_idx, g_col in enumerate(self.group):
                    filtered_ddf = filtered_ddf[filtered_ddf[g_col] == file_info['group_key'][g_idx]]
            
            # Compute the result
            chunk_data = filtered_ddf.compute()
        else:
            # No chunking by time, get all data for this group
            # Create pandas dataframe directly
            data_dict = {}
            for key in array_keys:
                data_dict[key] = zarr_store[key][:]
            
            chunk_data = pd.DataFrame(data_dict)
            
            # Filter by group if applicable
            if self.group and file_info['group_key'] is not None:
                for g_idx, g_col in enumerate(self.group):
                    chunk_data = chunk_data[chunk_data[g_col] == file_info['group_key'][g_idx]]
        
        # Apply label encoders
        for col in self.cat + self.group:
            if col in chunk_data.columns:
                chunk_data[col] = self.label_encoders[col].transform(
                    chunk_data[col].values.reshape(-1, 1)
                ).flatten()
        
        return chunk_data
    
    def __len__(self):
        """Return the total number of items in the dataset."""
        return self.total_length
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        # Find which chunk this index belongs to
        chunk_idx = 0
        local_idx = idx
        cumulative_length = 0
        
        for i, info in enumerate(self.file_info):
            if idx < cumulative_length + info['length']:
                chunk_idx = i
                local_idx = idx - cumulative_length
                break
            cumulative_length += info['length']
        
        # Get chunk data
        chunk_data = self._get_chunk(chunk_idx)
        
        # Get single item from chunk
        if local_idx >= len(chunk_data):
            # Handle edge case where chunk size estimation was off
            local_idx = len(chunk_data) - 1
        
        data = chunk_data.iloc[local_idx:local_idx+1]
        
        # Convert to tensors
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        result = {
            "t": torch.tensor(data[self.time].values, dtype=torch.float32, device=device),
            "y": torch.tensor(data[self.target].values, dtype=torch.float32, device=device) if self.target else torch.tensor([], dtype=torch.float32, device=device),
            "x": torch.tensor(data[self.num].values, dtype=torch.float32, device=device) if self.num else torch.tensor([], dtype=torch.float32, device=device),
            "g": torch.tensor(data[self.group].values, dtype=torch.int64, device=device) if self.group else torch.tensor([], dtype=torch.int64, device=device),
            "st": torch.tensor(data[self.static].values, dtype=torch.float32, device=device) if self.static else torch.tensor([], dtype=torch.float32, device=device)
        }
        
        return result
    
    def get_metadata(self):
        """Return metadata about the dataset."""
        metadata = {
            "cols": {
                "y": self.target,
                "x": self.num + self.cat,
                "st": self.static,
            },
            "col_type": {},
            "col_known": {},
            "cat_index": []
        }
        
        # Set column types and known status
        all_cols = self.target + self.num + self.cat + self.static
        for i, col in enumerate(self.num + self.cat):
            if col in self.cat:
                metadata['cat_index'].append(i)
            
            metadata["col_type"][col] = "C" if col in self.cat else "F"
            metadata["col_known"][col] = "K" if col in self.known else "U"
        
        metadata['encoders'] = self.label_encoders
        
        return metadata
    
    @classmethod
    def from_pandas_files(cls, dataframes, zarr_paths, **kwargs):
        """Create MultiFileZarrTSDataset from multiple pandas DataFrames.
        
        Parameters
        ----------
        dataframes : List[pd.DataFrame]
            List of input DataFrames
        zarr_paths : List[str]
            List of paths to save zarr stores
        **kwargs : dict
            Additional arguments to pass to MultiFileZarrTSDataset constructor
            
        Returns
        -------
        MultiFileZarrTSDataset
            New dataset instance
        """
        if len(dataframes) != len(zarr_paths):
            raise ValueError("Number of dataframes must match number of zarr paths")
        
        # Create zarr stores
        for df, path in zip(dataframes, zarr_paths):
            store = zarr.open(path, mode='w')
            
            # Save each column as a zarr array
            for col in df.columns:
                store.create_dataset(col, data=df[col].values, chunks=True)
        
        return cls(zarr_paths, **kwargs)


# Example usage:
def create_example_data():
    """Create example data with different groups across files."""
    # Create first dataframe with groups A and B
    df1 = pd.DataFrame({
        'time': pd.date_range(start='2025-01-01', periods=100, freq='D'),
        'group': ['A'] * 50 + ['B'] * 50,
        'value': np.random.randn(100),
        'category': np.random.choice(['cat1', 'cat2', 'cat3'], size=100)
    })
    
    # Create second dataframe with groups B and C
    df2 = pd.DataFrame({
        'time': pd.date_range(start='2025-02-01', periods=100, freq='D'),
        'group': ['B'] * 50 + ['C'] * 50,
        'value': np.random.randn(100),
        'category': np.random.choice(['cat2', 'cat3', 'cat4'], size=100)  # Note: cat4 is new
    })
    
    # Print group ranges
    print("File 1:")
    for group in ['A', 'B']:
        group_data = df1[df1['group'] == group]
        print(f"  Group {group}: {group_data['time'].min()} to {group_data['time'].max()}")
    
    print("File 2:")
    for group in ['B', 'C']:
        group_data = df2[df2['group'] == group]
        print(f"  Group {group}: {group_data['time'].min()} to {group_data['time'].max()}")
    
    return [df1, df2]


def _coerce_to_list(obj):
    """Coerces input to a list."""
    if obj is None:
        return []
    elif isinstance(obj, list):
        return obj
    else:
        return [obj]


# Example of using the dataset
if __name__ == "__main__":
    # Create example data
    dataframes = create_example_data()
    
    # Create zarr paths
    zarr_paths = ['file1.zarr', 'file2.zarr']
    
    # Create dataset
    dataset = MultiFileZarrTSDataset.from_pandas_files(
        dataframes=dataframes,
        zarr_paths=zarr_paths,
        time='time',
        target='value',
        group=['group'],
        cat=['category'],
        chunk_size={'time': pd.Timedelta(days=10)}
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=16, num_workers=2)
    
    # Print metadata
    print("\nDataset Metadata:")
    metadata = dataset.get_metadata()
    print(f"Encoders: {metadata['encoders']}")
    print(f"Categorical indices: {metadata['cat_index']}")
    
    # Print first batch
    print("\nFirst batch:")
    for batch in dataloader:
        print(f"Batch shapes: y={batch['y'].shape}, x={batch['x'].shape}, g={batch['g'].shape}")
        break