""""
CURRENT STATUS (ON HOLD, DEVELOPMENT WILL BE RESUMED ONCE BASELINE D1D2+MODELLING INTEGRATION IS DONE):
This simplified version:
    Handles multiple CSV files instead of Zarr stores
    Processes data in chunks using pandas' chunked reading
    Maintains a global label encoder that updates as new categories are encountered
    Provides a DataLoader for efficient data handling
    Includes an example of how to create and use the dataset


The main differences from the Zarr version are:
    Uses CSV files instead of Zarr stores
    Simpler chunking mechanism using pandas' chunked reading
    No Dask dependency
    More straightforward file handling
"""

"""
TODO
linking d1d2
    d2 should be in sync with d1

weights in the return (use d1 as reference)
return also the groups ()


post on github about the current state of the code

d2 layer: precompute the dataset/ stream the the data while using dataloader
doubt: how to normalize(preprocess) the data if we dont read whole data at once. 

look into : prepare data in pytorch lighting module. 

secondary tasks:
implement min max scalar on train and validation data which cant fit in memory.
implement standard scalar on train and validation data which cant fit in memory.

"""


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder
from typing import List, Dict, Optional, Union
import os

class MultiFileTSDataset(Dataset):
    """PyTorch Dataset for handling multiple files with different groups.
    
    This class processes multiple files containing time series data with different
    groups, maintains a global label encoder, and handles data in chunks.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        time_col: str,
        group_cols: List[str],
        target_cols: List[str],
        cat_cols: List[str],
        chunk_size: int = 1000 
    ):
        """
        Parameters
        ----------
        file_paths : List[str]
            List of paths to CSV files
        time_col : str
            Name of the time column
        group_cols : List[str]
            Columns to group by
        target_cols : List[str]
            Target columns to predict
        cat_cols : List[str]
            Categorical columns to encode
        chunk_size : int
            Number of rows to process at a time
        """
        self.file_paths = file_paths
        self.time_col = time_col
        self.group_cols = group_cols
        self.target_cols = target_cols
        self.cat_cols = cat_cols
        self.chunk_size = chunk_size #on what basis chunking should be done while reading the data?
        # temporal information might get lost during chunking if user data is not sorted by default.
        
        #  label encoders for categorical and group columns
        self.label_encoders = {col: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) 
                              for col in cat_cols + group_cols}
        
        # lists to store file and group information
        self.file_info = []
        self.group_info = {}
        
        # Process files and build index
        self._process_files()
        
    def _process_files(self):
        """Process each file to extract group information and update encoders."""
        self.total_length = 0  #init total length of the dataset
        
        for file_idx, file_path in enumerate(self.file_paths):
            # Read each file in chunks
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                # updating label encoders with new categories from the chunk
                self._update_encoders(chunk)
                
                # If group columns are specified
                if self.group_cols:
                    # unique groups in the chunk
                    groups = chunk[self.group_cols].drop_duplicates()
                    
                    for _, group_row in groups.iterrows():
                        # Create a key from the group columns' values
                        group_key = tuple(group_row[self.group_cols].values)
                        
                        # If this group is not seen before, initialize its entry in group_info
                        if group_key not in self.group_info:
                            self.group_info[group_key] = []
                        # Add the current file index and file info index to the group's entry
                        self.group_info[group_key].append((file_idx, len(self.file_info)))
                        
                        # Filter data for the current group
                        group_mask = (chunk[self.group_cols] == group_row).all(axis=1)
                        group_data = chunk[group_mask]
                        
                        # Store information about the group in file_info
                        self.file_info.append({
                            'file_idx': file_idx,
                            'group_key': group_key,
                            'min_time': group_data[self.time_col].min(),
                            'max_time': group_data[self.time_col].max(),
                            'length': len(group_data)
                        })
                        
                        self.total_length += len(group_data)  # updating total length
                else:
                    # if no group columns -> store file information
                    self.file_info.append({
                        'file_idx': file_idx,
                        'group_key': None,
                        'min_time': chunk[self.time_col].min(),
                        'max_time': chunk[self.time_col].max(),
                        'length': len(chunk)
                    })
                    
                    self.total_length += len(chunk)  # Update total length
    
    def _update_encoders(self, data):
        """Update label encoders with new categories from the data."""
        for col in self.cat_cols + self.group_cols:  # for each categorical and group column
            if col in data.columns:  # If the column exists in the data
                unique_values = data[col].dropna().unique().reshape(-1, 1)  # Get unique values
                if len(unique_values) > 0:  # If there are unique values
                    self.label_encoders[col].fit(unique_values)  # update the encoder for this column
    
    def __len__(self):
        """Return the total number of items in the dataset."""
        return self.total_length
    
    def _calculate_weight(self, timestamp):
        """Calculate weight based on time difference from most recent sample."""
        # Convert timestamp to datetime if needed
        if isinstance(timestamp, (int, float)):
            timestamp = pd.to_datetime(timestamp, unit='s')
        elif isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Get most recent time in dataset
        most_recent = max(info['max_time'] for info in self.file_info)
        
        # Calculate time difference in days
        time_diff = (most_recent - timestamp).days
        
        # Apply exponential decay for weights
        # You can adjust the decay rate (0.1) as needed
        weight = np.exp(-0.1 * time_diff)
        
        return weight

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        # Find the file and group that the given index belongs to
        cumulative_length = 0
        for info in self.file_info:
            if idx < cumulative_length + info['length']:
                file_idx = info['file_idx']
                group_key = info['group_key']
                local_idx = idx - cumulative_length
                break
            cumulative_length += info['length']
        
        # Load the relevant chunk from the file
        chunk = self._load_chunk(file_idx, group_key)
        
        # Get the specific row from the chunk
        row = chunk.iloc[local_idx]
        
        # Encode categorical columns
        encoded_data = {}
        for col in self.cat_cols + self.group_cols:
            if col in row.index:
                encoded_data[col] = self.label_encoders[col].transform([[row[col]]])[0][0]
        
        # Convert time to numeric timestamp
        time_value = pd.to_datetime(row[self.time_col]).timestamp()
        
        # Extract target values and ensure they are numeric
        target_values = []
        for col in self.target_cols:
            target_values.append(float(row[col]))
        
        # Convert data to PyTorch tensors 
        result = {
            "t": pd.to_datetime(row[self.time_col]).timestamp(), # NOT TENSOR 
            "y": torch.tensor(target_values, dtype=torch.float32),
            "x": torch.tensor(list(encoded_data.values()), dtype=torch.float32),
            "weights": torch.tensor(self._calculate_weight(row[self.time_col]), dtype=torch.float32)
}
        
        return result
    
    def _load_chunk(self, file_idx, group_key):
        """Load a chunk of data from a specific file and group."""
        # Load the next chunk from the specified file
        chunk = pd.read_csv(self.file_paths[file_idx], chunksize=self.chunk_size).__next__()
        
        # Convert the time column to datetime format
        chunk[self.time_col] = pd.to_datetime(chunk[self.time_col])
        
        # If a group key is specified, filter the chunk by that group
        if group_key is not None:
            group_mask = True
            for i, col in enumerate(self.group_cols):
                group_mask = group_mask & (chunk[col] == group_key[i])
            return chunk[group_mask]
        return chunk

# Example usage
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

if __name__ == "__main__":
    # Create example files
    create_example_files()
    
    # Create dataset instance
    dataset = MultiFileTSDataset(
        file_paths=['file1.csv', 'file2.csv'],
        time_col='time',
        group_cols=['group'],
        target_cols=['value'],
        cat_cols=['category'],
        chunk_size=50
    )
    
    # Print dataset information
    print("=== Dataset Information ===")
    print(f"Total items: {len(dataset)}")
    print("\nFile and Group Information:")
    for i, info in enumerate(dataset.file_info):
        print(f"Item {i}:")
        print(f"  File: {dataset.file_paths[info['file_idx']]}")
        print(f"  Group: {info['group_key']}")
        print(f"  Time Range: {info['min_time']} to {info['max_time']}")
        print(f"  Length: {info['length']}")
    
    # Print encoder information
    print("\n=== Encoder Information ===")
    for col, encoder in dataset.label_encoders.items():
        print(f"{col} categories: {encoder.categories_[0].tolist()}")
        print(f"{col} mapping: {dict(zip(encoder.categories_[0], range(len(encoder.categories_[0]))))}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0)
    
    # Print first batch information
    for batch in dataloader:
        print("\n=== First Batch Information ===")
        print(f"Batch size: {batch['t'].size(0)}")
        print(f"Time shape: {batch['t'].shape}")
        print(f"Values shape: {batch['y'].shape}")
        print(f"Encoded categories shape: {batch['x'].shape}")
        
        # Print sample data from the batch
        print("\nSample data from batch:")
        for i in range(min(5, batch['t'].size(0))):
            print(f"Item {i}:")
            print(f"  Time: {batch['t'][i].item()} (timestamp)")
            print(f"  Value: {batch['y'][i].item()}")
            print(f"  Encoded categories: {batch['x'][i].tolist()}")
        break