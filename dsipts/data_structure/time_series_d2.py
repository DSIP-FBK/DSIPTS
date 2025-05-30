"""
Time Series D2 Layer Module

This module provides the D2 layer for time series data processing:
- TSDataModule: LightningDataModule for time series data with support for training, validation, and testing
- TimeSeriesSubset: Subset implementation for train/val/test splits
- custom_collate_fn: Custom collate function for handling mixed data types

Key Features:
- Creates sliding windows from time series data
- Handles train/validation/test splits (percentage-based or group-based)
- Validates data points based on minimum valid requirements
- Creates DataLoaders for PyTorch Lightning integration
- Efficiently manages memory with caching mechanisms
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

# Import the D1 layer
from dsipts.data_structure.time_series_d1 import MultiSourceTSDataSet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TSDataModule(pl.LightningDataModule):
    """D2 Layer - Processes time series data for model consumption.
    
    This module:
    1. Creates sliding windows from time series data
    2. Handles train/validation/test splits
    3. Creates DataLoaders for PyTorch Lightning
    """
    
    def __init__(
        self,
        d1_dataset: MultiSourceTSDataSet,
        past_len: int,
        future_len: int,
        batch_size: int = 32,
        min_valid_length: Optional[int] = None,
        split_method: str = 'percentage',
        split_config: Optional[tuple] = None,
        num_workers: int = 0,
        sampler: Optional[Sampler] = None,
        memory_efficient: bool = False,
        known_cols: Optional[List[str]] = None,
        unknown_cols: Optional[List[str]] = None,
        precompute: bool = True
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
            precompute: Whether to precompute valid indices and create datasets
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
        
        # Default split configuration based on method
        if split_config is None:
            if split_method == 'percentage':
                self.split_config = (0.7, 0.15, 0.15)  # Default: 70% train, 15% val, 15% test
            else:
                raise ValueError("For 'group' split method, split_config must be provided")
        
        # Whether to precompute valid indices and create datasets
        self.precompute = precompute
        
        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
                
        # Initialize the module
        self._initialize()
        
        # Add max classes information to metadata
        self._add_max_classes_to_metadata()
    
    def _initialize(self):
        """Initialize the dataset by computing valid indices and mappings."""
        print("Computing valid indices and mappings...")
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
        
        # Create subset datasets for train/val/test using the indices if precompute is enabled
        # Otherwise, the datasets will be created on-demand during setup
        if self.precompute:
            print("Precomputing datasets for train/val/test splits...")
            if hasattr(self, 'train_indices') and self.train_indices:
                self.train_dataset = TimeSeriesSubset(self, self.train_indices)
            if hasattr(self, 'val_indices') and self.val_indices:
                self.val_dataset = TimeSeriesSubset(self, self.val_indices)
            if hasattr(self, 'test_indices') and self.test_indices:
                self.test_dataset = TimeSeriesSubset(self, self.test_indices)
        else:
            print("Datasets will be created on-demand during setup...")
    
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
        # Load the group data from D1 dataset
        group_data = self.d1_dataset[group_idx]
            
        return group_data
        
    def _add_max_classes_to_metadata(self):
        """
        Add information about maximum number of classes for categorical features.
        This is useful for model architecture decisions.
        """
        # Copy max classes information from D1 dataset metadata
        if 'max_classes' in self.d1_dataset.metadata:
            self.metadata['max_classes'] = self.d1_dataset.metadata['max_classes'].copy()
            
        # Ensure known/unknown column categorization is properly reflected in metadata
        self.metadata['known_cols'] = self.known_cols
        self.metadata['unknown_cols'] = self.unknown_cols
        
        # Add categorical/numerical classification for known/unknown columns
        self.metadata['known_cat_cols'] = [col for col in self.known_cols if col in self.d1_dataset.cat_cols]
        self.metadata['known_num_cols'] = [col for col in self.known_cols if col not in self.d1_dataset.cat_cols]
        self.metadata['unknown_cat_cols'] = [col for col in self.unknown_cols if col in self.d1_dataset.cat_cols]
        self.metadata['unknown_num_cols'] = [col for col in self.unknown_cols if col not in self.d1_dataset.cat_cols]
    
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
            
            # Try to sort by time if available
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
        
        # If precompute is False, create datasets on-demand during setup
        # Otherwise, datasets were already created during initialization
        if not self.precompute:
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
            data_module: The TSDataModule instance (stored as reference, not copy)
            indices: List of indices to include in this subset
        """
        # In Python, this assignment creates a reference to the original data_module object
        # No copying occurs, so all subsets share the same data_module instance
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
