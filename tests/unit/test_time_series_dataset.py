import unittest
import os
import sys
import pandas as pd
import numpy as np
import torch
from datetime import timedelta
import tempfile
import shutil

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from notebooks.sandeep.time_series_dataset import (
    extend_time_df,
    MultiSourceTSDataSet,
    TSDataModule
)

class TestExtendTimeDF(unittest.TestCase):
    """Test cases for the extend_time_df function."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple dataframe with time gaps
        self.df = pd.DataFrame({
            'time': [1, 3, 6, 10],
            'group': ['A', 'A', 'A', 'A'],
            'value': [10, 20, 30, 40]
        })
        
        # Create a dataframe with multiple groups
        self.multi_group_df = pd.DataFrame({
            'time': [1, 3, 6, 10, 1, 4, 8],
            'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50, 60, 70]
        })
        
        # Create a dataframe with datetime values
        self.datetime_df = pd.DataFrame({
            'time': pd.date_range(start='2025-01-01', periods=4, freq='2D'),
            'group': ['A', 'A', 'A', 'A'],
            'value': [10, 20, 30, 40]
        })
    
    def test_extend_numeric_time(self):
        """Test extending dataframe with numeric time values."""
        # Extend with frequency 1
        extended_df = extend_time_df(self.df, 'time', 1)
        
        # Check that all time points are filled
        expected_times = list(range(1, 11))
        self.assertEqual(len(extended_df), len(expected_times))
        self.assertListEqual(sorted(extended_df['time'].tolist()), expected_times)
        
        # Check that original values are preserved
        for _, row in self.df.iterrows():
            extended_row = extended_df[extended_df['time'] == row['time']]
            self.assertEqual(extended_row['value'].iloc[0], row['value'])
        
        # Check that new rows have NaN values
        for t in [2, 4, 5, 7, 8, 9]:
            extended_row = extended_df[extended_df['time'] == t]
            self.assertTrue(np.isnan(extended_row['value'].iloc[0]))
    
    def test_extend_with_groups(self):
        """Test extending dataframe with group columns."""
        # Extend with frequency 1 and group column
        extended_df = extend_time_df(self.multi_group_df, 'time', 1, group_cols='group')
        
        # Check that each group is extended separately
        group_a = extended_df[extended_df['group'] == 'A']
        group_b = extended_df[extended_df['group'] == 'B']
        
        # Group A should have time points 1-10
        self.assertEqual(len(group_a), 10)
        self.assertListEqual(sorted(group_a['time'].tolist()), list(range(1, 11)))
        
        # Group B should have time points 1-8
        self.assertEqual(len(group_b), 8)
        self.assertListEqual(sorted(group_b['time'].tolist()), list(range(1, 9)))
    
    def test_extend_datetime(self):
        """Test extending dataframe with datetime values."""
        # Extend with frequency 1 day
        extended_df = extend_time_df(self.datetime_df, 'time', timedelta(days=1))
        
        # Should have 7 rows (2025-01-01 to 2025-01-07)
        self.assertEqual(len(extended_df), 7)
        
        # Check that original values are preserved
        for _, row in self.datetime_df.iterrows():
            extended_row = extended_df[extended_df['time'] == row['time']]
            self.assertEqual(extended_row['value'].iloc[0], row['value'])
        
        # Check that new rows have NaN values
        for date in pd.date_range(start='2025-01-02', end='2025-01-06', freq='2D'):
            extended_row = extended_df[extended_df['time'] == date]
            self.assertTrue(np.isnan(extended_row['value'].iloc[0]))


class TestMultiSourceTSDataSet(unittest.TestCase):
    """Test cases for the MultiSourceTSDataSet class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV files
        self.create_test_files()
        
        # Define common parameters
        self.file_paths = [
            os.path.join(self.temp_dir, f'test_data_{i}.csv') for i in range(2)
        ]
        self.group_cols = 'group'
        self.time_col = 'time'
        self.feature_cols = ['feature_0', 'feature_1']
        self.target_cols = ['target_0']
        self.cat_cols = ['cat_feature']
        
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test CSV files."""
        # Generate two CSV files with different groups
        for file_idx in range(2):
            data = []
            
            # Generate data for each group
            for group_idx in range(3):
                # Determine which file gets which groups
                if (group_idx % 2 == 0 and file_idx == 0) or (group_idx % 2 == 1 and file_idx == 1):
                    # Generate time series for this group
                    for t in range(10):
                        row = {
                            'group': f'group_{group_idx}',
                            'time': t,
                            'feature_0': np.sin(t/10 + group_idx) + np.random.normal(0, 0.1),
                            'feature_1': np.cos(t/10 + group_idx) + np.random.normal(0, 0.1),
                            'target_0': np.sin(t/5 + group_idx) + np.random.normal(0, 0.1),
                            'cat_feature': f'cat_{np.random.randint(0, 3)}'
                        }
                        data.append(row)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(self.temp_dir, f'test_data_{file_idx}.csv'), index=False)
    
    def test_init_memory_efficient_false(self):
        """Test initialization with memory_efficient=False."""
        d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            memory_efficient=False
        )
        
        # Check that the dataset was initialized correctly
        self.assertEqual(len(d1_dataset.file_paths), 2)
        self.assertEqual(d1_dataset.time_col, 'time')
        self.assertEqual(d1_dataset.group_cols, ['group'])
        self.assertListEqual(d1_dataset.feature_cols, ['feature_0', 'feature_1'])
        self.assertListEqual(d1_dataset.target_cols, ['target_0'])
        self.assertListEqual(d1_dataset.cat_cols, ['cat_feature'])
        self.assertFalse(d1_dataset.memory_efficient)
        
        # Check that data was preloaded
        self.assertTrue(len(d1_dataset.data_cache) > 0)
        
        # Check that metadata was created
        self.assertTrue('cols' in d1_dataset.metadata)
        self.assertTrue('max_classes' in d1_dataset.metadata)
        self.assertTrue('known_cat_cols' in d1_dataset.metadata)
        self.assertTrue('known_num_cols' in d1_dataset.metadata)
    
    def test_init_memory_efficient_true(self):
        """Test initialization with memory_efficient=True."""
        d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            memory_efficient=True
        )
        
        # Check that the dataset was initialized correctly
        self.assertEqual(len(d1_dataset.file_paths), 2)
        self.assertEqual(d1_dataset.time_col, 'time')
        self.assertEqual(d1_dataset.group_cols, ['group'])
        self.assertListEqual(d1_dataset.feature_cols, ['feature_0', 'feature_1'])
        self.assertListEqual(d1_dataset.target_cols, ['target_0'])
        self.assertListEqual(d1_dataset.cat_cols, ['cat_feature'])
        self.assertTrue(d1_dataset.memory_efficient)
        
        # Check that data was not preloaded
        self.assertEqual(len(d1_dataset.data_cache), 0)
    
    def test_getitem(self):
        """Test __getitem__ method."""
        d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            memory_efficient=False
        )
        
        # Get data for the first group
        group_data = d1_dataset[0]
        
        # Check that the returned data has the expected format
        self.assertTrue('x' in group_data)
        self.assertTrue('y' in group_data)
        self.assertTrue('t' in group_data)
        self.assertTrue('group_id' in group_data)
        
        # Check dimensions
        self.assertEqual(group_data['x'].shape[1], len(self.feature_cols))
        self.assertEqual(group_data['y'].shape[1], len(self.target_cols))
        self.assertEqual(len(group_data['t']), len(group_data['x']))
    
    def test_known_unknown_cols(self):
        """Test specifying known and unknown columns."""
        # Specify custom known and unknown columns
        known_cols = ['feature_0']
        unknown_cols = ['feature_1', 'target_0']
        
        d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            known_cols=known_cols,
            unknown_cols=unknown_cols,
            memory_efficient=False
        )
        
        # Check that known and unknown columns were set correctly
        self.assertListEqual(d1_dataset.known_cols, known_cols)
        self.assertListEqual(d1_dataset.unknown_cols, unknown_cols)
        
        # Check metadata
        self.assertListEqual(d1_dataset.metadata['known_cols'], known_cols)
        self.assertListEqual(d1_dataset.metadata['unknown_cols'], unknown_cols)
        self.assertListEqual(d1_dataset.metadata['known_num_cols'], known_cols)  # feature_0 is numerical
        self.assertEqual(len(d1_dataset.metadata['known_cat_cols']), 0)  # No categorical known cols


class TestTSDataModule(unittest.TestCase):
    """Test cases for the TSDataModule class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV files
        self.create_test_files()
        
        # Create D1 dataset
        self.file_paths = [
            os.path.join(self.temp_dir, f'test_data_{i}.csv') for i in range(2)
        ]
        self.group_cols = 'group'
        self.time_col = 'time'
        self.feature_cols = ['feature_0', 'feature_1']
        self.target_cols = ['target_0']
        self.cat_cols = ['cat_feature']
        
        self.d1_dataset = MultiSourceTSDataSet(
            file_paths=self.file_paths,
            group_cols=self.group_cols,
            time_col=self.time_col,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            cat_cols=self.cat_cols,
            memory_efficient=False
        )
        
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test CSV files."""
        # Generate two CSV files with different groups
        for file_idx in range(2):
            data = []
            
            # Generate data for each group
            for group_idx in range(3):
                # Determine which file gets which groups
                if (group_idx % 2 == 0 and file_idx == 0) or (group_idx % 2 == 1 and file_idx == 1):
                    # Generate time series for this group
                    for t in range(20):  # Longer sequences for window creation
                        row = {
                            'group': f'group_{group_idx}',
                            'time': t,
                            'feature_0': np.sin(t/10 + group_idx) + np.random.normal(0, 0.1),
                            'feature_1': np.cos(t/10 + group_idx) + np.random.normal(0, 0.1),
                            'target_0': np.sin(t/5 + group_idx) + np.random.normal(0, 0.1),
                            'cat_feature': f'cat_{np.random.randint(0, 3)}'
                        }
                        data.append(row)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(self.temp_dir, f'test_data_{file_idx}.csv'), index=False)
    
    def test_init_percentage_split(self):
        """Test initialization with percentage split."""
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
            memory_efficient=False
        )
        
        # Check that the module was initialized correctly
        self.assertEqual(d2_module.past_len, 5)
        self.assertEqual(d2_module.future_len, 2)
        self.assertEqual(d2_module.batch_size, 32)
        self.assertEqual(d2_module.min_valid_length, 4)
        self.assertEqual(d2_module.split_method, 'percentage')
        self.assertEqual(d2_module.split_config, (0.7, 0.15, 0.15))
        
        # Check that splits were created
        self.assertTrue(hasattr(d2_module, 'train_indices'))
        self.assertTrue(hasattr(d2_module, 'val_indices'))
        self.assertTrue(hasattr(d2_module, 'test_indices'))
        
        # Check that the sum of split sizes equals the total number of windows
        total_windows = len(d2_module.mapping)
        split_sum = len(d2_module.train_indices) + len(d2_module.val_indices) + len(d2_module.test_indices)
        self.assertEqual(split_sum, total_windows)
        
        # Check approximate split ratios (allowing for rounding)
        self.assertAlmostEqual(len(d2_module.train_indices) / total_windows, 0.7, delta=0.05)
        self.assertAlmostEqual(len(d2_module.val_indices) / total_windows, 0.15, delta=0.05)
        self.assertAlmostEqual(len(d2_module.test_indices) / total_windows, 0.15, delta=0.05)
    
    def test_init_group_split(self):
        """Test initialization with group split."""
        # Get all group indices
        all_groups = list(range(len(self.d1_dataset)))
        
        # Split groups into train/val/test
        train_groups = all_groups[:int(0.7 * len(all_groups))]
        val_groups = all_groups[int(0.7 * len(all_groups)):int(0.85 * len(all_groups))]
        test_groups = all_groups[int(0.85 * len(all_groups)):]
        
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='group',
            split_config=(train_groups, val_groups, test_groups),
            memory_efficient=False
        )
        
        # Check that the module was initialized correctly
        self.assertEqual(d2_module.past_len, 5)
        self.assertEqual(d2_module.future_len, 2)
        self.assertEqual(d2_module.split_method, 'group')
        
        # Check that splits were created
        self.assertTrue(hasattr(d2_module, 'train_indices'))
        self.assertTrue(hasattr(d2_module, 'val_indices'))
        self.assertTrue(hasattr(d2_module, 'test_indices'))
        
        # Check that each split only contains windows from the assigned groups
        for idx in d2_module.train_indices:
            group_idx, _ = d2_module.mapping[idx]
            self.assertIn(group_idx, train_groups)
            
        for idx in d2_module.val_indices:
            group_idx, _ = d2_module.mapping[idx]
            self.assertIn(group_idx, val_groups)
            
        for idx in d2_module.test_indices:
            group_idx, _ = d2_module.mapping[idx]
            self.assertIn(group_idx, test_groups)
    
    def test_get_window(self):
        """Test _get_window method."""
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
            memory_efficient=False
        )
        
        # Get a window from the dataset
        if d2_module.train_indices:
            idx = d2_module.train_indices[0]
            group_idx, local_idx = d2_module.mapping[idx]
            window = d2_module._get_window(group_idx, local_idx)
            
            # Check that the window has the expected format
            self.assertTrue('past_features' in window)
            self.assertTrue('past_time' in window)
            self.assertTrue('future_targets' in window)
            self.assertTrue('future_time' in window)
            
            # Check dimensions
            self.assertEqual(window['past_features'].shape[0], d2_module.past_len)
            self.assertEqual(window['past_features'].shape[1], len(self.feature_cols))
            self.assertEqual(window['future_targets'].shape[0], d2_module.future_len)
            self.assertEqual(window['future_targets'].shape[1], len(self.target_cols))
            self.assertEqual(len(window['past_time']), d2_module.past_len)
            self.assertEqual(len(window['future_time']), d2_module.future_len)
    
    def test_known_unknown_override(self):
        """Test overriding known and unknown columns."""
        # Specify custom known and unknown columns
        known_cols = ['feature_0']
        unknown_cols = ['feature_1', 'target_0']
        
        d2_module = TSDataModule(
            d1_dataset=self.d1_dataset,
            past_len=5,
            future_len=2,
            batch_size=32,
            min_valid_length=4,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
            memory_efficient=False,
            known_cols=known_cols,
            unknown_cols=unknown_cols
        )
        
        # Check that known and unknown columns were set correctly
        self.assertListEqual(d2_module.known_cols, known_cols)
        self.assertListEqual(d2_module.unknown_cols, unknown_cols)
        
        # Check metadata
        self.assertListEqual(d2_module.metadata['known_cols'], known_cols)
        self.assertListEqual(d2_module.metadata['unknown_cols'], unknown_cols)
        self.assertListEqual(d2_module.metadata['known_num_cols'], known_cols)  # feature_0 is numerical
        self.assertEqual(len(d2_module.metadata['known_cat_cols']), 0)  # No categorical known cols


if __name__ == '__main__':
    unittest.main()
