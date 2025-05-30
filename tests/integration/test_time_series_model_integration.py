import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

# Add the project root to the Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import both D1 and D2 layers
from notebooks.sandeep.time_series_dataset import MultiSourceTSDataSet, TSDataModule

# Import the Base class from dsipts/models
from dsipts.models.base import Base
from dsipts.models.utils import get_scope

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple custom model that inherits from Base
class SimpleTimeSeriesModel(Base):
    """
    A simple time series model that inherits from Base for testing purposes.
    This model just applies a simple linear transformation to the input features.
    """
    # Define class attributes required by Base
    handle_multivariate = True
    handle_future_covariates = True  # Updated to handle future covariates
    handle_categorical_variables = True  # Updated to handle categorical variables
    handle_quantile_loss = False
    description = get_scope(handle_multivariate, handle_future_covariates, 
                           handle_categorical_variables, handle_quantile_loss)
    
    def __init__(self, past_len, future_len, feature_dim, target_dim, cat_dims=None, verbose=True):
        """
        Initialize the SimpleTimeSeriesModel.
        
        Args:
            past_len: Number of past time steps
            future_len: Number of future time steps to predict
            feature_dim: Number of feature dimensions
            target_dim: Number of target dimensions
            cat_dims: Dictionary of categorical dimensions {col_name: num_classes}
            verbose: Whether to print model information
        """
        super(SimpleTimeSeriesModel, self).__init__(verbose=verbose)
        
        # Model parameters
        self.past_len = past_len
        self.future_len = future_len
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.cat_dims = cat_dims or {}
        
        # Simple linear layers
        self.encoder = torch.nn.Linear(feature_dim, 64)
        self.decoder = torch.nn.Linear(64, target_dim)
        self.activation = torch.nn.ReLU()
        
        # Add embedding layers for categorical variables if needed
        self.embeddings = torch.nn.ModuleDict()
        for col, dim in self.cat_dims.items():
            self.embeddings[col] = torch.nn.Embedding(dim, min(16, dim // 2 + 1))
        
        # Required by Base class
        self.optim = None
        self.optim_config = {'lr': 1e-4}
        self.scheduler_config = None
    
    def forward(self, batch):
        """
        Forward pass of the model.
        
        Args:
            batch: Dictionary containing input data
                x_num_past: Past numerical features [batch_size, past_len, feature_dim]
                x_cat_past: Past categorical features [batch_size, past_len, cat_dim]
                x_num_future: Future numerical features [batch_size, future_len, feature_dim]
                x_cat_future: Future categorical features [batch_size, future_len, cat_dim]
                y: Target values [batch_size, future_len, target_dim] (optional, for training)
        
        Returns:
            Tensor of predictions [batch_size, future_len, target_dim]
        """
        # Extract past numerical features
        x = batch['x_num_past']  # [batch_size, past_len, feature_dim]
        batch_size = x.shape[0]
        
        # Process each time step
        encoded = self.encoder(x)  # [batch_size, past_len, hidden_dim]
        encoded = self.activation(encoded)
        
        # Use the last hidden state to predict future values
        last_hidden = encoded[:, -1, :]  # [batch_size, hidden_dim]
        
        # Expand to match future sequence length
        expanded = last_hidden.unsqueeze(1).expand(-1, self.future_len, -1)
        
        # Decode to get predictions
        predictions = self.decoder(expanded)  # [batch_size, future_len, target_dim]
        
        return predictions

# Function to convert D2 batch format to model input format
def convert_batch_for_model(batch, known_cat_cols=None, known_num_cols=None, unknown_cat_cols=None, unknown_num_cols=None):
    """
    Convert batch from D2 layer format to the format expected by the model
    
    Args:
        batch: Batch from D2 layer
        known_cat_cols: List of known categorical columns
        known_num_cols: List of known numerical columns
        unknown_cat_cols: List of unknown categorical columns
        unknown_num_cols: List of unknown numerical columns
        
    Returns:
        Dict with keys expected by the model
    """
    model_batch = {}
    
    # Get batch dimensions
    batch_size = batch['past_features'].shape[0] if 'past_features' in batch else 2
    past_len = batch['past_features'].shape[1] if 'past_features' in batch else 5
    future_len = batch['future_targets'].shape[1] if 'future_targets' in batch else 2
    
    # Extract past numerical features
    if 'past_features' in batch:
        model_batch['x_num_past'] = batch['past_features']
    
    # Extract future targets (for training)
    if 'future_targets' in batch:
        model_batch['y'] = batch['future_targets']
    
    # Add categorical features if available
    if 'past_cat_features' in batch:
        model_batch['x_cat_past'] = batch['past_cat_features']
    else:
        # Empty tensor for categorical features
        model_batch['x_cat_past'] = torch.zeros((batch_size, past_len, 0), dtype=torch.long)
    
    # Add future numerical features if available
    if 'future_num_features' in batch:
        model_batch['x_num_future'] = batch['future_num_features']
    else:
        # Empty tensor for future numerical features
        model_batch['x_num_future'] = torch.zeros((batch_size, future_len, 0), dtype=torch.float)
    
    # Add future categorical features if available
    if 'future_cat_features' in batch:
        model_batch['x_cat_future'] = batch['future_cat_features']
    else:
        # Empty tensor for future categorical features
        model_batch['x_cat_future'] = torch.zeros((batch_size, future_len, 0), dtype=torch.long)
    
    # Add static features if available
    if 'static' in batch:
        model_batch['static'] = batch['static']
    
    # Add group_id if available
    if 'group_id' in batch:
        model_batch['group_id'] = batch['group_id']
    
    # Add idx_target - the index of target variable in the input features
    # In our case, we're using the first feature as the target
    model_batch['idx_target'] = [0]  # Assuming the first feature corresponds to the target
    
    return model_batch

def create_dummy_dataset(path):
    """Create a simple time series dataset for testing"""
    logger.info(f"Creating dummy dataset at {path}")
    
    # Create a simple time series with 3 groups
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    groups = ['A', 'B', 'C']
    
    data = []
    for group in groups:
        for date in dates:
            value = np.sin(date.day/15) + np.random.normal(0, 0.1)
            feature1 = np.cos(date.day/10) + np.random.normal(0, 0.05)
            feature2 = date.day/30 + np.random.normal(0, 0.02)
            
            # Add categorical features for testing
            hour_of_day = date.hour  # 0-23
            day_of_week = date.dayofweek  # 0-6
            month = date.month  # 1-12
            
            data.append({
                'time': date,
                'group': group,
                'target': value,
                'feature1': feature1,
                'feature2': feature2,
                'hour': hour_of_day,
                'day_of_week': day_of_week,
                'month': month
            })
    
    df = pd.DataFrame(data)
    csv_path = path / 'dummy_ts.csv'
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Created dummy dataset with shape {df.shape}")
    return df, csv_path

def main():
    try:
        # Create a temporary directory for test data
        tmp_path = Path(os.getcwd()) / "test_data"
        tmp_path.mkdir(exist_ok=True)

        # Choose data source: dummy (default) or monash
        use_monash = os.environ.get("USE_MONASH_DATA", "0") == "1"
        logger.info(f"USE_MONASH_DATA={use_monash}")

        if use_monash:
            logger.info("Loading data via Monash loader...")
            from dsipts.data_management.monash import Monash
            # Pick a small dataset id for test (e.g., 1); change as needed
            monash_data_dir = tmp_path / "monash"
            monash_data_dir.mkdir(exist_ok=True)
            monash_obj = Monash(str(monash_data_dir / "monash_table"), rebuild=False)
            monash_obj.download_dataset(str(monash_data_dir), id=1, rebuild=False)
            df = monash_obj.generate_dataset(1)
            logger.info(f"Loaded Monash dataset shape: {df.shape}")
            # Infer columns (try to match expected names)
            time_col = 'start_timestamp' if 'start_timestamp' in df.columns else 'time'
            target_col = 'series_value' if 'series_value' in df.columns else 'target'
            group_col = 'series_name' if 'series_name' in df.columns else 'group'
            # Use all columns except time, group, target as features
            feature_cols = [c for c in df.columns if c not in [time_col, group_col, target_col]]
            # Save to CSV for pipeline compatibility
            csv_path = monash_data_dir / "monash_data.csv"
            df.to_csv(csv_path, index=False)
        else:
            logger.info("Using dummy dataset...")
            # Create a dummy dataset
            df, csv_path = create_dummy_dataset(tmp_path)
            time_col = 'time'
            target_col = 'target'
            group_col = 'group'
            feature_cols = ['feature1', 'feature2']

        logger.info(f"Dataset columns: {df.columns.tolist()}")
        logger.info(f"Dataset shape: {df.shape}")

        logger.info(f"Creating D1 dataset with: time_col={time_col}, target_col={target_col}, "
                   f"group_col={group_col}, feature_cols={feature_cols}")

        # Define categorical and known/unknown columns
        cat_cols = ['hour', 'day_of_week', 'month', 'group']
        known_cols = ['feature1', 'hour', 'day_of_week', 'month']  # Known at prediction time
        unknown_cols = ['feature2', 'target']  # Unknown at prediction time (need to be predicted)
        
        d1 = MultiSourceTSDataSet(
            file_paths=[str(csv_path)],
            group_cols=group_col,
            time_col=time_col,
            feature_cols=feature_cols + ['hour', 'day_of_week', 'month'],  # Include categorical features
            target_cols=[target_col],
            cat_cols=cat_cols,  # Specify categorical columns
            known_cols=known_cols,  # Specify known columns
            unknown_cols=unknown_cols,  # Specify unknown columns
            memory_efficient=False,
            chunk_size=1000
        )

        logger.info(f"D1 dataset created with {len(d1)} groups")
        
        # Define window sizes for past and future
        past_len = 5  # Number of past time steps
        future_len = 2  # Number of future time steps to predict
        
        # 3. Prepare D2 (TSDataModule)
        logger.info("Creating D2 TSDataModule")
        
        # Define split method and configuration
        # Option 1: Percentage-based split (temporal)
        split_method = 'percentage'
        split_config = (0.7, 0.15, 0.15)  # 70% train, 15% val, 15% test
        
        # Option 2: Group-based split with temporal ordering within groups
        # split_method = 'group'
        # split_config = (['A'], ['B'], ['C'])  # Group A for train, B for val, C for test
        
        logger.info(f"Using split method: {split_method} with config: {split_config}")
        
        d2 = TSDataModule(
            d1_dataset=d1,
            past_len=past_len,
            future_len=future_len,
            batch_size=2,
            split_method=split_method,
            split_config=split_config,
            # We're not overriding known/unknown columns here, using D1's metadata
            precompute=True
        )
        
        logger.info("Setting up D2 module")
        d2.setup()
        
        logger.info("Getting train dataloader")
        loader = d2.train_dataloader()
        
        logger.info("Fetching a batch")
        batch = next(iter(loader))
        
        logger.info(f"Batch keys: {batch.keys()}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: tensor with shape {v.shape}")
            else:
                logger.info(f"  {k}: {type(v)}")
        
        # 4. Instantiate SimpleTimeSeriesModel
        logger.info("Creating SimpleTimeSeriesModel")
        
        # Get dimensions from the batch
        feature_dim = batch['past_features'].shape[2] if 'past_features' in batch else 2
        target_dim = batch['future_targets'].shape[2] if 'future_targets' in batch else 1
        
        logger.info(f"Feature dimension: {feature_dim}, Target dimension: {target_dim}")
        
        # Get categorical dimensions from metadata if available
        cat_dims = {}
        if 'max_classes' in d2.metadata:
            cat_dims = d2.metadata['max_classes']
            logger.info(f"Categorical dimensions from metadata: {cat_dims}")
        
        # Create the SimpleTimeSeriesModel
        model = SimpleTimeSeriesModel(
            past_len=past_len,
            future_len=future_len,
            feature_dim=feature_dim,
            target_dim=target_dim,
            cat_dims=cat_dims,
            verbose=True
        )
        
        logger.info(f"Created SimpleTimeSeriesModel")
        
        # Log D2 metadata to understand column categorization
        logger.info("D2 Metadata:")
        for key, value in d2.metadata.items():
            if key in ['known_cat_cols', 'known_num_cols', 'unknown_cat_cols', 'unknown_num_cols']:
                logger.info(f"  {key}: {value}")
        
        # Extract batch structure information
        known_cat_cols = d2.metadata.get('known_cat_cols', [])
        known_num_cols = d2.metadata.get('known_num_cols', [])
        unknown_cat_cols = d2.metadata.get('unknown_cat_cols', [])
        unknown_num_cols = d2.metadata.get('unknown_num_cols', [])
        
        # Convert batch to model format
        logger.info("Converting batch to model format")
        model_batch = convert_batch_for_model(
            batch,
            known_cat_cols=known_cat_cols,
            known_num_cols=known_num_cols,
            unknown_cat_cols=unknown_cat_cols,
            unknown_num_cols=unknown_num_cols
        )
        
        logger.info(f"Model batch keys: {model_batch.keys()}")
        for k, v in model_batch.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: tensor with shape {v.shape}")
        
        # Run forward pass
        logger.info("Running forward pass")
        with torch.no_grad():  # Disable gradient calculation for inference
            output = model(model_batch)
        
        # 5. Check output shape
        logger.info(f"Output shape: {output.shape}")
        assert isinstance(output, torch.Tensor)
        
        # Check that output batch dimension matches input
        assert output.shape[0] == model_batch['x_num_past'].shape[0], "Batch size mismatch"
        assert output.shape[1] == future_len, "Sequence length mismatch"
        assert output.shape[2] == target_dim, "Feature dimension mismatch"
        
        # Additional validation for the enhanced batch structure
        logger.info("Validating enhanced batch structure...")
        
        # Check if we have the expected keys for categorical/numerical features
        expected_keys = ['x_num_past', 'x_cat_past', 'x_num_future', 'x_cat_future', 'y']
        for key in expected_keys:
            assert key in model_batch, f"Missing expected key: {key}"
            
        # Check if the dimensions are consistent
        assert model_batch['x_num_past'].shape[0] == model_batch['x_cat_past'].shape[0], "Batch size mismatch between numerical and categorical features"
        assert model_batch['x_num_future'].shape[1] == model_batch['y'].shape[1], "Future sequence length mismatch"
        
        logger.info("Test completed successfully!")
        logger.info("The D1/D2 layers work properly with the Base model.")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ SUCCESS: D1/D2 layers work properly with the model!")
    else:
        print("\n❌ FAILED: There were issues with the D1/D2 layers integration.")
