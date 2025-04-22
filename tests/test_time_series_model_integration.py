import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path

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
    handle_future_covariates = False
    handle_categorical_variables = False
    handle_quantile_loss = False
    description = get_scope(handle_multivariate, handle_future_covariates, 
                           handle_categorical_variables, handle_quantile_loss)
    
    def __init__(self, past_len, future_len, feature_dim, target_dim, verbose=True):
        """
        Initialize the SimpleTimeSeriesModel.
        
        Args:
            past_len: Number of past time steps
            future_len: Number of future time steps to predict
            feature_dim: Number of feature dimensions
            target_dim: Number of target dimensions
            verbose: Whether to print model information
        """
        super(SimpleTimeSeriesModel, self).__init__(verbose=verbose)
        
        # Model parameters
        self.past_len = past_len
        self.future_len = future_len
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        
        # Simple linear layers
        self.encoder = torch.nn.Linear(feature_dim, 64)
        self.decoder = torch.nn.Linear(64, target_dim)
        self.activation = torch.nn.ReLU()
        
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
def convert_batch_for_model(batch):
    """
    Convert batch from D2 layer format to the format expected by the LinearTS model
    
    Args:
        batch: Batch from D2 layer
        
    Returns:
        Dict with keys expected by the model
    """
    model_batch = {}
    
    # Extract past features
    if 'past_features' in batch:
        model_batch['x_num_past'] = batch['past_features']
    
    # Extract future targets (for training)
    if 'future_targets' in batch:
        model_batch['y'] = batch['future_targets']
    
    # Add other required fields with empty tensors if needed
    if 'x_cat_past' not in model_batch:
        # Empty tensor for categorical features (not used in this test)
        batch_size = model_batch['x_num_past'].shape[0] if 'x_num_past' in model_batch else 2
        model_batch['x_cat_past'] = torch.zeros((batch_size, 1, 0), dtype=torch.long)
    
    # Add idx_target - the index of target variable in the input features
    # In our case, we're using the last feature as the target
    feature_dim = model_batch['x_num_past'].shape[2] if 'x_num_past' in model_batch else 2
    # Create a list of indices (0 for the first target variable)
    model_batch['idx_target'] = [0]  # Assuming the first feature corresponds to the target
    
    # Add x_num_future (empty tensor since we're not using future covariates)
    batch_size = model_batch['x_num_past'].shape[0] if 'x_num_past' in model_batch else 2
    future_len = model_batch['y'].shape[1] if 'y' in model_batch else 2
    model_batch['x_num_future'] = torch.zeros((batch_size, future_len, 0), dtype=torch.float)
    
    # Add x_cat_future (empty tensor since we're not using categorical future covariates)
    model_batch['x_cat_future'] = torch.zeros((batch_size, future_len, 0), dtype=torch.long)
    
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
            
            data.append({
                'time': date,
                'group': group,
                'target': value,
                'feature1': feature1,
                'feature2': feature2
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
        
        logger.info("Starting D1/D2 model integration test with LinearTS model")
        
        # Create a dummy dataset
        df, csv_path = create_dummy_dataset(tmp_path)
        
        logger.info(f"Dataset columns: {df.columns.tolist()}")
        logger.info(f"Dataset shape: {df.shape}")
        
        # 2. Prepare D1 dataset
        time_col = 'time'
        target_col = 'target'
        group_col = 'group'
        feature_cols = ['feature1', 'feature2']
        
        logger.info(f"Creating D1 dataset with: time_col={time_col}, target_col={target_col}, "
                   f"group_col={group_col}, feature_cols={feature_cols}")
        
        d1 = MultiSourceTSDataSet(
            file_paths=[str(csv_path)],
            group_cols=group_col,
            time_col=time_col,
            feature_cols=feature_cols,
            target_cols=[target_col],
            memory_efficient=False,
            chunk_size=1000
        )
        
        logger.info(f"D1 dataset created with {len(d1)} groups")
        
        # Define window sizes for past and future
        past_len = 5  # Number of past time steps
        future_len = 2  # Number of future time steps to predict
        
        # 3. Prepare D2 (TSDataModule)
        logger.info("Creating D2 TSDataModule")
        d2 = TSDataModule(
            d1_dataset=d1,
            past_len=past_len,
            future_len=future_len,
            batch_size=2,
            split_method='percentage',
            split_config=(0.7, 0.15, 0.15),
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
        
        # Create the SimpleTimeSeriesModel
        model = SimpleTimeSeriesModel(
            past_len=past_len,
            future_len=future_len,
            feature_dim=feature_dim,
            target_dim=target_dim,
            verbose=True
        )
        
        logger.info(f"Created SimpleTimeSeriesModel")
        
        # Convert batch to model format
        logger.info("Converting batch to model format")
        model_batch = convert_batch_for_model(batch)
        
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
