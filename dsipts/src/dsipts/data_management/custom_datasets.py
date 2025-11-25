# Loading custom datasets

# Simple case t,X_{1},X_{2},..,X_{d},y, where t is the time index, X_{1},X_{2},..,X_{d} are the covariates and y is the target variable, X can be categorical or numeric, y can be numeric or categorical. But for this case all X, and y are numeric.
# NOTE: FOLOSM AS A CASE STUDY AND TEST THE CODE

from typing import Tuple
import os
import pandas as pd
from typing import List,Dict
import logging

logging.basicConfig(level=logging.INFO)


# ----- UTIL FUNCTIONS -----
def load_csv(path:str, filename:str, **kwargs)->pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    ASSUMPTION: dataset is a csv file, with ',' as separator, and first row as header.
    Args:
        path (str): Path to the CSV file.
        dataset (str): Name of the dataset.
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    if not filename.endswith('.csv'):
        filename = filename + '.csv'
    file_path = os.path.join(path, filename)
    logging.info(f"Loading {filename}")
    try:
        df = pd.read_csv(file_path, sep=',',na_values=-9999, **kwargs) # match read_public_dataset
    except Exception as e:
        logging.error(f"Failed to load {filename}: {e}")
        raise
    logging.info(f"✓ Loaded {filename}")
    return df

def _rename_columns(df:pd.DataFrame, possible_timestamp_names:list=['time', 'timestamp', 'datetime', 'valtime'], possible_target_names:list=['target', 'y', 'label'], target_name:str=None, timestamp_name:str='t')->pd.DataFrame:
    """
    Rename target, and timestamp columns columns in dataset, to t, and y.
    Args:
        df (pd.DataFrame): DataFrame to rename columns.
        possible_timestamp_names (list): List of possible timestamp column names.
        possible_target_names (list): List of possible target column names.
    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    # time stamp column can vary like time, timestamp, datetime, valtime, etc.
    # target column can vary like target, y, label, specific column name etc.
    # rename_dict is a dictionary mapping old column names to new column names.
    # rename_dict = {'target': 'y', 'timestamp': 't'}
    # check if one of possible timestamp names is in the df
    df = df.copy()
    if timestamp_name not in df.columns:
        for possible_timestamp_name in possible_timestamp_names:
            if possible_timestamp_name in df.columns:
                df.rename(columns={possible_timestamp_name: timestamp_name}, inplace=True)
                logging.info(f"✓ Renamed {possible_timestamp_name} column to {timestamp_name}")
                break
    # check if one of possible target names is in the df
    if target_name and target_name not in df.columns:
        for possible_target_name in possible_target_names:
            if possible_target_name in df.columns:
                df.rename(columns={possible_target_name: target_name}, inplace=True)
        logging.info(f"✓ Renamed {possible_target_name} column to {target_name}")
    return df



def load_custom_dataset(path:str, 
                        dataset:str, 
                        files:List[str]=None, 
                        custom_format:Dict[str,str]={}, 
                        target_tz:str='UTC', 
                        target_name:str='y', 
                        timestamp_name:str='time', 
                        possible_timestamp_names:list=['time', 'timestamp', 'datetime', 'valtime'], 
                        possible_target_names:list=['target', 'y', 'label'],
                        drop_columns:list=[],
                        target_file_name:str=None,
                        set_index_timestamps:bool=True)->Dict[str,pd.DataFrame]:
    """
    
    Load CSV files, set index timestamps to UTC, and add a 'PST_Time' column.
    NOTE: DECOUPLING IS NOT DONE YET, as data must be loaded then fixed timestamps are added.
    NOTE: same signature as read_public_dataset
    Args:
        data_dir (str, optional): Directory containing the CSV files. Defaults to 'data'.
    """
    dir_path = os.path.join(path, dataset)
    logging.info(f"Loading {dataset}, from {dir_path} Dataset")

    # if files exist only select the file names existed in files list, other wise load all csv files
    if files:
        # files doesn't end with csv, add csv to the end 
        data_files = [f if f.endswith('.csv') else f + '.csv' for f in files]
    else:
        data_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

    logging.debug(f"Found {len(data_files)} file, {data_files}") 
    dfs = {}
    

    for file in data_files:
        df_name = file.replace('.csv', '').replace('-', '_')
        file_path = os.path.join(dir_path, file)
        
        logging.info(f"Loading {file} into dfs['{df_name}', {file_path}]")
        if custom_format:
            try: 
                dataset_dict = custom_format['dataset_name'] 
                index_col = dataset_dict['index_col']
                parse_dates = dataset_dict['date_cols']
                df = load_csv(dir_path, filename=file, parse_dates=parse_dates, index_col=index_col)
                logging.info(f"✓ Loaded {file} with custom format")
            except:
                # error message
                logging.error(f"Failed to load {file} with custom format")
            try:
                # --- Ensure the index is timezone-aware UTC ---
                if df.index.tz is None:
                    # Localize the naive timestamp index to UTC
                    df.index = df.index.tz_localize('UTC')
                    logging.info(f"✓ Localized naive index to UTC for {df_name}")
                elif target_tz:
                    # If already timezone-aware, convert it to the target timezone
                    df.index = df.index.tz_convert(target_tz)
                    logging.info(f"✓ Converted existing index to {target_tz} for {df_name}")
                else:
                    # If already timezone-aware, convert it to UTC to standardize
                    df.index = df.index.tz_convert('UTC')
                    logging.info(f"✓ Converted existing index to UTC for {df_name}")
                
            except Exception as e:
                logging.error(f"Could not process file {file}: {e}")
                
        else:
            try:
                df = load_csv(dir_path, filename=file)
                logging.info(f"✓ Loaded {file} with default format")
            except:
                # error message
                logging.error(f"Failed to load {file} with default format")
            

            # --- ADD: Add the 'PST_Time' column ---
            #df['PST_Time'] = df.index.tz_convert('America/Los_Angeles')
            #print(f"✓ Added 'PST_Time' column ({target_tz}) to {file}")

            if df_name == target_file_name :
                df = _rename_columns(df, target_name=target_name, timestamp_name=timestamp_name, possible_timestamp_names=possible_timestamp_names, possible_target_names=possible_target_names)
            else:
                df = _rename_columns(df, target_name=None, timestamp_name=timestamp_name, possible_timestamp_names=possible_timestamp_names)

            if set_index_timestamps:
                df = df.set_index(timestamp_name)
            # for each df in dfs i need to include df, and list(set(df.columns).difference(set(['time','y'])))
            columns_to_drop = []
            if drop_columns:
                for col in drop_columns:
                    if col in df.columns:
                        columns_to_drop.append(col)
                df = df.drop(columns=columns_to_drop)
            dfs[df_name] = (df, list(set(df.columns).difference(set(['time','y']))))
    
            for i, f in enumerate(dfs.keys()):
                logging.info(f"✓ File {i}: {f}, shape: {dfs[f][0].shape}, columns: {dfs[f][1]}") 
            
    return dfs



