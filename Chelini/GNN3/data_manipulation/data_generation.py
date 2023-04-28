import os
import pandas as pd
import argparse
import pickle
import torch
import numpy as np
import json

from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from tqdm import tqdm
from configparser import ConfigParser
from datetime import datetime  
  
split = {'start_train': datetime(2017,11,15,0), 
        'end_train':datetime(2020,1,1,0),
        'start_validation':datetime(2020,1,1,0), 
        'end_validation':datetime(2020,12,1,0),
        'start_test': datetime(2021,3,5,0), 
        'end_test':datetime(2022,4,6,0)}

def get_concentration_matrix(df: pd.DataFrame, h_pass:int, h_future:int):
    d = {}
    
    i = 0
    for i in tqdm(range(len(df.date)-(h_future+h_pass))):
        t_init = df.date.values[i]
        t_end = df.date.values[i+h_pass+h_future]
        if np.timedelta64(t_end-t_init, 'h').astype(int) == (h_future+h_pass):
            d[i] = df.y.values[i:i+h_pass+h_future]
        i += 1
    df_tmp = pd.DataFrame(d)
    df_tmp.reset_index(drop = True, inplace=True)
    df_tmp = df_tmp.T
    Sigma = df_tmp.cov()
    Theta = np.linalg.inv(Sigma)
    Theta[np.where(abs(Theta)<0.000002)] = 0
    return Theta


def get_data(df: pd.DataFrame,
             h_pass:int,
             h_future:int, 
             edges: torch.tensor,
             config: ConfigParser, 
             columns:list,
             task: str):
    path_dataset = os.path.join(config['paths']['dataset'], f"dataset_features_{task}.pkl")
    path_date = os.path.join(config['paths']['dataset'], f"dates_dataset_{task}.csv")
    if (os.path.exists(path_dataset))&(os.path.exists(path_date)):
        with open(path_dataset, 'rb') as f :
            data = pickle.load(f)
        date = pd.read_csv(path_date, index_col=0)
        return data, date
    else:
        l = []
        date = []
        for i in tqdm(range(len(df.date)-(h_future+h_pass)), desc = f"df for {task}" ):
            t_init = df.date.values[i]
            t_end = df.date.values[i+h_pass+h_future]
            if np.timedelta64(t_end-t_init, 'h').astype(int) == (h_future+h_pass):
                x = torch.tensor(df[columns].values[i:i+h_pass+h_future])
                y = torch.tensor(df.y.values[i+h_pass:i+h_pass+h_future]).reshape(1, -1)
                x[h_pass:, -1] = torch.tensor(0)
                data = Data(x = x,
                            y = y,
                            edge_index = edges)
                l.append(data)
                date.append(df.date.values[i+h_pass:i+h_pass+h_future])
        date = pd.DataFrame(date)
        date.columns = range(1,h_future+1,1)
        with open(path_dataset, 'wb') as f :
            pickle.dump(l, f)
        date.to_csv(path_date)
        return l, date

def crete_dataframe(config_path: str):
    config = ConfigParser()
    config.read(config_path)
    # creo tutte le directory che il config ha
    for path in config['paths']:
        os.makedirs(config['paths'][path], exist_ok = os.path.exists(config['paths'][path]))
    
    print(f'{" Download the raw data ":=^60s}') 
    pd.options.mode.chained_assignment = None  # default='warn'
    experiment_data = os.path.join(config['paths']['data'], 'data.pkl')
    with open(experiment_data, 'rb') as f:
        df_el,col_el,df_term,col_term,le,max_id, meteo_columns = pickle.load(f)
        df_el.tempo = pd.to_datetime(df_el.tempo)    
    print(f'{" Download complete ":=^60s}') 
    
    categorical = json.loads(config.get('dataset','categorical'))
    non_categorical = json.loads(config.get('dataset','non_categorical'))
    
    columns =  categorical + non_categorical
    df_el = df_el[['data'] +columns]
    df = df_el[['data'] +columns]
    df = df.fillna(-1)
    df = df.drop(list(df[df.y<=0].index))
    df.ora = df.ora.astype(int)
    df['date'] = df.data.astype(str)+' ' + df.ora.astype(str)
    df['date'] = list(map(lambda x : datetime.strptime(x, '%Y-%m-%d %H'), df['date']))
    df = df.sort_values(by = ['date'])
    df = df[['date'] + columns]
    df = df.drop([0,1]).reset_index(drop=True)

    print(f'{" Creating the edges ":=^60s}') 
    h_future = config.getint('model','h_future')
    h_pass = config.getint('model','h_pass')


    if config.getboolean('model', 'correlation'):
        ds_train = df[(df.date>split['start_train'])&(df.date<split['end_train'])]
        Theta = get_concentration_matrix(df = ds_train, 
                                        h_future = h_future, 
                                        h_pass = h_pass)
        rows, cols = np.where(Theta != 0)
        edges = torch.tensor([rows.tolist(), cols.tolist()])
    else:
        A = np.triu(np.ones((200,200)))-np.diag(np.ones(200))
        rows, cols = np.where(A != 0)
        edges = torch.tensor([rows.tolist(), cols.tolist()])
    

    with open(os.path.join(config['paths']['dataset'], 'edges.pkl'), 'wb') as f:
        pickle.dump(edges, f)
    print(f'{" Creation complete ":=^60s}')

    print(f'{" Creating the dataset ":=^60s}') 
    # qua creo i vettori che vanno ad indicare quali sono i nodi da allenare e quale no 
    d = {}
    pd.options.mode.chained_assignment = None  # default='warn'
    for step in ['train','validation','test']:
        d[step] = df[(df.date>split[f'start_{step}'])&(df.date<split[f'end_{step}'])]
    scaler = MinMaxScaler()
    scaler.fit(d['train'][non_categorical].values)
    scaler_path = os.path.join(config['paths']['dataset'], f"scaler.pkl")
    if os.path.exists(scaler_path)!= True:
        with open(scaler_path, 'wb') as f :
            pickle.dump(scaler,f)
    for step in d.keys():
        d[step].loc[:, non_categorical] = scaler.transform(d[step][non_categorical].values)


    ############################ CREO I VARI DATASET PER OGNI TASK  ##########################
    row, col = np.where(A!=0)
    edges = torch.tensor([row.tolist(),col.tolist()])
    ds = {}
    date = {}
    for key in d.keys():
        ds[key], date[key] = get_data(df = d[key], 
                                        h_pass = h_pass, 
                                        h_future = h_future,
                                        edges = edges, 
                                        config = config,
                                        columns=columns,
                                        task = key)

    path_dataset = os.path.join(config['paths']['dataset'], f"dataframes.pkl")    
    path_date = os.path.join(config['paths']['dataset'], f"dates.pkl")    
    with open(path_dataset, 'wb') as f :
        pickle.dump(ds, f)
        
    with open(path_date, 'wb') as f :
        pickle.dump(date, f)

    ################## DECIDO COME FARE EMBEDDING DELLE VARIABILI NON CATEGORICHE ############
    emb = {}
    for col in categorical:
        if col != "is_low":
            emb[col]=(len(df_el[col].unique()), len(df_el[col].unique())//3)
        else:
            emb[col]=(len(df_el[col].unique()), len(df_el[col].unique()))
    emb_path = os.path.join(config['paths']['dataset'], f"embedding_setting.pkl")
    if os.path.exists(emb_path)!= True:
        with open(emb_path, 'wb') as f :
            pickle.dump(emb,f)
    print(f'{" Creation complete ":=^60s}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation")
    parser.add_argument("-c", "--config", 
                        type=str, 
                        required=True, 
                        help="It storse the path of the config file")
    args = parser.parse_args()
    
    crete_dataframe(args.config)
#166