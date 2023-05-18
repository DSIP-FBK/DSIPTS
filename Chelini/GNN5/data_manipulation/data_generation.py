import os
import pandas as pd
import numpy as np
import argparse
import pickle
import json
import torch

from sklearn.preprocessing import MinMaxScaler
from configparser import ConfigParser
from datetime import datetime  
from torch.utils.data import Dataset
from tqdm import tqdm


split = {'start_train': datetime(2017,11,15,0), 
        'end_train':datetime(2020,1,1,0),
        'start_validation':datetime(2020,1,1,0), 
        'end_validation':datetime(2020,12,1,0),
        'start_test': datetime(2021,3,5,0), 
        'end_test':datetime(2022,4,6,0)}

class MyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                 h_pass:int,
                 h_future:int, 
                 config: ConfigParser, 
                 columns:list,
                 task: str):
        path_date = os.path.join(config['paths']['dataset'], f"dates_dataset_{task}.csv")
        
        self.x = []
        self.y = []
        self.date = []
        for i in tqdm(range(len(df.date)-(h_future+h_pass)), desc = f"df for {task}" ):
            t_init = df.date.values[i]
            t_end = df.date.values[i+h_pass+h_future]
            if np.timedelta64(t_end-t_init, 'h').astype(int) == (h_future+h_pass):
                x = torch.tensor(df[columns].values[i:i+h_pass+h_future])
                y = torch.tensor(df.y.values[i+h_pass:i+h_pass+h_future])
                x[h_pass:, -1] = torch.tensor(0)
                self.x.append(x)
                self.y.append(y)
                self.date.append(df.date.values[i+h_pass:i+h_pass+h_future])
                
        self.date = pd.DataFrame(self.date)
        self.date.columns = range(1,h_future+1,1)
        self.date.to_csv(path_date)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idxs):
        return self.x[idxs], self.y[idxs]
    
    def getdate(self, idxs):
        return self.date.values[idxs]

def crete_dataframe(config_path: str):

    config = ConfigParser()
    config.read(config_path)
    
    
    ##### creo tutte le directory necessarie al config #########
    for path in config['paths']:
        os.makedirs(config['paths'][path], exist_ok = os.path.exists(config['paths'][path]))
    ############################################################

    print(f'{" Download the raw data ":=^60s}') 
    experiment_data = os.path.join(config['paths']['data'], 'data.pkl')
    with open(experiment_data, 'rb') as f:
        df_el,col_el,df_term,col_term,le,max_id, meteo_columns = pickle.load(f)
        df_el.tempo = pd.to_datetime(df_el.tempo)    
    print(f'{" Download complete ":=^60s}') 
    

    ##### variabili di riferimento del dataset ####
    categorical = json.loads(config.get('dataset','categorical'))
    non_categorical = json.loads(config.get('dataset','non_categorical'))
    columns =  categorical + non_categorical
    h_future = config.getint('model','h_future')
    h_pass = config.getint('model','h_pass')
    ###############################################

    
    ###### Pulizia del dataset ####################
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
    ###############################################


    ####### splitting the dataset #################
    print(f'{" Creating the dataset ":=^60s}') 
    # qua creo i vettori che vanno ad indicare quali sono i nodi da allenare e quale no 
    d = {}
    pd.options.mode.chained_assignment = None  # default='warn'
    for step in ['train','validation','test']:
        d[step] = df[(df.date>split[f'start_{step}']) & (df.date<split[f'end_{step}'])]

    scaler = MinMaxScaler()
    scaler.fit(d['train'][non_categorical].values)
    scaler_path = os.path.join(config['paths']['dataset'], f"scaler.pkl")

    if os.path.exists(scaler_path)!= True:
        with open(scaler_path, 'wb') as f :
            pickle.dump(scaler,f)
    for step in d.keys():
        d[step][non_categorical] = scaler.transform(d[step][non_categorical].values)


    ############################ CREO I VARI DATASET PER OGNI TASK  ##########################
    ds = {}
    date = {}
    for key in d.keys():
        ds[key] = MyDataset(df = d[key], 
                            h_pass = h_pass, 
                            h_future = h_future,
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
        n_cat = len(df_el[col].unique())
        emb[col]=(len(df_el[col].unique()), min(600, round(1.6 * n_cat**0.56)))

    path_emb = os.path.join(config['paths']['dataset'], f"embedding_setting.pkl")    

    with open(path_emb, 'wb') as f :
        pickle.dump(emb, f)
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