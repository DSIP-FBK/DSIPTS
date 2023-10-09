import pandas as pd
import os
import numpy as np
from typing import List, Tuple
import logging
import requests
from bs4 import BeautifulSoup as bs

def build_venice(path:str,url='https://www.comune.venezia.it/it/content/archivio-storico-livello-marea-venezia-1')->None:
    
      
    with requests.Session() as s:
        r = s.get(url)
    soup = bs(r.content)

    print('CARE THE STRUCTURE OF THE SITE CAN BE CHANGED')

    def cast_string(x):
        if np.isfinite(x) is False:
            return x
        if x<10:
            return f'0{int(x)}:00'
        else:
            return f'{int(x)}:00'
        
    def cast_month(x):
        try:
            return x.replace('gen','01').replace('feb','02').replace('mar','03').replace('apr','04').replace('mag','05').replace('giu','06').replace('lug','07').replace('ago','08').replace('set','09').replace('ott','10').replace('nov','11').replace('dic','12')
        except:
            return x
        
    def remove_float(table,column):
        if table[column].dtype in [int,float]:
            table[column] = table[column].apply(lambda x:cast_string(x))
        else:
            pass
        
    def remove_str(table,column):
        table[column] = table[column].apply(lambda x:cast_month(x))
        
    def normalize(table):
        columns = table.columns
        if 'Data_ora(solare)' in columns:
            table['time'] = table['Data_ora(solare)'] 
        
        elif 'GIORNO' in columns and 'ORA solare' in columns:
            remove_float(table,'ORA solare')
            table['time'] = table['GIORNO'] +' '+ table['ORA solare'] 
        
        elif 'data' in columns and 'ora solare' in columns:
            remove_float(table,'ora solare')
            table['time'] =table['data'] +' '+ table['ora solare'] 
        
        elif 'Data' in columns and 'Ora solare' in columns:
            remove_str(table,'Data')
            remove_float(table,'Ora solare')
            table['time'] = table['Data'] +' '+ table['Ora solare'] 
        else:
            import pdb
            pdb.set_trace()
       
        for c in columns:
            if 'Salute' in c:
                table['y'] = table[c].values
                if 'cm' in c:
                    table['y']/=100
        res = table[['time','y']].dropna()
        res['time'] = pd.to_datetime(res['time'])
        return res
    tot= []
    for row in soup.find_all("table")[1].find('tbody').find_all('tr'):
        for i,column in enumerate(row.find_all('td')):
            tmp_links = column.find_all('a')
            if len(tmp_links)>0:
                for x in tmp_links:
                    if 'orari' in x['href']:
                        tmp =  pd.read_csv('https://www.comune.venezia.it/'+x['href'],sep=';', parse_dates=True)
                        tot.append(normalize(tmp))
    
    res = pd.concat(tot)
    res.sort_values(by='time',inplace=True)
    res.to_csv(f'{path}/venice.csv',index=False)


def read_public_dataset(path:str,dataset:str)->Tuple[pd.DataFrame,List[str]]:
    """    Returns the public dataset chosen. Pleas download the dataset from here https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy or ask to agobbi@fbk.eu. 
    Extract the data and leave the name all_six_datasets in the path folder

    Args:
        path (str): path to data
        dataset (str): dataset (one of 'electricity','etth1','etth2','ettm1','ettm2','exchange_rate','illness','traffic','weather')

    Returns:
        Tuple[pd.DataFrame,List[str]]: The target variable is *y* and the time index is *time* and the list of the covariates
    """
    
    

    if os.path.isdir(path):
        pass
    else:
        logging.info('I will try to create the folder')
        os.mkdir(path)
        
    files = os.listdir(path)
    if 'all_six_datasets' in files:
        pass
    else:
        logging.error('Please dowload the zip file form here and unzip it https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy')
        return None,None
    
    
    if dataset not in ['electricity','etth1','etth2','ettm1','ettm2','exchange_rate','illness','traffic','weather','venice']:
        logging.error(f"Dataset {dataset} not available, use one among ['electricity','etth1','etth2','ettm1','ettm2','exchange_rate','illness','traffic','weather','venice']")
        return None,None

    if dataset=='electricity':
        dataset = pd.read_csv(os.path.join(path,'all_six_datasets/electricity/electricity.csv'),sep=',',na_values=-9999)
    elif dataset=='etth1':
        dataset = pd.read_csv(os.path.join(path,'all_six_datasets/ETT-small/ETTh1.csv'),sep=',',na_values=-9999)  
    elif dataset=='etth1':
        dataset = pd.read_csv(os.path.join(path,'all_six_datasets/ETT-small/ETTh2.csv'),sep=',',na_values=-9999)
    elif dataset=='ettm1':
        dataset = pd.read_csv(os.path.join(path,'all_six_datasets/ETT-small/ETTm1.csv'),sep=',',na_values=-9999)
    elif dataset=='ettm2':
        dataset = pd.read_csv(os.path.join(path,'all_six_datasets/ETT-small/ETTm2.csv'),sep=',',na_values=-9999)
    elif dataset=='exchange_rate':
        dataset = pd.read_csv(os.path.join(path,'all_six_datasets/exchange_rate/exchange_rate.csv'),sep=',',na_values=-9999)
    elif dataset=='exchange_rate':
        dataset = pd.read_csv(os.path.join(path,'all_six_datasets/illness/national_illness.csv'),sep=',',na_values=-9999)
    elif dataset=='traffic':
        dataset = pd.read_csv(os.path.join(path,'all_six_datasets/traffic/traffic.csv'),sep=',',na_values=-9999) 
    elif dataset=='weather':
        dataset = pd.read_csv(os.path.join(path,'all_six_datasets/weather/weather.csv'),sep=',',na_values=-9999) 
    elif dataset=='venice':
        if os.path.isfile(os.path.join(path,'venice.csv')):
            dataset = pd.read_csv(os.path.join(path,'venice.csv')) 
        else:
            logging.info('I WILL TRY TO DOWNLOAD IT, if there are errors please have a look to `build_venice` function')
            build_venice(path,url='https://www.comune.venezia.it/it/content/archivio-storico-livello-marea-venezia-1')
            dataset = pd.read_csv(os.path.join(path,'venice.csv')) 
    else:
        logging.error(f'Dataset {dataset} not found')
        return None, None
    dataset.rename(columns={'date':'time','OT':'y'},inplace=True)
    dataset.time = pd.to_datetime(dataset.time)
    logging.info(f'Dataset loaded with shape {dataset.shape}')
    
    return dataset, list(set(dataset.columns).difference(set(['time','y'])))