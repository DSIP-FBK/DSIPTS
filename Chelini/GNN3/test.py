import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
import sys

from model import my_Dataset, my_lstm
from train import get_prediction

def get_odps_cross(odps:list, 
                   k:int):
    d = {}
    m = len(odps)//k
    for i in range(k):
        if i != k-1:
            d[i+1] = {'validation': odps[i*m:(i+1)*m],
                      'train': list(set(odps)-set(odps[i*m:(i+1)*m]))}
        else:
            d[i+1] = {'validation': odps[i*m:],
                  'train': odps[:i*m]}
    return d

def get_score_val(model, 
                  dataloader_train, 
                  dataloader_validation, 
                  optimizer,
                  scheduler, 
                  loss_function,
                  n_fold: int,
                  epochs: int):
                
    be = np.infty
    bm = model
    pbar = tqdm(total=epochs, desc = f"training the fold number {n_fold}")
    e0=0
    for e in range(epochs):
        ##TRAIN STEP
        model, loss = get_prediction(model = model,
                                     train = True, 
                                     get_predictions= False,
                                     dataloader = dataloader_train, 
                                     loss_function = loss_function, 
                                     optimizer = optimizer)
        model, loss = get_prediction(model = model,
                                     train = False,
                                     get_predictions= False,
                                     dataloader = dataloader_validation, 
                                     loss_function = loss_function, 
                                     optimizer = optimizer)
        if e in [800,900,1000, 1300]:
            scheduler.step()
        k = 200
        if e%k==0:
            pbar.update(e-e0)
            e0 = e
            sys.stdout.flush()    
        if loss < be:
            be = loss
            bm = model
    pbar.close()
    return be, bm

def get_cross_validation(odptest_d: tuple,
                         CV_results: dict,
                         cv_key: tuple,
                         params_model: tuple,
                         df, 
                         y,
                         device, 
                         config):
        
        odps_test, d = odptest_d
        run, key = cv_key
        feat,emb,n_layers,hidden_size = params_model
        # CREO IL MODELLO COI PARAMETRI SALVATI
        for n_fold in d.keys():
            model = my_lstm(n_features = feat, 
                            n_fasi = len(df.fase.unique()), 
                            n_emb = emb, 
                            n_layers = n_layers, 
                            n_hidden = hidden_size,
                            device = device)
            model.to(device)
            optimizer = optim.Adam(params = model.parameters(),lr=0.001)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.5)
            
            dataset_train = my_Dataset(df[df.odp.isin(d[n_fold]['train'])], y)
            dataset_validation = my_Dataset(df[df.odp.isin(d[n_fold]['validation'])], y)
            dataset_test = my_Dataset(df[df.odp.isin(odps_test)], y)

            dataloader_train = DataLoader(dataset_train, batch_size = 1, shuffle = True, drop_last = True)
            dataloader_validation = DataLoader(dataset_validation, batch_size = 1, shuffle = True, drop_last = True)
            dataloader_test = DataLoader(dataset_test, batch_size = 1, shuffle = True, drop_last = True)

            dataloaders = {
                'train': dataloader_train,
                'validation': dataloader_validation,
                'test': dataloader_test,
            }
            # questo if serve a dirmi a che punto della cross validation, per un determinato modello, sono arrivato
            if n_fold not in CV_results[run][key].keys():
                # semplicemente alleno il modello con quei deteminati parametri e estrapolo il miglior risultato in validation
                be, model  = get_score_val(model = model, 
                                            dataloader_train = dataloader_train, 
                                            dataloader_validation = dataloader_validation, 
                                            optimizer = optimizer,
                                            scheduler = scheduler,
                                            loss_function = nn.L1Loss(),
                                            n_fold = n_fold,
                                            epochs = config['model']['epochs'])
                # QUA CI STA LA PARTE IN CUI CREO IL DATAFRAME
                # part in ['train','validation','test']
                out = pd.DataFrame()
                for part in dataloaders.keys():
                    tmp = get_prediction(model = model, 
                                         train = False, 
                                         get_predictions= True, 
                                         dataloader=dataloaders[part], 
                                         loss_function=nn.L1Loss(), 
                                         optimizer=None)
                    tmp['type'] = part
                    out = pd.concat([out,tmp], axis=0)

                out['run'] = run
                out['n_fold'] = n_fold
                out['is_enriched'] = config['model']['enriched']
                out['is_reduced'] = True if len(key)==4 else False
                out.reset_index(drop = True, inplace = True)
                path = os.path.join(config['paths']['cross_validation'],  f"{key} params {params_model}", f'run{run}')
                os.makedirs(path, exist_ok=os.path.exists(path))
                

                out.to_csv(os.path.join(path, f'run{run}_fold{n_fold}.csv'))

                # SALVO QUANTO OTTENUTO
                # #RUN
                # (enriched, reduced)
                # #fold 
                CV_results[run][key][n_fold] = be

                with open(os.path.join(config['paths']['prediction_dir'], 'cross_val.pkl'), "wb") as f:
                    pickle.dump(CV_results,f)  
def CV(key:tuple,
        df_enriched:pd.DataFrame,
        df_not_enriched:pd.DataFrame,
        best_models:dict,
        odps_cross:list,
        d: dict,
        run:int,
        CV_results: dict,
        y,
        device, 
        config):

    en, reduced = key

    # SETTO I VARI PARAMETRI IN FUNZIONE DEI DUE VALORI PRECEDENTI
    df_cv = df_enriched if en == "enriched" else df_not_enriched
    df_cv = df_cv[[x for x in df_cv.columns if "temp" in x]+ ["odp", "fase"]] if reduced == "reduced" else df_cv
    # params_model = feat,emb,n_layers,hidden_size
    params_model = best_models[key] if reduced == "reduced" else (len(df_cv.drop(columns = ['fase', 'odp']).columns),*best_models[key])
    config['model']['enriched'] = True if en == "enriched" else False

    print(str(f" {en} and {reduced} with {best_models[key]} params").center(60, "=")) 
    
    # Questo dice che una run non è mai stata fatta
    if run not in CV_results.keys():
        CV_results[run] = {} 

    # Questo dice se un modello ha mai visto una determinata run
    if key not in CV_results[run].keys():
        # significa che non ho mai visto questo modello e che quindi non esiste ancora un dizionario in cui ho salvato qualche risultato
        # per questo motivo devo crearlo 
        CV_results[run][key] = {}

    # Per selezionarie gli odp di test basta inserire come chiave dentro "runs" a che run siamo
    get_cross_validation(odptest_d = (odps_cross, d),
                    CV_results = CV_results, 
                    cv_key = (run, key), 
                    params_model = params_model,
                    df = df_cv, 
                    y = y,
                    config = config,
                    device = device)
    print(pd.DataFrame(data = CV_results[run][key].items()))                         

def serial_test(runs:dict,
                CV_results:dict,
                best_models:dict,
                odps:list,
                df_enriched: pd.DataFrame,
                df_not_enriched: pd.DataFrame,
                y,
                config,
                device):
    # Per fare cross validation è necessario che una delle seguenti condizioni sia vera:
    #  1) una run non è mai stata vista 
    #  2) una run non è stata portata a termine per tutti e 4 i modelli (ciò significa che un modello puù non aver concluso tutti gli n fold)
    for run in runs.keys():            
        print(str(f" we are duning the run number {run} ").center(60, "=")) 
        # GLI ODP SU CUI FACCIO CROSS VALIDATION SONO DA RANDOMIZZARE
        odps_cross = list(set(odps)-set(runs[run]))
        random.shuffle(odps_cross)
        d = get_odps_cross(odps_cross, config['model']['n_fold'])

        for key in best_models.keys():
            CV(key = key,
                df_enriched = df_enriched,
                df_not_enriched = df_not_enriched,
                best_models = best_models,
                odps_cross = odps_cross,
                d = d,
                run = run,
                CV_results = CV_results,
                y = y,
                device = device, 
                config = config
            )
    conclusion(best_models = best_models, 
                runs = runs,
                config = config)


def parallel_test(run:int,
                  model:int,
                runs:dict,
                CV_results:dict,
                best_models:dict,
                odps:list,
                df_enriched: pd.DataFrame,
                df_not_enriched: pd.DataFrame,
                y,
                config,
                device):
        print(str(f" CV for the {run} run and the model {model} ").center(60, "=")) 
        odps_cross = list(set(odps)-set(runs[run]))
        random.shuffle(odps_cross)
        d = get_odps_cross(odps_cross, config['model']['n_fold'])
        
        if run not in CV_results.keys():
            CV_results[run] = {} 
        keys = list(best_models.keys())
        key = keys[key]
        en, reduced = key[model]

        # SETTO I VARI PARAMETRI IN FUNZIONE DEI DUE VALORI PRECEDENTI
        df_cv = df_enriched if en == "enriched" else df_not_enriched
        df_cv = df_cv[[x for x in df_cv.columns if "temp" in x]+ ["odp", "fase"]] if reduced == "reduced" else df_cv
        # params_model = feat,emb,n_layers,hidden_size
        params_model = best_models[key] if reduced == "reduced" else (len(df_cv.drop(columns = ['fase', 'odp']).columns),*best_models[key])
        config['model']['enriched'] = True if en == "enriched" else False

        print(str(f" {en} and {reduced} with {best_models[key]} params").center(60, "=")) 
        
        # Questo dice se un modello ha mai visto una determinata run
        if key not in CV_results[run].keys():
            # significa che non ho mai visto questo modello e che quindi non esiste ancora un dizionario in cui ho salvato qualche risultato
            # per questo motivo devo crearlo 
            CV_results[run][key] = {}

        # Per selezionarie gli odp di test basta inserire come chiave dentro "runs" a che run siamo
        get_cross_validation(odptest_d = (runs[run], d),
                            CV_results = CV_results, 
                            cv_key = (run, key), 
                            params_model = params_model,
                            df = df_cv, 
                            y = y,
                            config = config,
                            device = device)
        print(pd.DataFrame(data = CV_results[run][key].items()))


def conclusion(best_models:dict, 
                runs: dict,
                config):
    with open(os.path.join(config['paths']['prediction_dir'], 'cross_val.pkl'), "rb") as f:
        CV_results = pickle.load(f)
    out = {}
    for key in best_models.keys():
        out[key] = [] 
    out['run'] = []
    for run in runs.keys():
        out['run'].append(run)
        for key in best_models.keys():
            out[key].append(np.mean(list(CV_results[run][key].values())))
    out = pd.DataFrame.from_dict(out)
    print(out)

    out.to_csv(os.path.join(config['paths']['cross_validation'], "results_CV.csv"))
