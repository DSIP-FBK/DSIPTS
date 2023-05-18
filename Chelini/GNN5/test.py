import os
import pickle
from configparser import ConfigParser
import torch
from torch.utils.data import DataLoader

from model import GAT
from inference import get_plot

def test(config: ConfigParser, name_model:str) -> None:
    print(f'{" Upload the dataframe ":=^60s}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_dataset = os.path.join(config['paths']['dataset'], f"dataframes.pkl")    
    path_emb = os.path.join(config['paths']['dataset'], f"embedding_setting.pkl")    

    with open(path_dataset, 'rb') as f :
        ds = pickle.load(f)

    with open(path_emb, 'rb') as f :
        emb = pickle.load(f)

    for key in ds.keys():
        config['dataset'][f'n_obs_{key}']=f"{len(ds[key])}"


    vars = name_model.split(sep = "_")
    config['model']['id_model'] = name_model[4:-3]
    config['model']['hidden'] = vars[1]
    config['model']['in_head'] = vars[2]
    config['model']['out_head'] = vars[3]
    config['model']['drop_out'] = vars[4][:3]
    
    params_model = {}
    for key in list(dict(config.items('model')).keys()):
        if key != "id_model":
            try:
                params_model[key] = config.getint('model', key)
            except:
                params_model[key] = config.getfloat('model', key)

    id_model = config['model']['id_model']
    # initialization 
    model = GAT(in_feat = 8, 
                hid = params_model["hidden"],
                in_head = params_model["in_head"],
                out_head = params_model["out_head"], 
                drop_out = params_model["drop_out"],
                num_layer1=params_model["num_layer1"],
                num_layer2=params_model["num_layer2"],
                hid_out_features1=params_model["hid_out_features1"],
                hid_out_features2=params_model["hid_out_features2"],
                emb = emb,
                past = 200, 
                future = 65, 
                device = device)
    model.load_state_dict(torch.load(os.path.join(config['paths']['net_weights_train'], f"{name_model}")))
    model.to(device)
    print(f'{f" configuration {id_model} ":=^60s}')