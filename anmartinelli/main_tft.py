import torch
from omegaconf import OmegaConf
from model import Model
import os
import argparse
from dataloading import dataloading
from conf_dicts import dictConfiguration
import pickle as pkl
from sklearn.preprocessing import StandardScaler

def main(cluster: bool, run_train: bool, conf_params:dictConfiguration):
    # main used for training and inference
    # todo: only inference_script.py for only inference!
    
    #* Set path to get data and name to save the model 
    if cluster:
        conf = OmegaConf.load('/storage/DSIP/edison/anmartinelli/config.yaml')
        path_data = conf.cl.data
        path_folder_works = conf.cl.folder_works
    else:
        conf = OmegaConf.load('/home/andrea/timeseries/anmartinelli/config.yaml')
        path_data = conf.loc.data
        path_folder_works = conf.loc.folder_works
    path_work_model = path_folder_works + conf_params.name + '/'
    #ex: '/home/andrea/timeseries/anmartinelli/works/Mix_Prec/'
    if not os.path.exists(path_work_model):
        os.makedirs(path_work_model)
    # path for MODEL FILES (ADD ONLY .PKL, .PT, .PNG)
    saving_path_model_name = path_work_model + conf_params.name


    #* SELECT MODEL
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(conf_params, device, saving_path_model_name)
    print(f'MODEL TRAINABLE PARAMETERS: < {model.count_parameters()} >')
    
    #* LOAD DATA
    scaler_type = StandardScaler()
    train_dl, val_dl, test_dl, scaler_y = dataloading(batch_size=conf_params.train_dict['bs'], 
                                                      batch_size_test = conf_params.test_dict['bs_t'], 
                                                      seq_len = conf_params.model_dict['seq_len'],
                                                      lag = conf_params.model_dict['lag'],
                                                      hour_learning = conf_params.train_dict['hour'], 
                                                      hour_inference = conf_params.test_dict['hour_test'], 
                                                      train_bool = True,
                                                      scaler_y = scaler_type,
                                                      step = 1,
                                                      path = path_data)
    
    #* Get training tools
    optimizer = conf_params.get_optim(model)
    loss_fun = conf_params.get_loss_fun()
    scheduler = conf_params.get_scheduler(optimizer)
    
    # import pdb
    # pdb.set_trace()
    if run_train:
        with open(saving_path_model_name + '.pkl', 'wb') as f: #! PATH for PKL
            pkl.dump([conf_params, 'no_trained_yet'], f)
            f.close()
        #* TRAINING
        model.learning(epochs = conf_params.train_dict['epochs'], 
                    dl_training = train_dl, 
                    dl_validation = val_dl, 
                    cost_func = loss_fun, 
                    optimizer= optimizer, 
                    scheduler = scheduler)
    
    #* LEARNING
    model.inference(dl = test_dl, 
                    scaler = scaler_y)
    
def get_model(conf_params: dict, device, path: str) -> Model:
    model = Model(use_target_past = conf_params.strategy_dict['use_target_past'],
                use_yprec = conf_params.strategy_dict['use_yprec'],
                n_cat_var = conf_params.model_dict['n_cat_var'],
                n_target_var = conf_params.model_dict['n_target_var'],
                seq_len = conf_params.model_dict['seq_len'],
                lag = conf_params.model_dict['lag'],
                d_model = conf_params.model_dict['d_model'],
                n_enc = conf_params.model_dict['n_enc_layers'],
                n_dec = conf_params.model_dict['n_dec_layers'],
                head_size = conf_params.model_dict['head_size'],
                num_heads = conf_params.model_dict['num_heads'],
                fw_exp = conf_params.model_dict['fw_exp'],
                dropout = conf_params.model_dict['dropout'],
                num_lstm_layers = conf_params.model_dict['num_lstm_layers'],
                device = device,
                quantiles = conf_params.strategy_dict['quantiles'],
                path_model_save = path,
                model_name = conf_params.name).to(device)
    return model
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Transformer Model")
    parser.add_argument("-cl", "--cluster", action="store_true", help="Flag for scripts running on cluster or not")
    parser.add_argument("-t", "--train", action="store_true", help="Flag for scripts running on cluster or not")
    parser.add_argument("-m", "--model_name", type=str, default='default0', help='String to set the name of the model, must be unique in works/ to avoid overwriting')
    args = parser.parse_args()
    if args.train:
        dict = dictConfiguration(args.model_name)
        main(args.cluster, args.train, dict)

    else:
        if args.cluster:
            conf = OmegaConf.load('/storage/DSIP/edison/anmartinelli/config.yaml')
            path_data = conf.cl.data
            path_folder_works = conf.cl.folder_works
        else:
            conf = OmegaConf.load('/home/andrea/timeseries/anmartinelli/config.yaml')
            path_data = conf.loc.data
            path_folder_works = conf.loc.folder_works

        path_work_model = path_folder_works + args.model_name + '/'
        #ex: '/home/andrea/timeseries/anmartinelli/works/Mix_Prec/'
        if not os.path.exists(path_work_model):
            print(f'Wrong model name, check it carefully')
        # path for MODEL FILES (ADD ONLY .PKL, .PT, .PNG)
        saving_path_model_name = path_work_model + args.model_name
        
        with open(saving_path_model_name + '.pkl', 'rb') as f: #! PATH for PKL
            dict, _  = pkl.load(f)
            f.close()
        main(args.cluster, args.train, dict)
        
