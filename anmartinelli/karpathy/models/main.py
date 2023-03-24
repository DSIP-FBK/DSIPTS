import torch
from omegaconf import OmegaConf
from model import Base_Model
from model_tft import Full_Model
import os
import argparse
from dataloading import dataloading

def main(cluster, train, mix, prec, tft, model_str, bs_test, hour_test):
    #TODO: UNPACK -mod / SELECT MODEL / ADAPT PATH / LOAD DATA / LEARNING / INFERENCE

    #* UNPACK -mod
    model_split = model_str.split('_')
    seq_len = int(model_split[0])
    lag = int(model_split[1])
    n_enc = int(model_split[2])
    n_dec = int(model_split[3])
    n_embd = int(model_split[4])
    num_heads = int(model_split[5])
    head_size = int(model_split[6])
    fw_exp = int(model_split[7])
    dropout = float(model_split[8])
    lr = float(model_split[9])
    wd = float(model_split[10])
    epochs = int(model_split[11])
    bs = int(model_split[12])
    hour = int(model_split[13])
    sched_step = int(model_split[14])

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print('-'*50+'\n')
    print(f'> {model_str} UNPACKED')

    #* SELECT MODEL
    # handle with
    #       -m   (MIX) : mixing x and y during embed process or maintaining them separated during encoder-selfAtt
    #       -p   (PREC): allow for Decoder embeddinged to know y_{\tau-1} 
    #       -tft (TFT) : use variable selection taken from the tft paper
    if tft:
        model = Full_Model(mix, prec, tft, seq_len, lag, n_enc, n_dec, n_embd, 
                       num_heads, head_size, fw_exp, dropout, device).to(device)
    else:
        model = Base_Model(mix, prec, tft, seq_len, lag, n_enc, n_dec, n_embd, 
                       num_heads, head_size, fw_exp, dropout, device).to(device)
    #only to count parameters and compare different structures
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('-'*50+'\n')
    print(f'> MODEL SELECTED')
    print(f'Model trainable params: {count_parameters(model)}')
    
    #* ADAPT PATH
    # cluster: /storage/DSIP/edison/anmartinelli/ with:
    #                                            models/(main, models, config)
    #                                            works/(diretories for each work)
    # local: /home/andrea/timeseries/anmartinelli/karpathy/ 
    #                                                      same structure
    if cluster:
        conf = OmegaConf.load('/storage/DSIP/edison/anmartinelli/models/config.yaml')
        path_data = conf.cl.data
        # path_folder_models = conf.cl.folder_models
        path_folder_works = conf.cl.folder_works
        # init path_work_model and check model folder
        path_work_model = path_folder_works+model_str+'/'
        if os.path.exists(path_work_model):
            if train:
                print(f'Retraining {path_work_model}')
            else:
                print(f'Model {model_str} found')
        else:
            if train:
                os.makedirs(path_work_model)
                print(f'Directory {path_work_model} created')
            else:
                print(f'Train mod off & {model_str} not found')
                print('Check if the model string is correct')
                print('Check if the model must be trained')
                print('-- RETURN --')
                return
    else:
        conf = OmegaConf.load('/home/andrea/timeseries/anmartinelli/karpathy/models/config.yaml')
        path_data = conf.loc.data
        # path_folder_models = conf.loc.folder_models
        path_folder_works = conf.loc.folder_works
        # check model folder
        path_work_model = path_folder_works+model_str+'/'
        if os.path.exists(path_work_model):
            if train:
                print(f'Retraining {path_work_model}')
            else:
                print(f'Model {model_str} found')
        else:
            if train:
                os.makedirs(path_work_model)
                print(f'Directory {path_work_model} created')
            else:
                print(f'Train mod off & {model_str} not found')
                print('Check if the model string is correct')
                print('Check if the model must be trained')
                print('-- RETURN --')
                return
    print('-'*50+'\n')
    print(f'> PATH ADAPTED')

    #* LOAD DATA
    if train:
        train_dl, val_dl, test_dl, scaler_y = dataloading(batch_size=bs, batch_size_test=bs_test, 
                                                        seq_len=seq_len, lag=lag,
                                                        hour_learning=hour, 
                                                        hour_inference=hour_test, 
                                                        train_bool=train,
                                                        path=path_data)
    else:
        _, _, test_dl, scaler_y = dataloading(batch_size=bs, batch_size_test=bs_test, 
                                                        seq_len=seq_len, lag=lag,
                                                        hour_learning=hour, 
                                                        hour_inference=hour_test, 
                                                        train_bool=train,
                                                        path=path_data)
    print('-'*50+'\n')
    print(f'> DATA LOADED')

    #* LEARNING
    # only if train = True, i.e. -t in parser

    # LEARNING PARAMS
    if train:
        path_model = path_work_model + model_str
        cost_function = torch.nn.L1Loss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        print('-'*50+'\n')
        print(f'> LEARNING START')
        model.learning(path_model, epochs, train_dl, val_dl, cost_function, optimizer, sched_step)
        print(f'> LEARNING END')

    #* INFERENCE
    # two different inference processes according to the usage or not of y_{\tau-1}

    # INFERENCE PARAMS
    path_model = path_work_model + model_str
    dl = test_dl
    scaler = scaler_y
    print('-'*50+'\n')
    print(f'> INFERENCE START')
    model.inference(path_model, dl, scaler)
    print(f'> INFERENCE DONE')


if __name__ == '__main__':

    ## PARSER
    parser = argparse.ArgumentParser(description="Transformer Model")

    # goal of the program
    parser.add_argument("-cl", "--cluster", action="store_true", help="Flag for scripts running on cluster or not")
    parser.add_argument("-t", "--train", action="store_true", help="Run also training or not")
    parser.add_argument("-m", "--mixed", action="store_true", help="Model with mixed x and y or not")
    parser.add_argument("-p", "--prec", action="store_true", help="Model that knows y in t-1 or not")
    parser.add_argument("-tft", "--tft", action="store_true", help="Model with variable selection as tft or not")

    # which model and testing
    parser.add_argument("-mod", "--model_str", type=str, default='265_65_2_2_8_2_4_2_0.2_1e-04_0.00_2_16_24_100', help='String to fetch the model: SeqLen_Lag_Enc_Dec_Embd_Head_HeadSize_FwExp_Dropout_LR_WD_E_BS_Hour_SchedStep')
        #  -mod SeqLen_Lag_Enc_Dec_Embd_Head_HeadSize_FwExp_Dropout_LR_WD_E_BS_Hour_SchedStep
        #  -mod 256_60_2_2_16_4_4_3_0.1_1e-05_0.01_500_64_24_200
        #  -mod 256_65_2_2_8_2_4_2_0.2_1e-04_0.00_2_16_24_100
    parser.add_argument("-hr_t", "--hour_test", default=24, type=int, help="Hour to start the training/prediction (24 = all)")
    parser.add_argument("-bs_t", "--batch_size_test", default=2, type=int, help="Number of batch compiled simultaneously ON TEST SET")

    args = parser.parse_args()
    print('-'*50+'\n')
    print('> DEVICE:')
    if args.cluster:
        print(f'   - Cluster')
    else:
        print(f'   - Local')
    print('> AIMS:')
    print(f'   - Also Train = {args.train}')
    print('> MODEL:')
    print(f'   - Mixing x and y = {args.mixed}')
    print(f'   - Know Y in t-1 = {args.prec}')
    print(f'   - Variable selection(TFT) = {args.tft}')

    print('DAJE ROMAAAA')

    main(cluster=args.cluster, train=args.train, mix=args.mixed, prec=args.prec, tft=args.tft,
         model_str=args.model_str, bs_test=args.batch_size_test, hour_test=args.hour_test)
