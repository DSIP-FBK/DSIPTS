import torch
from model import Transformer
from training import training
from dataloading import dataloading
from full_plot import full_plot
from omegaconf import OmegaConf
import os
import argparse


def main(
    cluster = True,
    which_hour_seq = 24,
    which_hour_seq_test = 12,
    train:bool = True,
    plot:bool = True,
    model_str:str = '',
    batch_size:int = 16,
    batch_size_test:int = 8,
    seq_len:int = 256,
    lag:int = 61,
    time_emb:int = 12,
    y_emb:int = 5,
    full_emb:int = 32,
    n_enc:int = 3,
    n_dec:int = 3,
    heads:int = 4,
    forward_exp:int = 6,
    dropout:float = 0.0,
    lr:float = 0.01,
    wd:float = 0.1,
    scheduler_step:int = 25,
    epochs:int = 4000
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # paths of the model
    if model_str=='':
        model_str = f'{which_hour_seq}_{batch_size}_{epochs}_{seq_len}_{lag}_{lr}_{wd}_{n_enc}_{n_dec}_{time_emb}_{y_emb}_{full_emb}_{heads}_{forward_exp}_{dropout}'
    else:
        model_str_split = model_str.split('_')
        which_hour_seq = int(model_str_split[0])
        batch_size = int(model_str_split[1])
        epochs = int(model_str_split[2])
        seq_len = int(model_str_split[3])
        lag = int(model_str_split[4])
        lr = float(model_str_split[5])
        wd = float(model_str_split[6])
        n_enc = int(model_str_split[7])
        n_dec = int(model_str_split[8])
        time_emb = int(model_str_split[9])
        y_emb = int(model_str_split[10])
        full_emb = int(model_str_split[11])
        heads = int(model_str_split[12])
        forward_exp= int(model_str_split[13])
        dropout = float(model_str_split[14])

    print('\n> MODEL: '+ model_str)
    

    if cluster:
        conf = OmegaConf.load('/storage/DSIP/edison/anmartinelli/directTsf/model_direct_tsf/config.yaml')
        
        if os.path.exists(conf.cl.save+model_str+'/'):
            print(f'Retraining {conf.cl.save+model_str}/')
        else:
            os.makedirs(conf.cl.save+model_str+'/')
            print(f'Directory created')
        
        save_result_path = conf.cl.save+model_str+'/'+conf.cl.direct+model_str
        data_path = conf.cl.data
        # print(save_result_path)
    else:
        conf = OmegaConf.load('./config.yaml')
        if os.path.exists(conf.loc.save+model_str+'/'):
            print(f'Retraining {conf.loc.save+model_str}/')
        else:
            os.makedirs(conf.loc.save+model_str+'/')
            print(f'Directory created')
        
        save_result_path = conf.loc.save+model_str+'/'+conf.loc.direct+model_str
        data_path = conf.loc.data
        # print(save_result_path)
    
    # Initialize the net and send it to device
    #* TRANSFORMER
    net=Transformer(seq_len=seq_len, lag = lag, 
                    n_enc=n_enc, n_dec=n_dec,
                    time_emb=time_emb, y_emb = y_emb, full_emb=full_emb, context_size=4, # x is [m,d,h,dow], so 5 variables as context
                    heads=heads, forward_exp=forward_exp, dropout=dropout, device=device).to(device)

    # number of params to check its size
    total_params = sum( param.numel() for param in net.parameters() )
    trainable_params = sum( p.numel() for p in net.parameters() if p.requires_grad )
    
    print(f'\t total_params     = {total_params}')
    print(f'\t trainable_params = {trainable_params}\n') 

    # Load the necessary data according to 'train' (bool var) and in case start with training: 
    #* DATA
    print('\n> Data Loading')
    # True : load all dataloaders and scaler and start the training phase
    # False : load only test_dl and scaler to evaluate
    if train:
        train_dl, val_dl, test_dl, scaler_y = dataloading(path=data_path, batch_size=batch_size, batch_size_test=batch_size_test,
        seq_len=seq_len, lag=lag, step=1, which_hour_seq=which_hour_seq, which_hour_seq_test=which_hour_seq_test, train_bool=train)
        print('\n Data Loaded\n')
        print('-'*50)
        #* TRAIN
        training(path_save = save_result_path, net=net, device=device, loader_train=train_dl, loader_val=val_dl,
                lag=lag, lr=lr, wd=wd, scheduler_step=scheduler_step, epochs=epochs)
    else:
        _, _, test_dl, scaler_y = dataloading(path=data_path, batch_size=batch_size, batch_size_test=batch_size_test,
        seq_len=seq_len, lag=lag, step=1, which_hour_seq=which_hour_seq, train_bool=train)
        print('\n Data Loaded\n')
        print('-'*50)
    if plot:
        print('> Start Plotting\n')
        full_plot(net=net, path=save_result_path, dl=test_dl, scaler=scaler_y, lag=lag, device=device)
        print('\n End Plotting\n')
        print('-'*50)

if __name__ == '__main__':

    ## PARSER
    parser = argparse.ArgumentParser(description="Transformer Model")

    # goal of the program
    parser.add_argument("-t", "--train", action="store_true", help="Run training of the nn")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot a batch of the test set with predictions")
    parser.add_argument("-cl", "--cluster", action="store_true", help="Flag for scripts running on cluster")

    parser.add_argument("-mod", "--model_str", type=str, default='', help='String to fetch the model')
    parser.add_argument("-whs", "--which_hour_seq", default=24, type=int, help="Hour to start the training/prediction (24 = all)")
    parser.add_argument("-whs_test", "--which_hour_seq_test", default=12, type=int, help="Hour to start the training/prediction (24 = all)")
    parser.add_argument("-bs", "--batch_size", default=12, type=int, help="Number of batch compiled simultaneously")
    parser.add_argument("-bs_t", "--batch_size_test", default=2, type=int, help="Number of batch compiled simultaneously ON TEST SET")
    parser.add_argument("-seq_len", "--sequence_length", default=256, type=int, help="Number of total size (context + target)")
    parser.add_argument("-lag", "--lag", default=60, type=int, help="Number of target size (add 1 to avoid 'Attention problems')")
    parser.add_argument("-xe", "--x_emb", default=4, type=int, help="Size of x embedding")
    parser.add_argument("-ye", "--y_emb", default=4, type=int, help="Size of y embedding")
    parser.add_argument("-fe", "--full_emb", default=8, type=int, help="Size of full embedding")
    parser.add_argument("-enc", "--encoder", default=8, type=int, help="Number of encoders")
    parser.add_argument("-dec", "--decoder", default=3, type=int, help="Number of decoders")
    parser.add_argument("-he", "--heads", default=3, type=int, help="Number of heads into the attention")
    parser.add_argument("-f", "--forward_expansion", default=2, type=int, help="Factor value for forward expansion into the attention")
    parser.add_argument("-d", "--dropout", default=0.0, type=float)
    parser.add_argument("-lr", "--lr", default=1e-04, type=float)
    parser.add_argument("-wd", "--wd", default=0.0, type=float)
    parser.add_argument("-s", "--scheduler_step", default=150, type=int)
    parser.add_argument("-E", "--epochs", default=500, type=int)

    args = parser.parse_args()
    print('\n> AIMS:')
    print(f'   - which_hour_seq = {args.which_hour_seq}')
    print(f'   - which_hour_seq_test = {args.which_hour_seq_test}')
    print(f'   - train = {args.train}')
    print(f'   - plot = {args.plot}')

    # conf = OmegaConf.load('config.yaml')
    # print(conf.path.save)
    # print(conf.path.save+'24_128_1000_256_60_0.0001_0.0_6_8_16_16_32_3_2_0.2/')

    # print(os.path.exists(conf.path.save+'24_128_1000_256_60_0.0001_0.0_6_8_16_16_32_3_2_0.2/'))
    # # print(os.path.exists('./directTsf/direct_direct_128_1000_256_60_0.0001_0.0_8_6_8_8_16_3_2_0.0'))

    
    # print(os.listdir(conf.locale.directTsf+'Tensorboard/models/'))
    # model_str = 'direct_all_128_1000_256_60_0.0001_0.0_8_6_8_8_16_3_2_0.0'
    # model_files = [filename for filename in os.listdir(conf.locale.directTsf+'Tensorboard/models/') 
    #             if filename.startswith("direct_direct_128_1000_256_60_0.0001_0.0_8_6_8_8_16_3_2_0.0")]
    # print(model_files)
    # for file in model_files:
    #     shutil.copy2(conf.locale.directTsf+'Tensorboard/models/'+file,
    #             conf.locale.directTsf+model_str+'/')
    # for file in os.listdir(conf.locale.directTsf+'Tensorboard/models/'):
    #     shutil.copy2(conf.locale.directTsf+'Tensorboard/models/'+file,
    #             conf.locale.directTsf+'direct_all_128_1000_256_60_0.0001_0.0_8_6_8_8_16_3_2_0.0/'+file)
    # print(os.path.abspath('./directTsf/direct_direct_128_1000_256_60_0.0001_0.0_8_6_8_8_16_3_2_0.0'))
    
    
    # if os.path.exists(conf.locale.directTsf+'/Tensorboard/models/direct_all_128_1000_256_60_0.0001_0.0_6_8_16_16_32_3_2_0.2/'):
    #     shutil.copy2('./directTsf/direct_direct_128_1000_256_60_0.0001_0.0_8_6_8_8_16_3_2_0.0',
    #                 conf.locale.directTsf+'direct_all_128_1000_256_60_0.0001_0.0_8_6_8_8_16_3_2_0.0')

    # else:
    #     os.makedirs('./directTsf/direct_all_128_1000_256_60_0.0001_0.0_8_6_8_8_16_3_2_0.0')

    #RUNNING MAIN
    main(cluster = args.cluster,
        which_hour_seq=args.which_hour_seq, which_hour_seq_test=args.which_hour_seq_test,                                                                                    
        train=args.train, plot=args.plot,                                                                                # aim bools
        model_str=args.model_str,                                                                                              # DEFAULT PATH 16_500_256_60_1e-07_0.0_4_2_4_4_16_3_2_0.0
        batch_size=args.batch_size, batch_size_test=args.batch_size_test,                                                      # batch sizes
        seq_len=args.sequence_length, lag=args.lag,                                                                            # parallelizing and sequence shapes
        time_emb=args.x_emb, y_emb=args.y_emb, full_emb=args.full_emb,                                                         # emb params
        n_enc=args.encoder, n_dec=args.decoder, heads=args.heads, forward_exp=args.forward_expansion, dropout=args.dropout,    # model size
        lr=args.lr, wd=args.wd, scheduler_step=args.scheduler_step, epochs=args.epochs)  # learning params
