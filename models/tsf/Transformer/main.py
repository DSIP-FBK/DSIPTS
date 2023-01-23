import torch
from model import Transformer
from training import training
from dataloading import dataloading
from plot_loss_train import plot_loss_train
from plot_test_batch import plot_test_batch
from plot_test_rmse import plot_test_rmse
import argparse


def main(
    load_all_seq:bool = True,
    train:bool = True,
    plot_loss:bool = True,
    plot_batch:bool = True,
    plot_rmse:bool = True,
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
    epochs:int = 4000,
    early_stopping:int = 2000
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

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

    # paths of the model
    if model_str=='':
        model_str = f'{batch_size}_{epochs}_{seq_len}_{lag}_{lr}_{wd}_{n_enc}_{n_dec}_{time_emb}_{y_emb}_{full_emb}_{heads}_{forward_exp}_{dropout}'
    
    print('> MODEL: '+ model_str)
    
    # Load the necessary data according to 'train' (bool var) and in case start with training: 
    #* DATA
    print('\n> Data Loading')
    # True : load all dataloaders and scaler and start the training phase
    # False : load only test_dl and scaler to evaluate
    if train:
        train_dl, val_dl, test_dl, scaler_y = dataloading(batch_size=batch_size, batch_size_test=batch_size_test,
        seq_len=seq_len, lag=lag, step=1, load_all_seq=load_all_seq, train_bool=train)
        print('\n Data Loaded\n')
        print('-'*50)
        #* TRAIN
        training(net, device, train_dl, val_dl, model_str=model_str,
                lag=lag, lr=lr, wd=wd, scheduler_step=scheduler_step, epochs=epochs)
    else:
        _, _, test_dl, scaler_y = dataloading(batch_size=batch_size, batch_size_test=batch_size_test,
        seq_len=seq_len, lag=lag, step=1, load_all_seq=load_all_seq, train_bool=train)
        print('\n Data Loaded\n')
        print('-'*50)

    # Use the .pkl file to plot train and validation losses of the model
    #* LOSSES
    if plot_loss:
        print('> Start Plot Losses\n')
        plot_loss_train(model_str=model_str, lag=lag)
        print('\n End Plot Losses\n')
        print('-'*50)
    
    # Wrt the value of 'plot' (bool var) save a plot of predictions of the corresponding model
    # In output we get the path of the plot
    #* BATCHES
    if plot_batch:
        print('> Start Plot Test Batch\n')
        plot_test_batch(net, model_str=model_str, dl=test_dl, scaler=scaler_y, lag=lag, device=device)
        print('\n End Plot Test Batch\n')
        print('-'*50)

    # Wrt the value of 'rmse' (bool var) save a plot of mean rmse of all predicion of the corresponding model over all test set
    #* RMSE
    if plot_rmse:
        print('> Start Plot Test RMSE\n')
        plot_test_rmse(net=net, model_str=model_str, dl=test_dl, scaler=scaler_y, lag=lag, device=device)
        print('\n End Plot Test RMSE\n')
        print('-'*50)

if __name__ == '__main__':

    ## PARSER
    parser = argparse.ArgumentParser(description="Transformer Model")

    # goal of the program
    parser.add_argument("-las", "--load_all_seq", action="store_true", help="Load all the sequences or only ones starting from a fixed hour (12, check dataloading.py)")
    parser.add_argument("-t", "--train", action="store_true", help="Run training of the nn")
    parser.add_argument("-l", "--loss", action="store_true", help="Plot both losses of the model")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot a batch of the test set with predictions")
    parser.add_argument("-r", "--rmse", action="store_true", help="Compute the rmse for each step of predicted value")

    parser.add_argument("-mod", "--model_str", type=str, default='', help='String to fetch the model')
    parser.add_argument("-bs", "--batch_size", default=16, type=int, help="Number of batch compiled simultaneously")
    parser.add_argument("-bs_t", "--batch_size_test", default=2, type=int, help="Number of batch compiled simultaneously ON TEST SET")
    parser.add_argument("-seq_len", "--sequence_length", default=256, type=int, help="Number of total size (context + target)")
    parser.add_argument("-lag", "--lag", default=60, type=int, help="Number of target size (add 1 to avoid 'Attention problems')")
    parser.add_argument("-xe", "--x_emb", default=4, type=int, help="Size of x embedding")
    parser.add_argument("-ye", "--y_emb", default=4, type=int, help="Size of y embedding")
    parser.add_argument("-fe", "--full_emb", default=16, type=int, help="Size of full embedding")
    parser.add_argument("-enc", "--encoder", default=4, type=int, help="Number of encoders")
    parser.add_argument("-dec", "--decoder", default=2, type=int, help="Number of decoders")
    parser.add_argument("-he", "--heads", default=3, type=int, help="Number of heads into the attention")
    parser.add_argument("-f", "--forward_expansion", default=2, type=int, help="Factor value for forward expansion into the attention")
    parser.add_argument("-d", "--dropout", default=0.0, type=float)
    parser.add_argument("-lr", "--lr", default=1e-07, type=float)
    parser.add_argument("-wd", "--wd", default=0.0, type=float)
    parser.add_argument("-s", "--scheduler_step", default=150, type=int)
    parser.add_argument("-E", "--epochs", default=500, type=int)
    parser.add_argument("-early", "--early_stopping", default=3000, type=int)

    args = parser.parse_args()
    print('\n> AIMS:')
    print(f'   - load_all_seq={args.load_all_seq}')
    print(f'   - train={args.train}')
    print(f'   - plot={args.plot}')
    print(f'   - rmse={args.rmse}')
    #RUNNING MAIN
    main(
        load_all_seq=args.load_all_seq,                                                                                        # data params
        train=args.train, plot_loss=args.loss, plot_batch=args.plot, plot_rmse=args.rmse,                                                           # aim bools
        model_str=args.model_str,
        batch_size=args.batch_size, batch_size_test=args.batch_size_test,
        seq_len=args.sequence_length, lag=args.lag,                                                # parallelizing and sequence shapes
        time_emb=args.x_emb, y_emb=args.y_emb, full_emb=args.full_emb,                                                         # emb params
        n_enc=args.encoder, n_dec=args.decoder, heads=args.heads, forward_exp=args.forward_expansion, dropout=args.dropout,    # model size
        lr=args.lr, wd=args.wd, scheduler_step=args.scheduler_step, epochs=args.epochs, early_stopping = args.early_stopping)  # learning params


    # DEFAULT PATH 16_500_256_60_1e-07_0.0_4_2_4_4_16_3_2_0.0