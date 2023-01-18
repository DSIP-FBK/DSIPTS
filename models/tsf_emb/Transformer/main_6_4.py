import torch
import numpy as np
from model import Transformer
from training import training
from dataloading import dataloading
from Transformer.plot_test_batch import testing
import matplotlib.pyplot as plt

def main(
    train:bool = True,
    batch_size:int = 20,
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
                    time_emb=time_emb, y_emb = y_emb, full_emb=full_emb, context_size=5,
                    heads=heads, forward_exp=forward_exp, dropout=dropout, device=device).to(device)

    # number of params to check its size
    total_params = sum( param.numel() for param in net.parameters() )
    trainable_params = sum( p.numel() for p in net.parameters() if p.requires_grad )
    print(f'\t total_params     = {total_params}')
    print(f'\t trainable_params = {trainable_params}')


    # Load the necessary data according to 'train' (bool var): 
    #* DATA
    # True : load all dataloaders and scaler and start the training phase
    # False : load only test_dl and scaler to evaluate
    if train:
        train_dl, val_dl, test_dl, scaler_y = dataloading(batch_size=batch_size, seq_len=seq_len, lag=lag, step=1, aim='train')
        training(net, device, train_dl, val_dl, 
                lag=lag, n_enc=n_enc, n_dec=n_dec,
                lr=lr, wd=wd, scheduler_step=scheduler_step, epochs=epochs)
    else:
        _, _, test_dl, scaler_y = dataloading(batch_size=batch_size, seq_len=seq_len, lag=lag, step=1, aim='test')

    #params for savings
    p_for_save = [epochs,lr,wd,n_enc,n_dec]
    # model path for evaluation
    best_model_path = f'./Tensorboard/models/{epochs}_{lr}_{wd}_{n_enc}_{n_dec}_best_model'
    last_model_path = f'./Tensorboard/models/{epochs}_{lr}_{wd}_{n_enc}_{n_dec}_last_model'
    
    testing(net, path_best=best_model_path, path_last=last_model_path, 
            dl=test_dl, scaler=scaler_y, lag=lag, 
            p_for_save=p_for_save, device=device)

    # pred_last = np.concatenate((y_data[:-lag],pred_last), axis=0)
    # pred_best = np.concatenate((y_data[:-lag],pred_best), axis=0)

    # # import pdb
    # # pdb.set_trace()

    # plt.cla()
    # # x = np.arange(pred_best.shape[0])
    # plt.plot(ds, pred_best, 'p' ,label = 'yhat_best')
    # plt.plot(ds, pred_last, 'p', label = 'yhat_last')
    # plt.plot(ds, y_data, label = ' y')
    # plt.legend()
    # plt.title(f'ep={epochs} lr={lr} wd={wd} n.enc={n_enc} n.dec={n_dec}')
    # plt.ylim([0, 10000])
    
    # # plt.show()
    # plt.savefig(f'./Tensorboard/models/{epochs}_{lr}_{wd}_{n_enc}_{n_dec}_plot.png')
    # print(f' MODEL ep{epochs}_lr{lr}_wd{wd}_nenc{n_enc}_ndec{n_dec} saved')

if __name__ == '__main__':
    main(
        train=True, batch_size=32, seq_len=256, lag=61, 
        time_emb=8, y_emb=8, full_emb=128,
        n_enc=6, n_dec=4, heads=4, forward_exp=2, dropout=0.0,
        lr=1e-06, wd=0.0, scheduler_step=25, epochs=1000, early_stopping = 2000)
