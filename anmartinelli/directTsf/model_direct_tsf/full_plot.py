import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from datetime import datetime
import matplotlib.dates as mdates
import pickle as pkl


def full_plot(net, path, dl, scaler, lag, device):

    fig,axs = plt.subplots(3, 2, figsize=(18, 18))
    title = path.split('/')[-1]
    fig.suptitle(title)

    import pdb
    pdb.set_trace()

    plots = axs.flat
    loss_plot = plots[0]
    rmse_plot = plots[1]
    batch_plot = plots[2:]
    # FIRST LINE FOR LOSSES AND RMSE
    # SECOND AND THIRD FOR TEST BATCHES

    ## AX[1]
    path_pkl = path+'.pkl'
    with open(path_pkl, 'rb') as f:
        losses = pkl.load(f)
        import pdb
        pdb.set_trace()
    
        train_loss = losses[0]
        val_loss = losses[1]
        actual_epochs = len(train_loss)+1
        x = np.arange(1, len(train_loss)+1)

        loss_plot.cla()
        
        loss_plot.plot(x, train_loss, label = 'train_loss')
        loss_plot.plot(x, val_loss, label = 'val_loss')
        loss_plot.grid(True)
        loss_plot.legend()

        # fig.savefig(path+f'_loss_{actual_epochs}.png')
        f.close()

    
    net_best = net
    net_last = copy.deepcopy(net)

    path_best = path+'_best.pt'
    net_best.load_state_dict(torch.load(path_best, map_location=device))
    net_best.eval()

    path_last = path+'_last.pt'
    net_last.load_state_dict(torch.load(path_last, map_location=device))
    net_last.eval()
 
    # ds_time = [] # only for test batches
    y_data = []
    pred_best = []
    pred_last = []

    for i,(ds,y,low) in enumerate(tqdm(dl, desc = " > On Test DataLoader - ")):
        ds = ds.to(device)
        y = y.to(device)
        low = low.to(device)
        y_clone_best = y.detach().clone().to(device)
        y_clone_last = y.detach().clone().to(device)
        
        output_best = net_best(ds,y_clone_best,low).detach().cpu().numpy() 
        output_last = net_last(ds,y_clone_last,low).detach().cpu().numpy() 

        # RESCALE THE OUTPUTS TO STORE Y_DATA,PRED_BEST, PRED_LAST
        y = scaler.inverse_transform(y[:,-lag:].cpu()).flatten()
        prediction_best = scaler.inverse_transform(output_best[:,-lag:].squeeze(2)).flatten()
        prediction_last = scaler.inverse_transform(output_last[:,-lag:].squeeze(2)).flatten()

        y_data.append(y.reshape(-1,lag))
        pred_best.append(prediction_best.reshape(-1,lag))
        pred_last.append(prediction_last.reshape(-1,lag))
        
        # if i<4:
        #     time = []
        #     for x in ds[0,-lag:]:
        #         time.append( datetime.strptime('-'.join(map(str,x[:-1].tolist())), '%Y-%m-%d-%H') ) # datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
        #     ds_time.append(time)

        del y,y_clone_best,y_clone_last,ds
        torch.cuda.empty_cache()
    
    assert len(y_data)==len(pred_best)
    assert len(y_data)==len(pred_last)
    
    y_data = np.vstack(y_data)
    pred_best = np.vstack(pred_best)
    pred_last = np.vstack(pred_last)
    #* -- RMSE --
    rmse_best = np.sqrt(np.mean((y_data-pred_best)**2, axis = 0))
    rmse_last = np.sqrt(np.mean((y_data-pred_last)**2, axis = 0))
    
    mean_rmse_best = np.mean(rmse_best)
    mean_rmse_last = np.mean(rmse_last)
    min_rmse_best = rmse_best.min()
    min_rmse_last = rmse_last.min()
    max_rmse_best = rmse_best.max()
    max_rmse_last = rmse_last.max()
    x=np.arange(1,lag+1)

    rmse_plot.cla()
    rmse_plot.plot(x, rmse_best, label = 'rmse_best')
    rmse_plot.plot(x, rmse_last, label = 'rmse_last')
    rmse_plot.grid(True)
    rmse_plot.set_title(f' BEST: mean {min_rmse_best:.2f}, min {mean_rmse_best:.2f}, max {max_rmse_best:.2f}\n LAST: mean {min_rmse_last:.2f}, min {mean_rmse_last:.2f}, max {max_rmse_last:.2f}')
    rmse_plot.legend()
        
    for k, ax in enumerate(batch_plot):
        ax.cla()
        y_lim = [0, 6000]
        ax.plot(x, pred_best[k], 'p' ,label = 'yhat_best')
        ax.plot(x, pred_last[k], 'p', label = 'yhat_last')
        ax.plot(x, y_data[k], label = ' y')
        ax.set_ylim(y_lim)
        ax.grid(True)
        ax.legend()

    # plt.show()
    fig.savefig(path+f'_plot_{actual_epochs}.png')
    
