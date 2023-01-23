import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from datetime import datetime
import matplotlib.dates as mdates


def plot_test_batch(net, model_str, dl, scaler, lag, device):

    net_best = net
    net_last = copy.deepcopy(net)

    path_best = './Tensorboard/models/'+model_str+'_best.pt'
    net_best.load_state_dict(torch.load(path_best, map_location=device))
    net_best.eval()

    path_last = './Tensorboard/models/'+model_str+'_last.pt'
    net_last.load_state_dict(torch.load(path_last, map_location=device))
    net_last.eval()
 
    ds_time = []
    y_data = []
    pred_best = []
    pred_last = []

    for i,(ds,y,low) in enumerate(tqdm(dl, desc = "Plot Test Batch - ")):
        
        # 4 PLOT
        if i>3:
            break

        ds = ds[:1].to(device)
        y = y[:1].to(device)
        low = low[:1].to(device)
        y_clone = y.detach().clone().to(device)

        output_best = net_best(ds,y_clone,low).cpu().detach().numpy()
        output_last = net_last(ds,y_clone,low).cpu().detach().numpy()

        # RESCALE THE OUTPUTS TO STORE Y_DATA,PRED_BEST, PRED_LAST
        y = scaler.inverse_transform(y[:,-lag:].cpu()).flatten()
        prediction_best = scaler.inverse_transform(output_best[:,-lag:].squeeze(2)).flatten()
        prediction_last = scaler.inverse_transform(output_last[:,-lag:].squeeze(2)).flatten()

        y_data.append(y)
        pred_best.append(prediction_best)
        pred_last.append(prediction_last)
        
        time = []
        for x in ds[0,-lag:]:
            time.append( datetime.strptime('-'.join(map(str,x[:-1].tolist())), '%Y-%m-%d-%H') ) # datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
        ds_time.append(time)

        del y,y_clone,ds
        torch.cuda.empty_cache()
    
    assert len(y_data)==len(pred_best)
    assert len(y_data)==len(pred_last)
    
    pred_best = np.vstack(pred_best)
    pred_last = np.vstack(pred_last)
    y_data = np.vstack(y_data)
    ds_time = np.vstack(ds_time)

    # import pdb
    # pdb.set_trace()
    #* -- PLOT -- 
    
    y_lim = [0, 6000]
    plt.cla()
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(model_str)
    fig.supxlabel('Time')
    fig.supylabel('Batches')

    for k, ax in enumerate(axs.flat):
        # pred_last_k = np.concatenate((y_data[k,:-lag],pred_last[k]), axis=0)
        # pred_best_k = np.concatenate((y_data[k,:-lag],pred_best[k]), axis=0)
        ax.cla()
        ax.plot(ds_time[k], pred_best[k], 'p' ,label = 'yhat_best')
        ax.plot(ds_time[k], pred_last[k], 'p', label = 'yhat_last')
        ax.plot(ds_time[k], y_data[k], label = ' y')
        # ax.set_title()
        ax.set_ylim(y_lim)

        datemin = np.datetime64(ds_time[k,0])
        datemax = np.datetime64(ds_time[k,-1])
        ax.set_xlim(datemin, datemax)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d-%H'))
        # ax.vlines(ds_time[k,-lag], y_lim[0], y_lim[1], linestyles ="dotted", colors ="k", label='Start Predicting')

        ax.grid(True)
        ax.legend()
    # plt.show()
    fig.savefig('./Tensorboard/models/'+model_str+'_plot.png')
    
    print(f' Prediction Plot of MODEL {model_str} saved')
