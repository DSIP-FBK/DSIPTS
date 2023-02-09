import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from datetime import datetime
import matplotlib.dates as mdates


def plot_test_batch(net, model_str, dl, scaler, lag, actual_epochs, device):

    net_best = net
    net_last = copy.deepcopy(net)

    path_best = './Tensorboard/models/direct_'+model_str+'_best.pt'
    net_best.load_state_dict(torch.load(path_best, map_location=device))
    net_best.eval()

    path_last = './Tensorboard/models/direct_'+model_str+'_last.pt'
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
        y_clone_best = y.detach().clone().to(device)
        y_clone_last = y.detach().clone().to(device)

        output_best = net_best(ds,y_clone_best,low).cpu().detach().numpy()
        output_last = net_last(ds,y_clone_last,low).cpu().detach().numpy()

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

        del y,y_clone_best,y_clone_last,ds
        torch.cuda.empty_cache()
    
    assert len(y_data)==len(pred_best)
    assert len(y_data)==len(pred_last)
    
    pred_best = np.vstack(pred_best)
    pred_last = np.vstack(pred_last)
    y_data = np.vstack(y_data)
    ds_time = np.vstack(ds_time)

    #* -- PLOT -- 
    
    y_lim = [0, 6000]
    plt.cla()
    fig, axs = plt.subplots(4, 1, figsize=(18, 18))
    fig.suptitle(f'DIRECT - {model_str} - {actual_epochs} - PLOT OF BATCHES')
    fig.supxlabel('TIME')
    fig.supylabel('BATCHES')

    for k, ax in enumerate(axs.flat):
        ax.cla()
        ax.plot(ds_time[k], pred_best[k], 'p' ,label = 'yhat_best')
        ax.plot(ds_time[k], pred_last[k], 'p', label = 'yhat_last')
        ax.plot(ds_time[k], y_data[k], label = ' y')
        ax.set_title(ds_time[k,0])
        ax.set_ylim(y_lim)

        datemin = np.datetime64(ds_time[k,0])
        datemax = np.datetime64(ds_time[k,-1])
        ax.set_xlim(datemin, datemax)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d-%H'))
        
        ax.grid(True)
        ax.legend()
    # plt.show()
    fig.savefig('./Tensorboard/models/direct_'+model_str+f'_plot_{actual_epochs}.png')
    
    print(f'\n Prediction Plot of MODEL {model_str} saved')


if __name__ == '__main__':
    plot_test_batch('direct_128_1000_256_60_0.0001_0.2_6_8_16_16_32_3_2_0.2',60)