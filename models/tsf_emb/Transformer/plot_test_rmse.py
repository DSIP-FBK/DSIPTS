import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

def plot_test_rmse(net, path_best, path_last, dl, scaler, bs, lag, device):

    net_best = net
    net_last = copy.deepcopy(net)

    net_best.load_state_dict(torch.load(path_best, map_location=device))
    net_best.eval()

    net_last.load_state_dict(torch.load(path_last, map_location=device))
    net_last.eval()

    y_data = []
    pred_best = []
    pred_last = []
    for i,(ds,y,low) in enumerate(tqdm(dl, desc = "Plot Test RMSE - ")):
        ds = ds.to(device)
        y = y.to(device)
        low = low.to(device)
        y_clone = y.detach().clone().to(device)

        output_best = net_best(ds,y_clone,low).detach().cpu().numpy() 
        output_last = net_last(ds,y_clone,low).detach().cpu().numpy() 

        # RESCALE THE OUTPUTS TO STORE Y_DATA,PRED_BEST, PRED_LAST
        y = scaler.inverse_transform(y[:,-lag:].cpu()).flatten()
        prediction_best = scaler.inverse_transform(output_best[:,-lag:].squeeze(2)).flatten()
        prediction_last = scaler.inverse_transform(output_last[:,-lag:].squeeze(2)).flatten()

        y_data.append(y.reshape(-1,lag)) #= np.concatenate((y_data,y), axis=0)
        pred_best.append(prediction_best.reshape(-1,lag)) #= np.concatenate((pred_best,prediction_best), axis=0)
        pred_last.append(prediction_last.reshape(-1,lag)) #= np.concatenate((pred_best,prediction_best), axis=0)
        
        del y,y_clone,ds
        torch.cuda.empty_cache()
        
    y_data = np.vstack(y_data)
    pred_best = np.vstack(pred_best)
    pred_last = np.vstack(pred_last)

    #* -- RMSE --
    rmse_best = np.sqrt(np.mean((y_data-pred_best)**2, axis = 0))
    rmse_last = np.sqrt(np.mean((y_data-pred_last)**2, axis = 0))

    path_str = path_best[:-11].split('/')[-1]
    x=np.arange(lag)
    plt.cla()
    plt.plot(x, rmse_best, label = 'rmse_best')
    plt.plot(x, rmse_last, label = 'rmse_last')
    plt.legend()
    plt.title('RMSE OVER LAG STEPS')
    # plt.vlines(1, -100, 10000, linestyles ="dotted", colors ="k")
    plt.ylim([0, 1000])
    
    # plt.show()
    plt.savefig(path_best[:-11]+'_plot_rmse.png')
    print(f' RMSE Plot of MODEL {path_str} saved') #[-11] to not print '_best_model'
    