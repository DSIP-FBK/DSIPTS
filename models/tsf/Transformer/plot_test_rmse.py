import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

def plot_test_rmse(net, model_str, dl, scaler, lag, actual_epochs, device):

    net_best = net
    net_last = copy.deepcopy(net)

    path_best = './Tensorboard/models/'+model_str+'_best.pt'
    net_best.load_state_dict(torch.load(path_best, map_location=device))
    net_best.eval()

    path_last = './Tensorboard/models/'+model_str+'_last.pt'
    net_last.load_state_dict(torch.load(path_last, map_location=device))
    net_last.eval()
   
    y_data = []
    pred_best = []
    pred_last = []
    for _,(ds,y,low) in enumerate(tqdm(dl, desc = "Plot Test RMSE - ")):
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
        
        del y,y_clone_best,y_clone_last,ds
        torch.cuda.empty_cache()
        
    y_data = np.vstack(y_data)
    pred_best = np.vstack(pred_best)
    pred_last = np.vstack(pred_last)

    #* -- RMSE --
    rmse_best = np.sqrt(np.mean((y_data-pred_best)**2, axis = 0))
    rmse_last = np.sqrt(np.mean((y_data-pred_last)**2, axis = 0))

    x=np.arange(1,lag+1)

    fig, ax = plt.subplots(1, 1, figsize=(18, 18))
    fig.suptitle(model_str + f' - {actual_epochs} - RMSE OVER LAG STEPS')
    fig.supxlabel('LAG STEPS')
    fig.supylabel('RMSE')

    ax.plot(x, rmse_best, label = 'rmse_best')
    ax.plot(x, rmse_last, label = 'rmse_last')
    ax.grid(True)
    ax.legend()

    fig.savefig('./Tensorboard/models/'+model_str+f'_rmse_{actual_epochs}.png')
    print(f'\n RMSE Plot of MODEL {model_str} saved')
    