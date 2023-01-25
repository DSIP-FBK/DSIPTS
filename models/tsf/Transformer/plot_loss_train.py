import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_train(model_str, lag):

    pck_str = './Tensorboard/models/'+model_str+'.pkl'
    with open(pck_str, 'rb') as f:
        losses = pkl.load(f)
        # import pdb
        # pdb.set_trace()
    
        train_loss = losses[0]
        val_loss = losses[1]
        actual_epochs = len(train_loss)+1
        x = np.arange(1, len(train_loss)+1)

        plt.cla()
        fig, ax = plt.subplots(1, 1, figsize=(18, 18))
        fig.suptitle(model_str+ f' - {actual_epochs} - PLOT OF LOSSES')
        fig.supxlabel('EPOCHS')
        fig.supylabel('LOSSES')

        ax.plot(x, train_loss, label = 'train_loss')
        ax.plot(x, val_loss, label = 'val_loss')
        ax.grid(True)
        ax.legend()

        fig.savefig('./Tensorboard/models/'+model_str+f'_loss_{actual_epochs}.png')
    
        print(f' Loss Plot of MODEL {model_str} saved')
        f.close()

    return actual_epochs

