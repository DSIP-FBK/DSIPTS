import torch

dict_params = {
    'path': '',

    'strategy': {
        'use_target_past': True,
        'use_yprec': True,
        'iter_forward': True,
        'quantiles': None
    },
    'model': {
        'n_cat_var': 5,
        'n_target_var': 1,
        'seq_len': 265,
        'lag': 65,
        'd_model': 16,
        'n_enc_layers': 3,
        'n_dec_layers': 3,
        'head_size': 4,
        'num_heads': 4,
        'fw_exp': 4,
        'dropout': 0.0,
        'num_lstm_layers': 4
    },
    'train': {
        'lr': 1e-04,
        'wd': 0.0,
        'bs': 32,
        'epochs': 1000,
        'hour': 24,
        'optimizer_index_selection': 0,
        'loss_index_selection': 0,
        'loss_reduction': 'mean',
        'sched_index_selection': 0,
        'sched_step': 100,
        'sched_gamma': 0.1
    },
    'test': {
        'bs_t': 2,
        'hour_test': 24
    }
}

def get_optim(model, optimizer_index_selection, lr, wd):
    if optimizer_index_selection == 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_index_selection == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_index_selection == 2:
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_index_selection == 3:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError('Non Valid Index for Optimizer\n\
                         Index = 0: AdamW\n\
                         Index = 1: Adam\n\
                         Index = 2: Adagrad\n\
                         Index = 3: SGD}')
    return optimizer

def get_loss_fun(loss_index_selection, reduction):
    if loss_index_selection==0:
        loss_fun = torch.nn.L1Loss(reduction=reduction)
    elif loss_index_selection==1:
        loss_fun = torch.nn.MSELoss(reduction=reduction)
    else:
        raise ValueError('Non Valid Index for Loss Function\n\
                         Index = 0: L1Loss\n\
                         Index = 1: MSELoss')
    return loss_fun

def get_scheduler(sched_index_selection, optimizer, sched_step, sched_gamma):
    if sched_index_selection==0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
    else:
        raise ValueError('Non Valid Index for Scheduler\n\
                         Index = 0: StepLR')
    return scheduler

def get_save_str(mix, prec):
    if mix:
        if prec:
            path = 'Mix_Prec'
        else:
            path = 'Mix_NoPrec'
    else:
        if prec:
            path = 'NoMix_Prec'
        else:
            path = 'NoMix_NoPrec'
    return path

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)