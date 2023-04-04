import torch

dict_params = {
    'model_type': '',

    'strategy': {
        'mix': True,
        'prec': True,
        'tft': True,
        'iter_forward': True,
        'quantile_loss': True,
        'quantiles': [0.1, 0.5, 0.9]
    },
    'model': {
        'seq_len': 265,
        'lag': 65,
        'n_enc': 1,
        'n_dec': 1,
        'n_embd': 32,
        'n_heads': 4,
        'head_dim': 8,
        'fw_exp': 2,
        'dropout': 0.0
    },
    'train': {
        'lr': 1e-04,
        'wd': 0.0,
        'bs': 64,
        'epochs': 1000,
        'hour': 24,
        'opt_index': 0,
        'loss_index': 0,
        'loss_reduction': 'mean',
        'sched_index': 0,
        'sched_step': 100,
        'sched_gamma': 0.1
    },
    'test': {
        'bs_t': 2,
        'hour_test': 24
    }
}

def get_optim(model, optimizer_index, lr, wd):
    if optimizer_index == 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_index == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_index == 2:
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_index == 3:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError('Non Valid Index for Optimizer\n\
                         Index = 0: AdamW\n\
                         Index = 1: Adam\n\
                         Index = 2: Adagrad\n\
                         Index = 3: SGD}')
    return optimizer

def get_loss_fun(loss_index, reduction):
    if loss_index==0:
        loss_fun = torch.nn.L1Loss(reduction=reduction)
    elif loss_index==1:
        loss_fun = torch.nn.MSELoss(reduction=reduction)
    else:
        raise ValueError('Non Valid Index for Loss Function\n\
                         Index = 0: L1Loss\n\
                         Index = 1: MSELoss')
    return loss_fun

def get_scheduler(sched_index, optimizer, sched_step, sched_gamma):
    if sched_index==0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
    else:
        raise ValueError('Non Valid Index for Scheduler\n\
                         Index = 0: StepLR')
    return scheduler

def get_save_str(mix, prec, tft):
    if mix:
        if prec:
            if tft:
                path = 'Mix_Prec_Tft'
            else:
                path = 'Mix_Prec_NoTft'
        else:
            if tft:
                path = 'Mix_NoPrec_Tft'
            else:
                path = 'Mix_NoPrec_NoTft'
    else:
        if prec:
            if tft:
                path = 'NoMix_Prec_Tft'
            else:
                path = 'NoMix_Prec_NoTft'
        else:
            if tft:
                path = 'NoMix_NoPrec_Tft'
            else:
                path = 'NoMix_NoPrec_NoTft'
    return path