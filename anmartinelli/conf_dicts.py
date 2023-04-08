import torch

class dictConfiguration():

    def __init__(self, name: str) -> None:
        self.name = name
        self.strategy_dict = {
                'use_target_past': True,
                'use_yprec': True,
                'iter_forward': True,
                'quantiles': None
            }
        self.model_dict = {
                'n_cat_var': 5,
                'n_target_var': 1,
                'seq_len': 265,
                'lag': 65,
                'd_model': 64,
                'n_enc_layers': 1,
                'n_dec_layers': 5,
                'head_size': 8,
                'num_heads': 4,
                'fw_exp': 2,
                'dropout': 0.0,
                'num_lstm_layers': 4
            }
        self.train_dict = {
                'lr': 1e-04,
                'wd': 0.0,
                'bs': 128,
                'epochs': 1000,
                'hour': 24,
                'optimizer_index_selection': 0,
                'loss_index_selection': 0,
                'loss_reduction': 'mean',
                'sched_index_selection': 0,
                'sched_step': 100,
                'sched_gamma': 0.1
            }
        self.test_dict = {
                'bs_t': 64,
                'hour_test': 24
            }
        
        
    def get_optim(self, model):
        if self.train_dict['optimizer_index_selection'] == 0:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.train_dict['lr'], weight_decay=self.train_dict['wd'])
        elif self.train_dict['optimizer_index_selection'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.train_dict['lr'], weight_decay=self.train_dict['wd'])
        elif self.train_dict['optimizer_index_selection'] == 2:
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.train_dict['lr'], weight_decay=self.train_dict['wd'])
        elif self.train_dict['optimizer_index_selection'] == 3:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.train_dict['lr'], weight_decay=self.train_dict['wd'])
        else:
            raise ValueError('Non Valid Index for Optimizer\n\
                            Index = 0: AdamW\n\
                            Index = 1: Adam\n\
                            Index = 2: Adagrad\n\
                            Index = 3: SGD}')
        return optimizer

    def get_loss_fun(self):
        if self.train_dict['loss_index_selection']==0:
            loss_fun = torch.nn.L1Loss(reduction=self.train_dict['loss_reduction'])
        elif self.train_dict['loss_index_selection']==1:
            loss_fun = torch.nn.MSELoss(reduction=self.train_dict['loss_reduction'])
        else:
            raise ValueError('Non Valid Index for Loss Function\n\
                            Index = 0: L1Loss\n\
                            Index = 1: MSELoss')
        return loss_fun

    def get_scheduler(self, optimizer):
        if self.train_dict['sched_index_selection']==0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.train_dict['sched_step'], gamma=self.train_dict['sched_gamma'])
        else:
            raise ValueError('Non Valid Index for Scheduler\n\
                            Index = 0: StepLR')
        return scheduler
