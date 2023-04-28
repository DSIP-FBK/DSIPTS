import torch

class dictConfiguration():

    def __init__(self, name: str) -> None:
        self.name = name
        self.strategy_dict = {
                'use_target_past': True,
                'use_yprec': False,
                'iter_forward': True,
                'quantiles': None
            }
        self.model_dict = {
                'n_cat_var': 5,
                'n_target_var': 1,
                'seq_len': 265,
                'lag': 65,
                'd_model': 16,
                'n_enc_layers': 1,
                'n_dec_layers': 1,
                'head_size': 8,
                'num_heads': 2,
                'fw_exp': 1,
                'dropout': 0.3,
                'num_lstm_layers': 1
            }
        self.train_dict = {
                'lr': 1e-05,
                'wd': 0.0,
                'bs': 256,
                'epochs': 100,
                'hour': 24,
                'optimizer_index_selection': 0,
                'loss_index_selection': 0,
                'loss_reduction': 'mean',
                'sched_index_selection': 0,
                'sched_step': 40,
                'sched_gamma': 0.1
            }
        self.test_dict = {
                'bs_t': 128,
                'hour_test': 24
            }

    def __str__(self):
        print('\n'+'- '*10)
        print(f'NAME = {self.name}')
        print('\n'+'- '*10)
        print('> Strategy Dict:')
        for val in self.strategy_dict.keys():
            print(f'   {val} - {self.strategy_dict[val]}')
        print('\n'+'- '*10)
        print('> Model Dict:')
        for val in self.model_dict.keys():
            print(f'   {val} - {self.model_dict[val]}')
        print('\n'+'- '*10)
        print('> Train Dict:')
        for val in self.train_dict.keys():
            print(f'   {val} - {self.train_dict[val]}')
        print('\n'+'- '*10)
        print('> Test Dict:')
        for val in self.test_dict.keys():
            print(f'   {val} - {self.test_dict[val]}')
        return ''
  
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

if __name__=='__main__':
    conf_dict = dictConfiguration('default')
    _ = print(conf_dict)