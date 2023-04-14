
# Usage of DSIPTS 
This repo collect some examples related to the use of [dsipts] (https://gitlab.fbk.eu/dsip/dsip_dlresearch/timeseries). Using this repo you can train some DL models for timeseries forecasting using public datasets like [Monarch](https://forecastingdata.org/) or (six_dataset)[https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy] used for benchmarking timeseries models. 

Here we report a complete approach using [Hydra](https://hydra.cc/) that allows to better manage multiple experiments.

This particular repo is structured for working with the six datasets (that are 9 because ethh has different resolution and periods): `['electricity','etth1','etth2','ettm1','ettm2','exchange_rate','illness','traffic','weather']`. The task is to predict `y` using past data and other past covariates for a variable number of steps depending on the paper used as reference.

# Installation


In a pre-generated environment install pytorch and pytorch-lightning (`pip install pytorch-lightning==1.9.4`) then go inside the lib folder and execute:

```
python setup_local.py install --force
```

Alternatively, you can install it from the package registry:

```
pip install --force dsipts --index-url https://dsipts:glpat-98SR11neR7hzxy__SueG@gitlab.fbk.eu/api/v4/projects/4571/packages/pypi/simple
```


# Configuration

- copy the folder `all_six_datasets` inside a data folder (in what follows `/home/agobbi/Projects/ExpTS/data`).
- place yoursel in `bash_examples`
- train the models
- create the folders `csv` and `plots` in the `pathdir` in this case `/home/agobbi/Projects/ExpTS`


# Hydra

In you environment install hydra and the joblib launcher for the paralellization tasks:
```
pip install hydra-core
pip install hydra-joblib-launcher
```

The script used are:
- **train.py** for trainint
- **inference.py** for inference 
- **compare.py** for comparing different models

Hydra is used for composing configuration files. In our case most of the parameter can be reused among the different models and are collected under the general configuration file `config/config.yaml`\. In what follows the `weather` dataset is used, and notice that this dataset has a frequency of **10 minutes**. The parameters here are the same described in the `dsitps` documentation but clearly some of them can not be modified since they depend on the selected time series.
The configuration files related to this experiment can be found in `config_weather`; a generic config folder contains:
```
config.yaml               # containing the global configuration, see below one example
compare.yaml              # instructions for comparing different models
architecture/             # the folder containing the configurations specific for all the models to test
config_used/              # this folder will be populated while training the models, and will be used in the comparison phase              

```

The config file in the case of the weather dataset is reported and commented below.

```
dataset:
  dataset: 'weather'
  path: '/home/agobbi/Projects/ExpTS/data' ##path to data. In the folder data must be present the folder six_dataset

scheduler_config:
  gamma: 0.1
  step_size: 100

optim_config:
  lr: 0.0005
  weight_decay: 0.01

model_configs:
  past_steps: 16
  future_steps: 16
  quantiles: [0.1,0.5,0.9] ##if you want to use quantile loss, otherwise set it to []
  past_channels : null #dataset dependent  hydra expect you to set it anyway also if it depends on data
  future_channels : null #dataset dependent
  embs: null #dataset dependent
  out_channels: null #dataset dependent

split_params:
  perc_train: 0.7
  perc_valid: 0.1
  range_train: null
  range_validation: null 
  range_test: null
  shift: 0
  starting_point: null
  skip_step: 1
  past_steps: model_configs@past_steps  ##this is a convinient what to reuse previous information!
  future_steps: model_configs@future_steps

train_config:
  dirpath: "/home/agobbi/Projects/ExpTS"
  num_workers: 0
  auto_lr_find: true
  devices: [0]                   

inference:
  output_path: "/home/agobbi/Projects/ExpTS"
  load_last: true
  batch_size: 200 
  num_workers: 4
  set: "validation"
  rescaling: false (sometimes you want to get the errors on normalized datasets)

#since now standard things, these two sessions are the most crucial and useful

defaults: ##keyword indicating the defaults for hydra
  - _self_ ##take all this configuration 
  - architecture: null ##and let the use specify the architecture to use (be aware that the filed here is the same as the folder containing the other yaml files)
  - override hydra/launcher: joblib ##use joblib for multiprocess allowing parallelization in case of multirun


hydra:
  launcher:
    n_jobs: 2 ##parameters indicate the number of parallel jobs in case of multirun
  output_subdir: null #do not save any file
  sweeper:
    params:
      architecture: glob(*) ##this is a way to train all the models in the architecure folder
```

In the `config_weather/architecture` folder there are the selected models that have the following structure:

```
# @package _global_  ##care this must be present!


#the specified parameters below overwrite the default configuration having a more compact representation
model:
  type: 'linear'
  
ts:
  name: 'weather'
  version: 1 # if you need to versioning a model
  enrich: ['hour'] 
  use_covariates: false #if true all the columns of the dataset will be used as past features

model_configs:
  cat_emb_dim: 16
  kernel_size_encoder: 15
  sum_emb: true
  hidden_size: 256
  kind: 'linear'

train_config:
  batch_size: 256
  max_epochs: 250

```

Hydra allows us to train a specific model using:

```
python train.py  architecture=linear --config-dir=config_weather
```

or a list of models in paralle:

```
python train.py  -m architecture=linear, dlinear --config-dir=config_weather
```

or all the implemented models:

```
python train.py  -m  --config-dir=config_weather
```
If the row `override hydra/launcher: joblib` is commented the train will be consecutive, otherwise in parallel. In the latter case the output in the terminal will be a mess, please check all is woking fine. In the future the logging will be more efficient.

Once the models are trained, the relative full configurations are saved in `config_used` and can be used for inference or comparison:

```
python compare.py -c config/compare.yaml

```
In the `dirpath` folder `/home/agobbi/Projects/ExpTS/` there are three folder now: `weights` containing the model and the weights, `plots` containing some plots coming from the `compare` script and the `csv` forder containing the files.

A typical example of plot is displayed below and shows the MSE at different lags in the test set for different models:

![plot](figures/weather_test_MSE.jpeg)

The loss plot is currenty broken on server, you can reproduce it form the notebook `4- results` (see the notebook section)


# Testing
You can use the `config_test` for testing your models. In this case you can use smaller model with fewer epochs:

```
python train.py -m --config-dir=config_test
```





