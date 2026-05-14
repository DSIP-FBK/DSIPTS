## 1.1.39
- upgrade dependencies
## 1.1.38
- Added explicit warning to PYTORCH LIGHTNING
## 1.1.35 to 1.1.37
- fix lightening quarantine issue
## 1.1.34
- bug fixing lr finder
## 1.1.33
- added chronos v2
- update python

## 1.1.32
- timekan multioutput fix

## 1.1.31
- aim & logger back working
- fixed validation with few sampler

## 1.1.21-22-23-26-27-30 (2026-02-02)
- fixing lightening version support (not yet solved for version >2.6)
- removed aim (compatibility)
- logger

## 1.1.20 (2026-02-01)
- automation of tags, version etc using only the makefile
- update newer version of python
## 1.1.14 (2025-12-02)
- Add CHANGELOG to the pip page
- Bug fixing for non autoregressive models
- Added TimesNet, TimeKAN
- Added 0 shot and full shot TTM
- added more option parameter to define an experiment (`split_params`, `optim_config` and `train_config`)
## 1.1.13 (2025-12-02)

- Added weight for the WeightedRandomSampler (only for the training part)
- Bug fixing when using groups in the categorical variables
- TTM fixed for working with zeropad for shorter sequences and future covariates

## 1.1.12 (2025-11-07)
- Bug fixing dynamo stuff for `TTM`
- Bug fixing loading weights after training process ('dynamo module can not load weights`)
- Force to not compile some models (there are piece of code that are not aligned with dynamo)
- Bug fixing test configurations

## 1.1.11 (2025-11-06)
- Added `torch.compile` for better performance on recent GPU
- Stable `TTM` model according to version 1.1.5, still under debug, use at your own risk
- Bux Fixing `cprs` inference (now produces 3 quantiles: `[0.05, 0.5, 0.95]`). The `persistence_weight` is the value of `alpha` in the paper (between 0 and 1)

## 1.1.9 (2025-09-19)
- Added `cprs` https://arxiv.org/pdf/2412.15832v1 loss function. In this case use the quantile parameter to ask for the ensembles: `quantiles = [1,2,3,4,5,6,7,8,9,10]` will create 10 ensembles. For now the inference part will return just the mean, TODO: estimate a confidence interval with the ensembles 
- Added `long_lag` the L1 error will be modulated with a linear weight depending on the lag in the future: the penalization goes from `1` to `persistence_weight`

## 1.1.8 (2025-09-12)
- Added `Simple` model (just two linear layers)

## 1.1.7 (2025-09-08)
- bug fixing `DilatedConv`
## 1.1.5 (2025-08-29)
- rewriting most of the modules for handling different future and past categorical variables
- extension of categorical and future covariates in almost all the models
- `uv` full management of the package
- refactoring almost all the structure and documentation

## 1.1.4 (2025-08-22)
- added `restart: true` tro model configuration to restart the training procedure: carefurl the max_epochs should be increased if you need to retrain

## 1.1.4 (2025-07-29)
- bug fixing tuner learning rate
- added TTM model and TimeXer
- added compatibility with newer version of lightening and torch

## 1.1.1 
- added [SAM optimizer](https://arxiv.org/pdf/2402.10198) 
```bash
 python train.py  --config-dir=config_test --config-name=config architecture=itransformer dataset.path=/home/agobbi/Projects/ExpTS/data train_config.dirpath=tmp inference=tmp model_configs.optim=SAM +optim_config.rho=0.5
 ```