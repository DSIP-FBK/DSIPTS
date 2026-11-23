## Shrinkage study, round 3 (2026-11-23) — Exchange rate, the AE's "persistence is the right forecast" case (PLAN §8).
## Prediction: beta -> 0 (or << 1) and the shrunk forecast recovers naive. 6 trained + 2 zero-shot = 8 jobs.
## Same setup as ETTm2 (iTransformer, frozen hyper-params), past 96 / future 96. NOT a script to run top to bottom.

## ---- 1. CLUSTER, from timeseries/bash_examples/
cd /storage3/DSIP/agobbi/Projects/timeseries && git pull && cd bash_examples
source /storage3/DSIP/agobbi/Projects/timeseries/dsipts/.venv/bin/activate
ls /storage3/DSIP/agobbi/Projects/ExpTS/data/all_six_datasets/exchange_rate/exchange_rate.csv   ## must exist
export HF_HUB_OFFLINE=1    ## weights already downloaded for round 2 (C0 in shrink2.sh)

## A. iTransformer l1 + mse x 3 seeds = 6 jobs
python train.py --config-dir=config_paper_persistence_exchange --config-name=config_abacus architecture=itransformer model_configs.loss_type=l1,mse train_config.seed=1,2,3 -m
## C. zero-shot Chronos-2 + Timestral = 2 jobs
python train.py --config-dir=config_paper_persistence_exchange --config-name=config_abacus architecture=chronos2,timestral -m

## ---- 2. CLUSTER, after all 8 finish: validation + test predictions
python compare.py --config-dir=config_paper_persistence_exchange --config-name=compare_abacus set=validation,test -m

## ---- 3. LAPTOP
rsync -avz -e "ssh -F $HOME/.ssh/config_jump" --exclude='*.ckpt' --exclude='*.pkl' --exclude='*_train_*' diclubclsin:/storage3/DSIP/agobbi/Projects/ExpTS/paper_persistence_v3_exchange /home/agobbi/Projects/ExpTS/
## then: loss_normalization_wip/real_shrink_rolling.py exchange
