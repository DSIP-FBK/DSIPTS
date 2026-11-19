## Shrinkage study, round 2 (2026-11-19) — robustness for the "shrink toward persistence" paper (PLAN §7).
## Run from timeseries/bash_examples/ on the new cluster, venv active. NOT a script to run top to bottom.
## 12 jobs total. Needs VALIDATION + TEST predictions (beta fit on validation / rolling window, scored on test).

## ---- 0. once: pull the configs (ettm2 ported to /storage3 + v3, new weather/architecture/simple_lin.yaml)
cd /storage3/DSIP/agobbi/Projects/timeseries && git pull && cd bash_examples
source /storage3/DSIP/agobbi/Projects/timeseries/dsipts/.venv/bin/activate
ls /storage3/DSIP/agobbi/Projects/ExpTS/data/all_six_datasets/ETT-small/ETTm2.csv   ## must exist

## ---- A. second real dataset: ETTm2, iTransformer (192/96), l1 + mse x 3 seeds = 6 jobs
## GATE: l1 horizon-mean test MAE must beat univariate ridge 2.740 (persistence 4.851). If not, ETTm2 is out.
python train.py --config-dir=config_paper_persistence_ettm2 --config-name=config_abacus architecture=itransformer model_configs.loss_type=l1,mse train_config.seed=1,2,3 -m

## ---- B. second architecture on weather: linear Simple (frozen from synthpers 1j), l1 + mse x 3 seeds = 6 jobs
python train.py --config-dir=config_paper_persistence_weather --config-name=config_abacus architecture=simple_lin model_configs.loss_type=l1,mse train_config.seed=1,2,3 -m

## ---- evaluation (after A and B finish; weather compare re-reads the old iTransformer runs too, that's fine)
python compare.py --config-dir=config_paper_persistence_ettm2 --config-name=compare_abacus set=validation,test -m
python compare.py --config-dir=config_paper_persistence_weather --config-name=compare_abacus set=validation,test -m

## ---- laptop
rsync -avz -e "ssh -F $HOME/.ssh/config_jump" --exclude='*.ckpt' --exclude='*.pkl' --exclude='*_train_*' diclubclsin:/storage3/DSIP/agobbi/Projects/ExpTS/paper_persistence_v3_ettm2 /home/agobbi/Projects/ExpTS/
rsync -avz -e "ssh -F $HOME/.ssh/config_jump" --exclude='*.ckpt' --exclude='*.pkl' --exclude='*_train_*' diclubclsin:/storage3/DSIP/agobbi/Projects/ExpTS/paper_persistence_v3_weather /home/agobbi/Projects/ExpTS/
## then: loss_normalization_wip/real_shrink_rolling.py weather ettm2

## ==== C. ZERO-SHOT foundation models (2026-11-19): does post-hoc / rolling beta improve a model nobody trained here?
## Chronos-2 (amazon/chronos-2) and Timestral-209M, 1 epoch at lr 0 (weights do not move), 3 datasets = 6 jobs.
## Deterministic -> no seeds. Smoke-tested on the laptop (load + forward only): shapes OK, lag-1 forecast ~ last value.
## 2026-11-19 OOM fix: the first launch died in training_step (lr 0 still built the full autograd graph: 14 GB on a V100).
## Configs now set freeze: all -> base_v2 runs batch 0 under no_grad and skips the rest; val_loss identical to lr 0.
## NB shard:2 does not cap VRAM, two jobs can land on the same GPU. Failed runs left no last.ckpt, relaunch as is.
## C0. once, on the LOGIN node (compute nodes may have no internet):
python -c "from huggingface_hub import snapshot_download as d; d('amazon/chronos-2', allow_patterns=['config.json','model.safetensors']); d('mistralai/Timestral-209M', allow_patterns=['config.json','model.safetensors'])"
export HF_HUB_OFFLINE=1    ## slurm passes the launching shell's environment to the jobs
## C1. 6 jobs (3 commands x 2 architectures)
for ds in weather electricity ettm2; do
  python train.py --config-dir=config_paper_persistence_$ds --config-name=config_abacus architecture=chronos2,timestral -m
done
## then the same compare.py (set=validation,test) as above for each dataset, incl. electricity:
python compare.py --config-dir=config_paper_persistence_electricity --config-name=compare_abacus set=validation,test -m
rsync -avz -e "ssh -F $HOME/.ssh/config_jump" --exclude='*.ckpt' --exclude='*.pkl' --exclude='*_train_*' diclubclsin:/storage3/DSIP/agobbi/Projects/ExpTS/paper_persistence_v3_electricity /home/agobbi/Projects/ExpTS/
