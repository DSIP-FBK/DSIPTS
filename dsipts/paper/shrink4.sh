## Controlled drift study (2026-11-24, PLAN §9): does rolling beta track a KNOWN change of the optimal beta?
## Data: loss_normalization_wip/gen_persistent.py OUT drift (SNR 1 -> 0.25 via the signal amplitude; exact oracle):
##   drift_none (control, == persistent_snr1), drift_step (drop at the middle of test), drift_ramp (linear from validation start).
## Model: linear Simple frozen from synthpers 1j. 3 scenarios x {l1, mse} x 3 seeds = 18 jobs (2 commands of 9).
## NOT a script to run top to bottom. Configs are in git (git pull); the 3 data files go by rsync (ExpTS/data is not in the repo).

## ---- 0. LAPTOP -> cluster: the 3 data files (12 MB); configs come with git pull
rsync -avz -e "ssh -F $HOME/.ssh/config_jump" /home/agobbi/Projects/ExpTS/data/generated_drift_none.csv /home/agobbi/Projects/ExpTS/data/generated_drift_step.csv /home/agobbi/Projects/ExpTS/data/generated_drift_ramp.csv diclubclsin:/storage3/DSIP/agobbi/Projects/ExpTS/data/

## ---- 1. CLUSTER
cd /storage3/DSIP/agobbi/Projects/timeseries && git pull && cd bash_examples
source /storage3/DSIP/agobbi/Projects/timeseries/dsipts/.venv/bin/activate
ls -l /storage3/DSIP/agobbi/Projects/ExpTS/data/generated_drift_*.csv     ## 3 files, 4.2 MB each
python train.py --config-dir=config_paper_persistence_synthdrift --config-name=config_abacus architecture=simple_lin ts.name=drift_none,drift_step,drift_ramp model_configs.loss_type=l1 train_config.seed=1,2,3 -m
python train.py --config-dir=config_paper_persistence_synthdrift --config-name=config_abacus architecture=simple_lin ts.name=drift_none,drift_step,drift_ramp model_configs.loss_type=mse train_config.seed=1,2,3 -m

## ---- 2. CLUSTER, after the 18 runs: validation + test predictions (NB: launch set=validation and set=test as two
## separate commands if the bash_examples exist_ok fix is not on the cluster yet -> plots/ csv/ folder race)
python compare.py --config-dir=config_paper_persistence_synthdrift --config-name=compare_abacus set=validation,test -m

## ---- 3. LAPTOP
rsync -avz -e "ssh -F $HOME/.ssh/config_jump" --exclude='*.ckpt' --exclude='*.pkl' --exclude='*_train_*' diclubclsin:/storage3/DSIP/agobbi/Projects/ExpTS/paper_persistence_v3_synthdrift /home/agobbi/Projects/ExpTS/
