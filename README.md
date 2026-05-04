

# DSIPTS: unified library for timeseries modelling
> [!CAUTION]
[PLEASE READ CAREFULLY, check the version of PYTORCH LIGHTNING if installed](https://semgrep.dev/blog/2026/malicious-dependency-in-pytorch-lightning-used-for-ai-training/)


> [!CAUTION]
THE DOCUMENTATION, README and notebook are somehow outdated, some architectures are under review, please be patient and wait for the version 2.0.0 if you want a stable package
> [!CAUTION]
Check frequently the [CHANGELOG.md](CHANGELOG.md) file for the updates!

This library allows to:

-  load timeseries in a convenient format
-  create tool timeseries with controlled categorical features
-  load public timeseries
-  train a predictive model using different PyTorch architectures
-  define more complex structures using Modifiers (e.g. combining unsupervised learning + deep learning)

## Disclamer
The original repository is located [here](https://gitlab.fbk.eu/dsip/dsip_dlresearch/timeseries) but there is a push mirror in gitlab and you can find it [here](https://github.com/DSIP-FBK/DSIPTS/). Depending on the evolution of the library we will decide if keep both or move definitively to github.

## Library

The library can now be found also on pip [here](https://pypi.org/project/dsipts/) and in github [here](https://github.com/DSIP-FBK/DSIPTS`). The readme of the library can be found [here](dsipts/README.md).

The pip package (experimental) is available [here](https://pypi.org/project/dsipts).

The documentation is [here](https://dsip-fbk.github.io/DSIPTS/)
## Suite for training models

[Here](bash_examples/README.md) you can find useful code for training and comparing different architectures using Hydra and Omegaconf (mulitprocess, slurm cluster and optuna sweepers).