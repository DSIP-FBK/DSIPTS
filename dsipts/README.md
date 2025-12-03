

# DSIPTS: unified library for timeseries modelling
> [!CAUTION]
The documentation, README and notebook are somehow outdated, some architectures are under review, please be patient. Moreover, there will some frequent changes due to refactoring or documentation update. Wait the version 1.2.0 for more stable library (in terms of structure and documentation) or even 2.0.0 for the tests, assertions and other standard stuff.

This library allows to:

-  load timeseries in a convenient format
-  create tool timeseries with controlled categorical features
-  load public timeseries
-  train a predictive model using different PyTorch architectures
-  define more complex structures using Modifiers (e.g. combining unsupervised learning + deep learning)

## Disclamer
The original repository is located [here](https://gitlab.fbk.eu/dsip/dsip_dlresearch/timeseries) but there is a push mirror in gitlab and you can find it [here](https://github.com/DSIP-FBK/DSIPTS/). Depending on the evolution of the library we will decide if keep both or move definitively to github.


## Background

Let $X(t)$ be a multivariate timeseries, e.g. $\forall t, X(t)\in \mathbf{R}^k$ for some $k$. The vector space $\mathbf{R}^k$ can be partitioned into two disjoint sets: the categorical features $\mathcal{C}\subset \mathbf{N}^c$ and continuous features $\mathcal{W}\subset \mathbf{R}^{k-c}$. We assume that $\mathcal{C}$ is known for each $t$. Let $\mathcal{F}\subset\mathbf{R}^{f}$ be the set of known variables for each $t$, $\mathcal{P}\subset\mathbf{R}^{p}$ be the set of variables known until time $t$,  and  $\mathcal{T}\subset\mathcal{P}\subset\mathbf{R}^{s}$ the target variables. Let also define $\tau\in N$ as the number of lag for wich we want a forecast, then the aim of a predictive model is to find a function $F:\mathbf{R}^k\rightarrow\mathbf{R}^{s \times \tau}$ such as:

$$
F(\mathcal{C}(t-K,\ldots,t+\tau),\mathcal{F}(t-K,\ldots,t+\tau),\mathcal{P}(t-K,\ldots,t),\mathcal{T}(t-K,\ldots,t) ) = \mathcal{T}(t+1,\ldots,t+\tau)
$$

for some K representing the maximum past context.

In the library we adopt some convention that must be used when developing a new model:
```
y : the target variable(s)
x_num_past: the numerical past variables
x_num_future: the numerical future variables
x_cat_past: the categorical past variables
x_cat_future: the categorical future variables
idx_target: index containing the y variables in the past dataset. Can be used during the training for train a differential model
```
by default, during the dataset construction, the target variable will be added to the `x_num_past` list. Moreover the set of categorical variable can be different in the past and the future but we choose to distinguish the two parts during the forward loop for seek of generability.

During the forward process, the batch is a dictionary with some of the key showed above, remember that not all keys are always present (check it please) and build a model according. The shape of such tensor are in the form $[B,L,C]$ where $B$ indicates the batch size, $L$ the length and $C$ the number of channels.

The output of a new model must be $[B,L,C,1]$ in case of single prediction or $[B,L,C,3]$ in case you are using quantile loss.


Try to reuse some of the common keywords while building your model. After the initialization of the model you can use whatever variable you want but during the initialization please use the following conventions.
This first block maybe is common between several architectures:

---

- **past_steps** = int. THIS IS CRUCIAL and self explanatory
- **future_steps** = int. THIS IS CRUCIAL and self explanatory
- **past_channels** = len(ts.num_var). THIS IS CRUCIAL and self explanatory
- **future_channels** = len(ts.future_variables). THIS IS CRUCIAL and self explanatory
- **out_channels** = len(ts.target_variables). THIS IS CRUCIAL and self explanatory
- **embs_past** = [ts.dataset[c].nunique() for c in ts.cat_past_var]. THIS IS CRUCIAL and self explanatory. 
- **embs_fut** = [ts.dataset[c].nunique() for c in ts.cat_fut_var]. THIS IS CRUCIAL and self explanatory.
 - **use_classical_positional_encoder** = classical positioal code are done with the combination of sin/cos/exponenstial function, otherwise the positional encoding is done with the `nn.Embedding` like the other categorical variables
 - **reduction_mode** = the categorical metafeatures can be summed, averaged or stacked depending on what behavior you like more.
- **emb_dim** = int. Dimension of embedded categorical variables, the choice here is to use a constant value and let the user chose if concatenate or sum the variables
- **quantiles** =[0.1,0.5,0.9]. Quantiles for quantile loss
- **kind** =str. If there are some similar architectures with small differences maybe is better to use the same code specifying some properties (e.g. GRU vs LSTM)
- **activation** = str ('torch.nn.ReLU' default). activation function between layers (see  [pytorch activation functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity))
- **optim** = str ('torch.optim.Adam' default). optimization function see [pytorch optimization functions](https://pytorch.org/docs/stable/optim.html)
- **dropout_rate** =float. dropout rate
- **use_bn** =boolean . Use or not batch normalization
- **persistence_weight** = float . Penalization weight for persistent predictions
- **loss_type** = str . There are some other metrics implemented, see the [metric section](#metrics) for details


---
some are more specific for RNN-CONV architectures:

---
- **hidden_RNN** = int. If there are some RNN use this and the following
- **num_layers_RNN** = int.
- **kernel_size** = int. If there are some convolutional layers

---

linear:

- **hidden_size** = int. Usually the hidden dimension, for some architecture maybe you can pass the list of the dimensions
- **kind** =str. Type of linear approach

---

or attention based models:

- **d_model** = int .d_model of a typical attention layer
- **n_heads** = int .Heads
- **dropout_rate** = float. dropout
- **n_layer_encoder** = int. encoder layers
- **n_layer_decoder** = int. decoder layers
---

## Install
Clone the repo (gitlab or github)
The library is structured to work with [uv](https://github.com/astral-sh/uv). After installing `uv` just run  
```bash
uv pip install .
```
You can install also the package from pip (be sure that the python version is less than 3.12, still sperimental):
```bash
uv venv --python 3.11
uv pip install dsipts
```


## For developers
- Remember to update the `pyproject.toml`
- use `uv add` and `uv sync` for update the project 
- `uv pip install -e .` for install dsipts
- `uv build` for building it
- `uv pip install dist/dsipts-X.Y.Z-py3-none-any.whl` for checking the installation 
- generate documentation with `uv run sphinx-quickstart docs` (just the first time)
- `uv run sphinx-apidoc -o docs/source src/dsipts`
- `uv run sphinx-build -b html docs/source ../docs`
## AIM
DSIPTS uses AIM for tracking losses, parameters and other useful information. The first time you use DSIPTS you may need to initialize aim executing:
```bash
aim init
```



## Usage

Let make an example with the public weather data (you can find it [here](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or [here](https://github.com/thuml/Time-Series-Library?tab=readme-ov-file))




```python
import pandas as pd
import numpy as np
from dsipts import TimeSeries, RNN,read_public_dataset
import matplotlib.pyplot as plt
from datetime import timedelta
import logging
import sys
data, columns = read_public_dataset(PATH_TO_DATA,'weather') 
```
define then how to use the information and define the time series. You can add automatically the `hour` categorical data using the key `enrich_cat` that will be automatically added to the categorical past and categorical future list of columns:
```pyhon
use_covariates = False  #use only y in the PAST
use_future_covariate = True #suppose to have some future covariates
ts = TimeSeries('weather')
ts.load_signal( data,enrich_cat=['hour'],target_variables=['y'],past_variables=columns if use_covariates else [], future_variables=columns if use_future_covariate else [] )
fig = ts.plot() # plot the target variable(s    )
```
The most important part is the method `ts.load_signal` where the user can specify the parameters of the timeseries such as:


- **data** (pd.DataFrame) – input dataset the column indicating the time must be called time

- **enrich_cat** (List[str], optional) – it is possible to let this function enrich the dataset for example adding the standard columns: hour, dow, month and minute. Defaults to [].

- **past_variables** (List[str], optional) – list of column names of past variables not available for future times . Defaults to [].

- **future_variables** (List[str], optional) – list of future variables available for future times. Defaults to [].

- **target_variables** (List[str], optional) – list of the target variables. They will added to past_variables by default unless check_past is false. Defaults to [].

- **cat_past_var** (List[str], optional) – list of the past categorical variables. Defaults to [].

- **cat_future_var** (List[str], optional) – list of the future categorical variables. Defaults to [].

- **check_past** (bool, optional) – see target_variables. Defaults to True.

- **group** (str or None, optional) – if not None the time series dataset is considered composed by homogeneous timeseries coming from different realization (for example point of sales, cities, locations) default None (and the relative series are not split during the sample generation. Defaults to)

- **check_holes_and_duplicates** (bool, optional) – if False duplicates or holes will not checked, the dataloader can not correctly work, disable at your own risk. Defaults True

- **silly_model (bool, optional)** – if True, target variables will be added to the pool of the future variables. This can be useful to see if information passes thought the decoder part of your model (if any)



Now we can define a forecasting problem (`past_steps` as context, `future_steps` as future horizon )

Let suppose to use a RNN encoder-decoder structure, then the model has the following parameters:
```python
past_steps = 12*7
future_steps = 12
config = dict(model_configs =dict(

                                    past_steps = past_steps, #TASK DEPENDENT 
                                    future_steps = future_steps,#TASK DEPENDENT  
    
                                    emb_dim = 16, # categorical stuff
                                    use_classical_positional_encoder = True, # categorical stuff
                                    reduction_mode = 'mean',# categorical stuff
    
                                    kind = 'gru',# model dependent
                                    hidden_RNN = 12,# model dependent
                                    num_layers_RNN = 2,# model dependent
                                    kernel_size = 15,# model dependent
                                    dropout_rate= 0.5,# model dependent
                                    remove_last= True,# model dependent
                                    use_bn = False,# model dependent
                                    activation= 'torch.nn.PReLU', # model dependent
    
                                    quantiles=[0.1,0.5,0.9], #LOSS
                                    persistence_weight= 0.010, #LOSS
                                    loss_type= 'l1', #LOSS
    
                                    optim= 'torch.optim.Adam', #OPTIMIZER
    
                                    past_channels = len(ts.past_variables), #parameter that depends on the ts dataset
                                    future_channels = len(ts.future_variables), #parameter that depends on the ts dataset
                                    embs_past = [ts.dataset[c].nunique() for c in ts.cat_past_var], #parameter that depends on the ts dataset
                                    embs_fut = [ts.dataset[c].nunique() for c in ts.cat_fut_var], #parameter that depends on the ts dataset
                                    out_channels = len(ts.target_variables)),             #parameter that depends on the ts dataset
              
                scheduler_config = dict(gamma=0.1,step_size=100),
                optim_config = dict(lr = 0.0005,weight_decay=0.01))
model_rnn = RNN(**config['model_configs'],optim_config = config['optim_config'],scheduler_config =config['scheduler_config'],verbose=False )

ts.set_model(model_rnn,config=config )

```


Now we are ready to split and train our model. First define the splitting configuration:
```python
split_params = {'perc_train':0.7,'perc_valid':0.1,                             ##if not None it will split 70% 10% 20%
               'range_train':None, 'range_validation':None, 'range_test':None, ## or we can split using ranges for example range_train=['2021-02-03','2022-04-08']
               'past_steps':past_steps,
               'future_steps':future_steps,
               'starting_point':None,                                          ## do not skip samples
               'skip_step' : 10                                                ## distance between two consecutive samples, aka the stride (larger it is, less point we have in train)
                             }

 ts.train_model(dirpath=PATH_TO_SAVING_STUFF,
                   split_params=split_params,
                   batch_size=128,
                   num_workers=4,
                   max_epochs=2,
                   gradient_clip_val= 0.0,
                   gradient_clip_algorithm='value',
                   precision='bf16',
                   auto_lr_find=True)

    ts.losses.plot()
    ts.save("weather") ##save all the metadata to use it in inference mode after

```

It is possble to split the data indicating the percentage of data to use in train, validation, test or the ranges. The `shift` parameters indicates if there is a shift constucting the y array. It cab be used for some attention model where we need to know the first value of the timeseries to predict. It may disappear in future because it is misleading. The `skip_step` parameters indicates how many temporal steps there are between samples. If you need a future signal that is long `skip_step+future_steps` then you should put `keep_entire_seq_while_shifting` to True (see Informer model).

During the training phase a log stream will be generated. If a single process is spawned the log will be displayed, otherwise a file will be generated. Moreover, inside the `weight` path there wil be the `loss.csv` file containing the running losses.

At the end of the training process it is possible to load the model passing the model class (`RNN`) and the saving name used before (`weather`)
If the same model and the same name are used for defining the time series, the training procedure will continue from the last checkpoint. Due to lightening related usage, the counting of the epochs will start from the last stage (if you trained if for 10 epochs and you want to train 10 epochs more you need to change it to 20).



```python

ts.load(RNN,"weather",load_last=True)
res = ts.inference_on_set(200,4,set='test',rescaling=True)
error = res.groupby('lag').apply(lambda x: np.nanmean((x.y-x.y_median)**2)).reset_index().rename(columns={0:'error'}) 

```
If a quantile loss has been selected the model generates three signals `_low, _median, _high`, if not the output the model is indicated with `_pred`. Lag indicates which step the prediction is referred (eg. lag=1 is the first output of the model along the sequence output). 

```
import matplotlib.pyplot as plt
mask = res.prediction_time=='2020-10-19 19:50:00'   
plt.plot(res.lag[mask],res.y[mask],label='real')
plt.plot(res.lag[mask],res.y_median[mask],label='median')
plt.legend()
```
Another useful plot is the error plot per lag where it is possible to observe the increment of the error in correlation with the lag time:

```
import numpy as np
res['error'] =np.abs( res['y']-res['y_median'])
res.groupby('lag').error.mean().plot()
```



This example can be found [here](/notebooks/public_timeseries.ipynb).

# Categorical variables
Most of the models implemented can deal with categorical variables (`cat_past_var` and `cat_fut_var`). In particulare there are some variables that you don't need to computed. When declaring a `ts` obejct you can pass also the parameter `enrich_cat=['dow']` that will add to the dataframe (and to the dataloader) the day of the week. Since now you can automatically add `hour, dow, month and minute`. If there are other categorical variables pleas add it to the list while loading your data.



# Models
A description of each model can be found in the class documentation [here](https://dsip.pages.fbk.eu/dsip_dlresearch/timeseries/). 
It is possible to use one of the following architectures:

- **RNN** (GRU, LSTM or xLSTM) models, (xLSTM)[https://arxiv.org/pdf/2405.04517] are taken from the [official repo](https://github.com/muditbhargava66/PyxLSTM) 
- **Linear** models based on the [official repository](https://github.com/cure-lab/LTSF-Linear), [paper](https://arxiv.org/pdf/2205.13504.pdf). An alternative model (alinear) has been implemented that drop the autoregressive part and uses only covariates
- **Crossformer** [official repository](https://github.com/cheerss/CrossFormer), [paper](https://openreview.net/forum?id=vSVLM2j9eie)
- **Informer** [official repository](https://github.com/zhouhaoyi/Informer2020), [paper](https://arxiv.org/abs/2012.07436)
- **Autoformer** [non official repository](https://github.com/yuqinie98/PatchTST/tree/main), [paper](https://arxiv.org/abs/2106.13008)
- **PatchTST** [official repository](https://github.com/yuqinie98/PatchTST/tree/main), [paper](https://arxiv.org/abs/2211.14730)
- **Persistent** baseline model
- **TFT** [paper](https://arxiv.org/abs/1912.09363)
- **DilatedConv** dilated convolutional RNN: the transfer of knowledge between past and future is performed reusing the final hidden status of the RNN of the encoder as initial hidden status of the decoder.
- **DilatedConvED** dilated convolutional RNN with an encoder/decoder structure.

- **ITransformer**  [paper](https://arxiv.org/abs/2310.06625), [official repo](https://github.com/thuml/iTransformer)
- **TIDE**  [paper](https://arxiv.org/abs/2304.08424)
- **Samformer**  [paper](https://arxiv.org/pdf/2402.10198) [official repo](https://github.com/romilbert/samformer/tree/main?tab=MIT-1-ov-)
- **Duet**  [paper](https://arxiv.org/abs/2412.10859) [official repo](https://github.com/decisionintelligence/DUET)

These models are under review because broken or not aligned with the recent distinction between past and future categorical data:

- **Diffusion** custom [diffusion process](https://arxiv.org/abs/2102.09672) using the attention mechanism in the subnets.
- **D3VAE** adaptation of the [official repository](https://github.com/PaddlePaddle/PaddleSpatial), [paper](https://arxiv.org/abs/2301.03028)
- **VQVAE** adaptation of [vqvae for images](https://nbviewer.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb) decribed in this [paper](https://arxiv.org/abs/1711.00937) paired with [GPT](https://github.com/karpathy/minGPT) transformer.
- **VVA** like VQVAE but the tokenization step is performed using a clustering standard procedure.

## Metrics
In some cases the persistence model is hard to beat and even the more complex model can fall in the persistence trap that propagates the last seen values. 
For this reason a set of metrics can be used trying to avoid the model to get stuck in the trap. In particular we implemented: MSE, L1, sinkhorn divergence, dilated
loss, quantile loss, MDA and a couple of experimental losses for minimizing the variance or penalizing the persistency. See the base model definition in `dsipts/models/base.py` for more details.



# Bash experiment
Most of the time you want to train the models in a cluster with a GPU and command line training procedure can help speedup the process. DSIPTS leverages on OmegaConf-Hydra to to this and in the folder `bash_examples` you can find an examples. Please read the documentation [here](/bash_examples/README.md)

## Losses

- `dilated`: `persistence_weight` between 0 and 1


# Modifiers

The VVA model is composed by two steps: the first is a clusterting procedure that divides the input time series in smaller segments an performs a clustering procedure in order to associate a label for each segment. A this point the GPT models works on the sequence of labels trying to predict the next cluster id. Using the centroids of the clusters (and the variace) the final ouput is reconstructed. This pipeline is quite unusual and does not fit with the automation pipeline, but it is possible to use a `Modifier` an abstract class that has 3 methods: 
- **fit_transform**: called before startin the training process and returns the train/validation pytorch datasets. In the aforementioned model the clustering model is trained.
- **transform**: used during the inference phase. It is similar to fit_transform but without the training process
- **inverse_transform**: the output of the model are reverted to the original shape. In the VVA model the centroids are used for reconstruct the predicted timeseries.




For user only: be sure that the the CI file has pages enabled, see [public pages](https://roneo.org/en/gitlab-public-pages-private-repo/)

# Adding new models
If you want to add a model:

- extend the `Base` class in `dsipts/models`
- add the export line in the `dsipts/__init__.py` 
- add a full configuration file in `bash_examples/config_test/architecture`
- optional: add in `bash_script/utils.py` the section to initializate and load the new model
- add the modifier in `dsipts/data_structure/modifiers.py` if it is required

# Testing
See [here](/bash_examples/README.md) for the testing session.

# Logging
From version 1.1.0, Aim is used for logging all the experiments and metrics. It is quite easy to install and to use. Just go inside the main folder (`bash_exaples`) and run:
```
aim init #only the first time
aim up
```
and then open the url (http://127.0.0.1:43800)[http://127.0.0.1:43800]. It will show the model parameters, some metrics and the losses during the training procedure
![plot](bash_examples/figures/aim1.png)
 but also some prediction (the first sample of the first batch of the validation set, every 10% of the maximum number of epochs.)
 ![plot](bash_examples/figures/aim2.png)


## TODO
[ ] some models can not work in a non-autoregressive way (target past variable is required). Relax some constraints in the forward loop can help this

[ ] reduce test time 

[ ] add pre-commit hook for code checking (`ruff check --ignore E501,E722 .`)

[ ] add pre-commit hook testing

[ ] clean code and standardize documentation

[ ] check all the code in the README 

[ ] check architecture description (which model can be used under certain assumption) 

[ ] complete the classification part (loss function + inference step)

[ ] check D3VAE, it seems broken in some configurations

[ ] add hybrid models https://www.sciencedirect.com/science/article/pii/S138912862400118X

[ ] add SOFTS https://github.com/Secilia-Cxy/SOFTS/blob/main/models/SOFTS.py

[ ] add https://github.com/Hank0626/PDF/blob/main/models/PDF.py

[ ] add https://github.com/decisionintelligence/pathformer

[ ] in 1.1.5 we split the future and past categorical variables. D3VAE &^ Diffusion to be revised 

[ ] all snippet of code and notebook must be review in 1.1.5 (categorical past and future, embedding layer parameters)

