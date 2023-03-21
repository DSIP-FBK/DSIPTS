# DSIPTS: unified library for timeseries modelling

This library allows to:
1- load timeseries in a convenient format
2- create tool timeseries with controlled categorical features (additive and multiplicative)
3- load public timeseries
4- train a predictive model using different pytroch architectures

##Background

Let $X(t)$ be a multivariate timeseries, e.g. $\forall t, X(t)\in \mathcal{R}^k$ for some $k$. The vector space $\mathcal{R}^k$ can be partitioned into two disjoint sets: the categorical features $\mathcal{C}\subset \mathcal{N}^c$ and continuous features $\mathcal{W}\subset \mathcal{R}^{k-c}$. We assume that $\mathcal{C}$ is known for each $t$. Let $\mathcal{F}\mathcal{R}^{f}$ be the set of the known variable for each $t$ and  $\mathcal{T}\mathcal{R}^{s}$ the target variables. Let also define $\tau\in N$ as the number of lag for wich we want a forecast, then the aim of a predictive model is to find a function $F:\mathcal{R}^k\rightarrow\mathcal{R}^{s \times \tau}$ such as:
$$
F(\mathcal{C}(t-K,\ldots,t+\tau),\mathcal{F}(t-K,\ldots,t+\tau),\mathcal{T}(t-K,\ldots,t) ) = \mathcal{T}(t+1,\ldots,t+\tau)
$$

In the library we adopt some convention that must be used when developing a new model:
```
y : the target variable(s)
x_num_past: the numerical past variables
x_num_future: the numerical future variables
x_cat_past: the categorical past variables
x_cat_future: the categorical future variables
```
by default, during the dataset construction, the target variable will be added to the `x_num_past` list. Moreover the set of categorical variable will be the same in the past and the future but we choose to distinguish the two parts during the forward loop for seek of generability.


##How to

In a pre-generated environment install pytorch and pytorch-lightning (`pip install pytorch-lightning`) then go inside the lib folder and execute:

``
python setup.py install --force
``

##Test 
You can test your model using a tool timeseries

```
##import modules
from dsipts import Categorical,TimeSeries, RNN, Attention

###########3#define some categorical features##########

##weekly, multiplicative
settimana = Categorical('settimanale',1,[1,1,1,1,1,1,1],7,'multiplicative',[0.9,0.8,0.7,0.6,0.5,0.99,0.99])

##montly, additive (here there are only 5 month)
mese = Categorical('mensile',1,[31,28,20,10,33],5,'additive',[10,20,-10,20,0])

##spot categorical variables: in this case it occurs every 100 days and it lasts 7 days adding 10 to the original timeseries
spot = Categorical('spot',100,[7],1,'additive',[10])

##initizate a timeseries object
ts = TimeSeries('prova')
```
The baseline tool timeseries is defined as:

$$
y(t) = \left[A(t) + 10\cos{\left(\frac{50}{l \cdot \pi}\right)} \right] * M(t) + Noise
$$

where $l$ is the length of the signal, $A(t)$ correspond to all the contribution of the additive categorical variable and $M(t)$ all the multiplicative contributions.

We can now generate a timeseries of length 5000 and the cateorical features described above:
```
ts.generate_signal(noise_mean=1,categorical_variables=[settimana,mese,spot],length=5000,type=0)
```

`type=0` is the base function used. In this case the name of the time variable is `time` and the timeseries is called `signal`.

Now we can define a forecasting problem, for example using the last 100 steps for predicting the 20 steps in the future. In this case we have one time series so:
```
past_steps = 100
future_steps = 20
multioutput = False
```

Let suppose to use a RNN encoder-decoder sturcture, then the model has the following parameters:
```

config = dict(model_configs =dict(
                                    embedding_final = 16,
                                    hidden_LSTM = 256,
                                    num_layers = 2,
                                    sum_embs = True,
                                    kernel_size_encoder = 20,
                                    seq_len = past_steps,
                                    pred_len = future_steps,
                                    channels_past = len(ts.num_var),
                                    channels_future = len(ts.future_variables),
                                    embs = [ts.dataset[c].nunique() for c in ts.cat_var],
                                    quantiles=[] if multioutput else [0.1,0.5,0.9],
                                    out_channels = 2 if multioutput else 1),
                scheduler_config = dict(gamma=0.1,step_size=100),
                optim_config = dict(lr = 0.0005,weight_decay=0.01))
model_sum = RNN(**config['model_configs'],optim_config = config['optim_config'],scheduler_config =config['scheduler_config'] )
ts.set_model(model_sum,quantile = model_sum.use_quantiles,config=config )

```

Notice that there are some free parameters: `embedding_final` for example represent the dimension of the embedded categorical variable, `sum_embs` will sum all the categorical contribution otherwise it will concatenate them. It is possible to use a quantile loss, specify some parameters of the scheduler (StepLR) and optimizer parameters (Adam). 
**TODO** Use omegaconf!

Now we are ready to split and train our model using:
```
ts.train_model(dirpath=<path to weights>,perc_train=0.6, perc_valid=0.2,past_steps = past_steps,future_steps=future_steps, range_train=None, range_validation=None, range_test=None,shift = 0,batch_size=100,num_workers=4,max_epochs=40,auto_lr_find=True,starting_point=None)
```
It is possble to split the data indicating the percentage of data to use in train, validation, test or the ranges. The `shift` parameters indicates if there is a shift





ATTENTION: se metto shift 0 e non metto il target nelle feature future lui non usa la y per predirre se stesso
se metto shift 1 e metto nel target lui usa le info categoriche del timestamp prima il che non mi sembra ragionevole ma non ho molte idee migliori per ora