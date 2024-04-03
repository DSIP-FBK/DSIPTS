import os
from dsipts import TimeSeries
import numpy as np
import pandas as pd
def ffill(x):
    prev = 0
    for i in range(len(x)):

        if np.isnan(x[len(x)-i-1]):
            x[len(x)-i-1] = prev
        else:
            prev = x[len(x)-i-1]
    return x
def build(N=2000):
    t = np.arange(0,N,0.1)
    x1 = np.sin(t/10)
    s = np.cos(t/5)
    x2 = np.cos(t/20)
    x3 = x2+x1-1.5*x1*x2
    s1 = (s>0)*1.0
    s2 = (s>0)*1.0
    s1[s1==0]=np.nan
    x3 = ffill(x3*s1)[:-100]
    s2 = s2[:-100]
    s1 = s1[:-100]
    x1 = x1[100:]
    x2 = x2[100:]
    x = np.stack([x1,x2,s2,x3]).T
    return pd.DataFrame({'time':list(range(len(x1))),'x1':x[:,0],'x2':x[:,1],'e1':x[:,2].astype(int),'y':x[:,3]})
 

def load_data(conf):


    df = build()

    ts = TimeSeries(conf.ts.name)
    ts.load_signal(df,past_variables =['x1','x2'],
                   target_variables =['y'],
                   cat_var =['e1'],
                   enrich_cat= [],silly_model=False)

 

    return ts

 