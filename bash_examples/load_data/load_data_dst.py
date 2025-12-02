from dsipts import TimeSeries
import pandas as pd
import os
import numpy as np
def load_data(conf):

    dst_data = pd.read_pickle(os.path.join(conf.dataset.path,'dst.pkl'))
    dst_data['ora_round'] = dst_data.ora.apply(lambda x:int(x.split(':')[0]))
    dati_agg =  dst_data.groupby(['data','ora_round']).agg({
                                                        'BX': np.mean,
                                                        'BY': np.mean,
                                                        'BZ': np.mean,
                                                        'FLOW_SPEED': np.mean,
                                                        'PROTON_DENSITY': np.mean,
                                                        'TEMPERATURE': np.mean,
                                                        'PRESSION': np.mean,   
                                                        'ELETTRIC': np.mean,
                                                    'y': np.mean})
    dati_agg.reset_index(inplace=True)
    dati_agg.sort_values(by = ['data','ora_round'],inplace=True)
    from datetime import timedelta
    dati_agg['time'] = dati_agg.apply(lambda x:x['data']+timedelta(hours=x['ora_round'] ),axis=1)
    
    ##care here
    M = int(dati_agg.shape[0]*conf.split_params.perc_train)
    dst_min = dati_agg['y'].values[0:M]

    bins = [dst_min.min() - 10] + list(np.arange(-300, dst_min.max() + 10, 10))
    h, b = np.histogram(dst_min, bins=bins)
    if len(np.argwhere(h == 0)) > 0:
        bins = np.delete(bins, np.argwhere(h == 0)[0] + 1)
        h, b = np.histogram(dst_min, bins=bins)
    w = h.max()/h
    
    def fix_weight(dst_v):
        pos = np.argwhere(np.abs(b - dst_v) == np.abs((b - dst_v)).min())[0,0]
        if dst_v - b[pos] < 0:
            pos = pos-1
        return w[pos]/h.max()

    fix_weight_v = np.vectorize(fix_weight)    
    weights = fix_weight_v(dst_min)
    print(weights.min(), weights.max())
    dati_agg['weights'] = 1.0
    dati_agg.loc[0:M-1,'weights'] = weights
    dati_agg.drop(columns=['data','ora_round'],inplace=True)
    #dati_agg['f'] = 1
    ts = TimeSeries(conf.ts.name)


    ts.load_signal(dati_agg, enrich_cat= [],
                   target_variables=['y'],
                    #cat_past_var=['f'],
                    #cat_fut_var=['f'],
                   sampler_weights = conf.ts.get('sampler_weights',None),
                   past_variables = ['BX', 'BY', 'BZ', 'FLOW_SPEED', 'PROTON_DENSITY','TEMPERATURE', 'PRESSION', 'ELETTRIC', 'y'],
                   #future_variables = ['BX'],
                   silly_model=conf.ts.get('silly',False))

    print(ts)
    return ts