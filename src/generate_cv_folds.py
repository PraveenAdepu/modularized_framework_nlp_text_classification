# -*- coding: utf-8 -*-
"""
@author: adepup
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def StratifiedFolds(df, folds, random_state):
    np.random.RandomState(random_state)
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=folds, random_state = random_state, shuffle = True)
    
    skf.get_n_splits(df, df['cv_folds_target'])
    
    df_CVindices = pd.DataFrame(columns=['index','CVindices'])

    count = 1
    for train_index, test_index in skf.split(df, df['cv_folds_target']):           
           df_index = pd.DataFrame(test_index,columns=['index']) 
           df_index['CVindices'] = count
           df_CVindices = df_CVindices.append(df_index)       
           count+=1
           
    df_CVindices.set_index('index', inplace=True)
    
    df = pd.merge(df, df_CVindices, left_index=True, right_index=True)  
    
    return df