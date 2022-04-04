# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 20:20:17 2021

@author: qianl
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:30:00 2020

@author: qianl
"""


#%% import functions
from datetime import datetime
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import random
import matplotlib as mpl
import re
import csv
import pickle
import sys
sys.path.insert(0,'D:/PhD/Behavior/Behavior_Analysis/batch_clean_selected')
from parse_data_2020 import Mouse_data
from parse_data_2020 import pickle_dict
from parse_data_2020 import load_pickleddata



#%% group seperation
mouse_group = {'C12':'deg_less','C13':'deg_less','C15':'deg_more','C17':'deg_more','C20':'cond_control_noempty','C22':'deg_more',
            'C23':'cond_control','C24':'cond_control','C25':'recond','C26':'deg','C28':'recond','C29':'c_control',
            'C30':'c_control','C32':'c_control','C33':'close','C34':'far','C35':'close','C36':'far',
        'D1-01':'deg','D1-02':'deg','D1-03':'deg','DAT-01':'deg',
        'D1-05-TDT':'deg','D2-02':'deg','D2-03':'deg','D2-04':'deg','D2-05-TDT':'deg'}
mouse_id = ['C12','C13','C15','C17','C20','C22','C23','C24','C25','C26','C28','C29','C30','C32','C33','C34','C35','C36',
        'D1-01','D1-02','D1-03','DAT-01',
        'D1-05-TDT','D2-02','D2-03','D2-04','D2-05-TDT']

#%% import the dataframe back 

mouse_dataframes = {}
for mouse_name in mouse_id:
    print(mouse_name)

    path = 'D:/PhD/Behavior/Behavior_Analysis/batch_clean_selected/parsed_dataframe_pickle'
    filepath = os.path.join(path,'{}_stats.pickle'.format(mouse_name))
    data = load_pickleddata(filepath)
    counter = {'cond':0,'deg':5,'deg_less':5,'deg_more':5,'far':5,'close':5,'recond':10,'C_control':5}
    training_types = data.training_type
    print(training_types)
    for index, key in enumerate(data.df_trials):
        
        
        data_trials = data.df_trials[key]
        data_trials_iscorrect = data.df_trials_iscorrect[key].drop(columns = ['trialtype'],axis = 1)
        data_trials_lick = data.df_trials_lick[key].drop(columns = ['trialtype'],axis = 1)
        step1 = pd.concat([data_trials,data_trials_iscorrect,data_trials_lick],axis=1, sort=False)
        
        training_type = training_types[index]
        step1['status'] = training_type
        step1['group'] = mouse_group[mouse_name]
        counter[training_type] += 1
        step1['session'] = counter[training_type]
        step1['trialnumber'] = step1.index+1
        step1['is_imaging'] = len(re.findall('-',mouse_name))
        step1['mouse_name'] = mouse_name
        if (training_type not in ['cond','recond']) | (counter['cond']>5):
            step1['phase'] = 2
        elif training_type =='recond':
            step1['phase'] = 3
        else:
            step1['phase'] = 1
            
        if index <1:
            df_this_mouse = pd.DataFrame(step1)
        else:
            df_this_mouse = df_this_mouse.append(step1)
    mouse_dataframes[mouse_name] = df_this_mouse
    
    
    

#%%
from datetime import date

version = date.today()

df_list = []
for key,value in mouse_dataframes.items():
    df_list.append(value)
combined_df = pd.concat(df_list)

save_path = path
filename = 'combined_all_clean_{}'.format(version)
pickle_dict(combined_df,save_path,filename)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    