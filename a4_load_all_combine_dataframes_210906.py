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
sys.path.insert(0,'D:/PhD/Behavior/Behavior_Analysis/batch_clean_database/functions/')
from a1_parse_data_v2 import Mouse_data
from a1_parse_data_v2 import pickle_dict
from a1_parse_data_v2 import load_pickleddata



#%% group seperation
mouse_group = {'C12':'deg_less','C13':'deg_less','C15':'deg_more','C17':'deg_more',
               'C20':'cond_control_noempty','C22':'deg_more',
                'C23':'cond_control','C24':'cond_control','C25':'recond',
                'C26':'deg','C28':'recond',
                'C29':'c_control_extra','C30':'c_control_extra','C32':'c_control_extra',
                'C33':'close',
                'C34':'far','C35':'close','C36':'far',
                'C50':'cond_control','C52':'cond_control', 'C53':'cond_control','C54':'cond_control',
                'C56':'deg','C57':'deg',
                'C60':'double_A','C61':'double_A','C62':'double_A',
                'D1-01':'deg','D1-02':'deg','D1-03':'deg','DAT-01':'deg',
                'D1-05-TDT':'deg','D1-09':'deg','D1-10':'deg','D1-12':'deg','D1-13':'deg','D1-15':'cond_control',
                'D2-02':'deg','D2-03':'deg','D2-04':'deg','D2-05-TDT':'deg',
                'D2-16':'deg','D2-17':'deg','D2-18':'cond_control',
                'D2-21':'cond_control','D2-23':'cond_control','D2-24':'cond_control',
                'M-1':'c_odor','M-2':'c_odor','M-4':'c_odor','M-5':'c_odor',
                'D1-280':'c_control_extra','D1-300':'c_control_extra',
        }
mouse_id = [token for token in mouse_group.keys()]

#%% import the dataframe back 

mouse_dataframes = {}
for mouse_name in mouse_id:
    print(mouse_name)

    path = 'D:/PhD/Behavior/Behavior_Analysis/batch_clean_database/parsed_dataframe_pickle'
    filepath = os.path.join(path,'{}_stats.pickle'.format(mouse_name))
    data = load_pickleddata(filepath)
    counter = {'cond':0,'deg':5,'deg_less':5,'deg_more':5,'far':5,'close':5,'recond':10,'C_control':5 ,'double_control':5}
    training_types = data.training_type
    if len(training_types[0]) > 8:
        training_types = [i[7:] for i in training_types]
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    