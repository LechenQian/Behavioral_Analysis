# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:57:26 2020

@author: qianl
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:53:56 2020

@author: qianl
"""
from datetime import datetime
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



#%%
### generate clean data

# read original eventcode
def read_filename(filedir,mouse_id):
        filedir = filedir +'/{}'.format(mouse_id)
        filenames = []
        for dirpath, dirnames, files in os.walk(filedir): # can walk through all levels down
            for f_name in files:
                if f_name.endswith('.xlsx'):
                    filenames.append(dirpath+'/'+f_name)
                    print(f_name)
        print('---------------------------------------------')    
        print('The files have been loaded from the following paths')
        
        return filenames   

def read_eventcode(file):
    
    date = re.search(r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})",file).group(0) # extract date: must be like format 2020-02-10            
    train_type = os.path.split(file)[-1][22:-5]
    
    
    data = pd.read_excel(file,header=None) #read orginal csv data  
    data.columns = ['Time','Event','Type']
    return data,date,train_type

def replace_eventype(data):
    for index, row in data.iterrows():
            if row['Type'] == 'trial_noncontR': 
                row['Type'] = 'trial4'
            elif row['Type'] == 'trial_contR': 
                row['Type'] = 'trial0'
            elif row['Type'] == 'trial_noncontNR': 
                row['Type'] = 'trial2'
            elif row['Type'] == 'trial_contNR': 
                row['Type'] = 'trial3'
    return data
            

# save the cleaned event code to clean_data in the format of excel

def save_to_excel(df,path,filename):
    try:
        os.makedirs(path) # create the path first
    except FileExistsError:
        print('the path exist.')
    filename = path +'/{}.xlsx'.format(filename)
    df.to_excel(filename, header = False, index = False, engine = 'xlsxwriter')
    print('saved!')
#%%
filedir = 'D:/PhD/Behavior/Behavior_Analysis/2020_batch_clean_selected'
mouse_id = 'C22'
all_filename = read_filename(filedir,mouse_id)
for i in range(len(all_filename)):
    
    chosen_file = all_filename[i]
    data,date,train_type = read_eventcode(chosen_file)
    print('--------------------------')
    
    print('Date: ',date)
    data['Type'] = data['Type'].replace(['trial_noncontR'],'trial4')
    data['Type'] = data['Type'].replace(['trial_contR'],'trial0')
    data['Type'] = data['Type'].replace(['trial_noncontNR'],'trial2')
    data['Type'] = data['Type'].replace(['trial_contNR'],'trial3')
    
    save_path = os.path.join(filedir,'replaced_data/{}'.format(mouse_id))
    postfix = train_type
    save_to_excel(data,save_path,'clean_{0}_{1}'.format(date,postfix))



























