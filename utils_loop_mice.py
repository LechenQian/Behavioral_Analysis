# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:31:27 2023

@author: qianl
"""
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
# sys.path.insert(0,os.path.join(path,'functions'))
from a1_parse_data_v2 import Mouse_data
from a1_parse_data_v2 import pickle_dict
from a1_parse_data_v2 import load_pickleddata
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy import signal
from statsmodels.gam.api import GLMGam
import statsmodels.gam.smooth_basis as sb
import statsmodels.api as sm
#%%
group = ['M-2','M-4','M-5','D1-01','D2-04','D2-02','D2-17','D1-02','DAT-01','D1-05-TDT','D1-09','D1-10','D1-13','D2-05-TDT','D2-16','D2-04','C23','C24','D1-15', 'D2-18','D2-21','D2-23','D2-24']
for mouse_id in group:
    load_path = os.path.join(path,'{0}_stats.pickle'.format(mouse_id))
    mouse = load_pickleddata(load_path)
    print(mouse_id,mouse.df_trials[mouse.all_days[0]][mouse.df_trials[mouse.all_days[0]]['trialtype'] == 'go']['go_odor'].values[0][0]
          )