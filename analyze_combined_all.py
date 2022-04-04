# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:34:53 2020

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
import seaborn as sns
import matplotlib as mpl
import re
import csv
import pickle
import sys
sys.path.insert(0,'D:/PhD/Behavior/Behavior_Analysis/batch_clean_selected')
from parse_data_2020 import Mouse_data
from parse_data_2020 import pickle_dict
from parse_data_2020 import load_pickleddata

#%% load combined dataframes
path = 'D:/PhD/Behavior/Behavior_Analysis/batch_clean_selected/parsed_dataframe_pickle'
filepath = os.path.join(path,'combined_all_clean_2021-03-24.pickle')
data = load_pickleddata(filepath)
savefig_path = 'D:/PhD/Behavior/Behavior_Analysis/batch_clean_selected/figures/final_analysis/{}'.format(date.today())

# groups I have 'cond_control','deg','c_control','far','close','recond','deg_less','deg_more'

#%% normalization
new_data_rate_antici = data[(data['status']=='cond')&(data['session']<=5)&(data['trialtype'].isin(['go','go_omit']))]
data1  =new_data_rate_antici.groupby(by = ['mouse_name','session']).mean()
max_anti_lick = {}
data['norm_rate_antici'] = 0
for mouse_name in data['mouse_name'].unique():
    max_lick = data1.loc[pd.IndexSlice[mouse_name,:], 'rate_antici'].max()
    max_anti_lick[mouse_name] = max_lick
    new_array = data[data['mouse_name']==mouse_name]['rate_antici']/max_anti_lick[mouse_name]
    data.loc[data['mouse_name']==mouse_name,'norm_rate_antici'] = new_array
    
#%% comparison bettween control and deg
    
x = 'session'
# y = 'norm_rate_antici'
y = 'latency_to_rew'
#%%
data_deg_cond = data[(data['group'].isin(['cond_control','deg','recond']))&(data['trialtype'].isin(['go','go_omit']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})

data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase'])[y].mean().reset_index()

sns.set_context("talk", font_scale=0.9)
sns.color_palette("Set2")

fig,ax = plt.subplots(1,1,figsize = (5,4))
# palette = sns.color_palette("mako_r", 2)
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',style = 'phase',ax = ax,markers = True,err_style="bars",
             ci=68, palette='Set2')
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()
plt.savefig(os.path.join(savefig_path,'control_vs_deg(recond)_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'control_vs_deg(recond)_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'control_vs_deg(recond)_{}.png'.format(y)), bbox_inches="tight", dpi = 400)


#%% comparison bettween control deg and recond
data_deg_cond = data[(data['group'].isin(['cond_control','deg','recond']))&(data['trialtype'].isin(['go','go_omit']))]
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase'])[y].mean().reset_index()

fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',style = 'phase',ax = ax,
             markers = True,err_style="bars",palette='Set2',
             ci=68,)
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()
plt.savefig(os.path.join(savefig_path,'control_vs_deg_vs_recond_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'control_vs_deg_vs_recond_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'control_vs_deg_vs_recond_{}.png'.format(y)), bbox_inches="tight", dpi = 400)
    

#%% comparison bettween control deg and c_control
data_deg_cond = data[(data['group'].isin(['cond_control','deg','c_control','recond']))&(data['trialtype'].isin(['go','go_omit']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase'])[y].mean().reset_index()

fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',style = 'phase',ax = ax,
             markers = True,err_style="bars",palette='Set2',
             ci=68,)
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()

plt.savefig(os.path.join(savefig_path,'control_vs_deg(recond)_vs_ccontrol_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'control_vs_deg(recond)_vs_ccontrol_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'control_vs_deg(recond)_vs_ccontrol_{}.png'.format(y)), bbox_inches="tight", dpi = 400)


#%% comparison bettween deg far and close groups
data_deg_cond = data[(data['group'].isin(['cond_control','deg','far','close','recond']))&(data['trialtype'].isin(['go','go_omit']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})
data_deg_cond['group'] = data_deg_cond['group'].replace({'close': 'long interval'})
data_deg_cond['group'] = data_deg_cond['group'].replace({'far': 'short interval'})
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase'])[y].mean().reset_index()

fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',style = 'phase',ax = ax,
             markers = True,err_style="bars",palette='Set2',
             ci=68,)
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()

plt.savefig(os.path.join(savefig_path,'control_deg(recond)_vs_far_vs_close_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'control_deg(recond)_vs_far_vs_close_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'control_deg(recond)_vs_far_vs_close_{}.png'.format(y)), bbox_inches="tight", dpi = 400)
#%%
data_deg_cond = data[(data['group'].isin(['cond_control','deg','recond']))&(data['trialtype'].isin(['go','go_omit']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})
data_deg_cond['is_imaging'] = data_deg_cond['is_imaging'].replace({2: 'True'})
data_deg_cond['is_imaging'] = data_deg_cond['is_imaging'].replace({1: 'True'})
data_deg_cond['is_imaging'] = data_deg_cond['is_imaging'].replace({0: 'False'})
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase','is_imaging'])[y].mean().reset_index()
fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',style = 'phase',size = 'is_imaging',ax = ax,
             markers = True,err_style="bars",palette='Set2',
             ci=68,)
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()
plt.savefig(os.path.join(savefig_path,'imaging_control_vs_deg(recond)_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'imaging_control_vs_deg(recond)_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'imaging_control_vs_deg(recond)_{}.png'.format(y)), bbox_inches="tight", dpi = 400)

#%% correction comparison bettween all
data_deg_cond = data[(data['group'].isin(['cond_control','deg','c_control','far','close','recond']))&(data['trialtype'].isin(['go','go_omit']))]

fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = 'session',y = 'is_Correct',hue = 'status',style = 'phase',ax = ax)
plt.legend(frameon=False,fontsize = 8,bbox_to_anchor = (0.12,0.6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#%%
x = 'session'
y = 'is_Correct'
data_deg_cond = data[(data['group'].isin(['cond_control','deg','recond']))&(data['trialtype'].isin(['go','go_omit']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})
fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',style = 'phase',size = 'is_imaging',ax = ax)
plt.legend(frameon=False,fontsize = 8)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)   
plt.savefig(os.path.join(savefig_path,'imaging_control_vs_deg(recond)_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'imaging_control_vs_deg(recond)_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'imaging_control_vs_deg(recond)_{}.png'.format(y)), bbox_inches="tight", dpi = 400)

#%% background data
    
x = 'session'
y = 'norm_rate_antici'
#%%
data_deg_cond = data[(data['group'].isin(['cond_control','deg','recond']))&(data['trialtype'].isin(['background']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})

data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase'])[y].mean().reset_index()

sns.set_context("talk", font_scale=0.9)


fig,ax = plt.subplots(1,1,figsize = (5,4))
# palette = sns.color_palette("mako_r", 2)
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',style = 'phase',ax = ax,markers = True,err_style="bars",
             ci=68, palette='Accent')
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()
plt.savefig(os.path.join(savefig_path,'bg_control_vs_deg(recond)_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'bg_control_vs_deg(recond)_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'bg_control_vs_deg(recond)_{}.png'.format(y)), bbox_inches="tight", dpi = 400)


#%% comparison bettween control deg and recond
data_deg_cond = data[(data['group'].isin(['cond_control','deg','recond']))&(data['trialtype'].isin(['background']))]
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase'])[y].mean().reset_index()

fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',style = 'phase',ax = ax,
             markers = True,err_style="bars",
             ci=68,palette='Accent')
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()
# plt.savefig(os.path.join(savefig_path,'bg_control_vs_deg_vs_recond_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
# plt.savefig(os.path.join(savefig_path,'bg_control_vs_deg_vs_recond_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
# plt.savefig(os.path.join(savefig_path,'bg_control_vs_deg_vs_recond_{}.png'.format(y)), bbox_inches="tight", dpi = 400)
    

#%% comparison bettween control deg and c_control
data_deg_cond = data[(data['group'].isin(['cond_control','deg','c_control','recond']))&(data['trialtype'].isin(['background']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase'])[y].mean().reset_index()

fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',style = 'phase',ax = ax,
             markers = True,err_style="bars",palette='Accent',
             ci=68,)
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()

plt.savefig(os.path.join(savefig_path,'bg_control_vs_deg(recond)_vs_ccontrol_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'bg_control_vs_deg(recond)_vs_ccontrol_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'bg_control_vs_deg(recond)_vs_ccontrol_{}.png'.format(y)), bbox_inches="tight", dpi = 400)


#%% comparison bettween deg far and close groups
data_deg_cond = data[(data['group'].isin(['cond_control','deg','far','close','recond']))&(data['trialtype'].isin(['background']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})
data_deg_cond['group'] = data_deg_cond['group'].replace({'close': 'long interval'})
data_deg_cond['group'] = data_deg_cond['group'].replace({'far': 'short interval'})
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase'])[y].mean().reset_index()

fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',style = 'phase',ax = ax,
             markers = True,err_style="bars",palette='Accent',
             ci=68,)
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()

# plt.savefig(os.path.join(savefig_path,'bg_control_deg(recond)_vs_far_vs_close_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
# plt.savefig(os.path.join(savefig_path,'bg_control_deg(recond)_vs_far_vs_close_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
# plt.savefig(os.path.join(savefig_path,'bg_control_deg(recond)_vs_far_vs_close_{}.png'.format(y)), bbox_inches="tight", dpi = 400)
#%%
data_deg_cond = data[(data['group'].isin(['cond_control','deg','recond']))&(data['trialtype'].isin(['background']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})
data_deg_cond['is_imaging'] = data_deg_cond['is_imaging'].replace({2: 'True'})
data_deg_cond['is_imaging'] = data_deg_cond['is_imaging'].replace({1: 'True'})
data_deg_cond['is_imaging'] = data_deg_cond['is_imaging'].replace({0: 'False'})
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase','is_imaging'])[y].mean().reset_index()
fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',style = 'phase',size = 'is_imaging',ax = ax,
             markers = True,err_style="bars",palette='Accent',
             ci=68,)
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
# sns.despine()
plt.savefig(os.path.join(savefig_path,'bg_imaging_control_vs_deg(recond)_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'bg_imaging_control_vs_deg(recond)_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'bg_imaging_control_vs_deg(recond)_{}.png'.format(y)), bbox_inches="tight", dpi = 400)

#%% go & no-go data
    
x = 'session'
y = 'norm_rate_antici'
# y = 'is_Correct'
#%%
data_deg_cond = data[(data['group'].isin(['cond_control','deg','recond']))&(data['trialtype'].isin(['go','no_go']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})

data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase','trialtype'])[y].mean().reset_index()

sns.set_context("talk", font_scale=0.9)


fig,ax = plt.subplots(1,1,figsize = (5,4))
# palette = sns.color_palette("mako_r", 2)
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',size = 'trialtype',style = 'phase',ax = ax,markers = True,err_style="bars",
             ci=68, palette='Accent')
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()
plt.savefig(os.path.join(savefig_path,'go&nogo_control_vs_deg(recond)_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'go&nogo_control_vs_deg(recond)_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'go&nogo_control_vs_deg(recond)_{}.png'.format(y)), bbox_inches="tight", dpi = 400)


#%% comparison bettween control deg and recond
data_deg_cond = data[(data['group'].isin(['cond_control','deg','recond']))&(data['trialtype'].isin(['go','no_go']))]
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase','trialtype'])[y].mean().reset_index()

fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',size = 'trialtype', style = 'phase',ax = ax,
             markers = True,err_style="bars",
             ci=68,palette='Accent')
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()
# plt.savefig(os.path.join(savefig_path,'go&nogo_control_vs_deg_vs_recond_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
# plt.savefig(os.path.join(savefig_path,'go&nogo_control_vs_deg_vs_recond_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
# plt.savefig(os.path.join(savefig_path,'go&nogo_control_vs_deg_vs_recond_{}.png'.format(y)), bbox_inches="tight", dpi = 400)
    

#%% comparison bettween control deg and c_control
data_deg_cond = data[(data['group'].isin(['cond_control','deg','c_control','recond']))&(data['trialtype'].isin(['go','no_go']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase','trialtype'])[y].mean().reset_index()

fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',size = 'trialtype',style = 'phase',ax = ax,
             markers = True,err_style="bars",palette='Accent',
             ci=68,)
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()

# plt.savefig(os.path.join(savefig_path,'go&nogo_control_vs_deg(recond)_vs_ccontrol_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
# plt.savefig(os.path.join(savefig_path,'go&nogo_control_vs_deg(recond)_vs_ccontrol_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
# plt.savefig(os.path.join(savefig_path,'go&nogo_control_vs_deg(recond)_vs_ccontrol_{}.png'.format(y)), bbox_inches="tight", dpi = 400)


#%% comparison bettween deg far and close groups
data_deg_cond = data[(data['group'].isin(['cond_control','deg','far','close','recond']))&(data['trialtype'].isin(['go','no_go']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})
data_deg_cond['group'] = data_deg_cond['group'].replace({'close': 'long interval'})
data_deg_cond['group'] = data_deg_cond['group'].replace({'far': 'short interval'})
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase','trialtype'])[y].mean().reset_index()

fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.lineplot(data = data_deg_cond,x = x,y = y,hue = 'group',size = 'trialtype',style = 'phase',ax = ax,
             markers = True,err_style="bars",palette='Accent',
             ci=68,)
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()

# plt.savefig(os.path.join(savefig_path,'bg_control_deg(recond)_vs_far_vs_close_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
# plt.savefig(os.path.join(savefig_path,'bg_control_deg(recond)_vs_far_vs_close_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
# plt.savefig(os.path.join(savefig_path,'bg_control_deg(recond)_vs_far_vs_close_{}.png'.format(y)), bbox_inches="tight", dpi = 400)
#%%
x = 'session'
y = 'norm_rate_antici'
data_deg_cond = data[(data['group'].isin(['cond_control','deg','recond']))&(data['trialtype'].isin(['go','no_go']))&(data['phase']<3)]
data_deg_cond['group'] = data_deg_cond['group'].replace({'recond': 'deg'})
data_deg_cond['is_imaging'] = data_deg_cond['is_imaging'].replace({2: 'True'})
data_deg_cond['is_imaging'] = data_deg_cond['is_imaging'].replace({1: 'True'})
data_deg_cond['is_imaging'] = data_deg_cond['is_imaging'].replace({0: 'False'})
data_deg_cond =  data_deg_cond.groupby(['mouse_name','session','group','phase','is_imaging','trialtype'])[y].mean().reset_index()
fig,ax = plt.subplots(1,1,figsize = (5,4))
sns.relplot(data = data_deg_cond,x = x,y = y,col='group',size = 'trialtype',style = 'phase',hue = 'is_imaging',ax = ax,
             markers = True,err_style="bars",palette='Set1',kind = 'line',
             ci=68,)
plt.legend(frameon=False,fontsize = 12,bbox_to_anchor=(1.01, 1))
sns.despine()
plt.savefig(os.path.join(savefig_path,'go&nogo_imaging_control_vs_deg(recond)_{}.pdf'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'go&nogo_imaging_control_vs_deg(recond)_{}.eps'.format(y)), bbox_inches="tight", dpi = 400)
plt.savefig(os.path.join(savefig_path,'go&nogo_imaging_control_vs_deg(recond)_{}.png'.format(y)), bbox_inches="tight", dpi = 400)

















#%%

# #%% comparison bettween control and deg
# data_deg_cond = data[(data['group'].isin(['cond_control','deg']))&(data['trialtype'].isin(['go','go_omit']))]
# data_deg_cond_mean = data_deg_cond.groupby(by = ['group','mouse_name']).mean()
# #%%
# fig,ax = plt.subplots(1,1,figsize = (5,3))
# sns.lineplot(data = data_deg_cond,x = 'session',y = 'rate_antici',hue = 'group',style = 'phase',ax = ax)
# plt.legend(frameon=False,fontsize = 8)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# #%% comparison bettween control deg and c_control
# data_deg_cond = data[(data['group'].isin(['cond_control','deg','c_control']))&(data['trialtype'].isin(['go','go_omit']))]
# data_deg_cond_mean = data_deg_cond.groupby(by = ['group','mouse_name']).mean()

# fig,ax = plt.subplots(1,1,figsize = (5,3))
# sns.lineplot(data = data_deg_cond,x = 'session',y = 'rate_antici',hue = 'group',style = 'phase',ax = ax)
# plt.legend(frameon=False,fontsize = 8)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# #%% comparison bettween control deg and recond
# data_deg_cond = data[(data['group'].isin(['cond_control','deg','recond']))&(data['trialtype'].isin(['go','go_omit']))]
# data_deg_cond_mean = data_deg_cond.groupby(by = ['group','mouse_name']).mean()

# fig,ax = plt.subplots(1,1,figsize = (5,3))
# sns.lineplot(data = data_deg_cond,x = 'session',y = 'rate_antici',hue = 'group',style = 'phase',ax = ax)

# # for t,l in zip(legend.texts,'')

# plt.legend(frameon=False,fontsize = 8,bbox_to_anchor = (0.35,0.45))
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)




