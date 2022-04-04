#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:30:46 2020

@author: lechenqian
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
sys.path.insert(0,'/Users/lechenqian/OneDrive - Harvard University/behavior_20_03/functions')
# from parse_data import Mouse_200310
from parse_data_2002 import Mouse_200218
from parse_data_2002 import pickle_dict
from parse_data_2002 import load_pickleddata
import seaborn as sns
#%%
def hide_spines():
    figures = [x for x in mpl._pylab_helpers.Gcf.get_all_fig_managers()]
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # Disable ticks.
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(axis = 'x' , rotation = 0 )
            
            ax.yaxis.set_ticks_position('left')
            
            
    
#%% import data
#mouse = ['C12','C13','C15','C17','C20','C22','OT-GC-2']
mouse = ['OT-GC-1','OT-GC-3']
mouse_id = mouse[1]
#load_path = 'D:/PhD/Behavior/behavior_20_02/parsed_dataframe_pickle/clean_{0}_stats.pickle'.format(mouse_id)
load_path = 'D:/PhD/Behavior/behavior_20_03/parsed_dataframe_pickle/clean_{0}_stats.pickle'.format(mouse_id)
df = load_pickleddata(load_path)

# #%%
# mpl.rcParams.update(mpl.rcParamsDefault)
#%% hist of latency of go trials  df.df_trials_lick!!!


column_name = 'latency'
trial_type = 'go'
is_save = True
save_dir = 'D:/PhD/Behavior/behavior_20_03/figures'
figurename = 'hist_latency_go'
#title fontsize
fsize = 11

#plot

fig,ax = plt.subplots(nrows=sum(df.is_cond_day), ncols=2, sharex=True,sharey=True,figsize = (8,15))
for index, day in enumerate(df.all_days[:sum(df.is_cond_day)]):
    

# select data for different trial types
#go
    index_go_trials = df.df_trials_lick[day]['trialtype'] == trial_type
    go_trials = df.df_trials_lick[day][index_go_trials]
    ax[index,0].hist(go_trials[column_name], color = '#c47898',bins = 20,range=(0,5),density=True) 
    ax[index,0].set_title('{} {}'.format(day,'condition' if df.is_cond_day[index] else 'degradation'),y=0.95, fontsize = fsize)
    #ax[index,0].spines["top"].set_visible(True)
    #ax[index,0].spines["right"].set_visible(True)
    ax[index,0].spines["top"].set_visible(False)
    ax[index,0].spines["right"].set_visible(False)
    ax[index,0].set_xlabel('Time after odor onset (s)' if index == sum(df.is_cond_day)-1 else '',fontsize = fsize)
    
offset = sum(df.is_cond_day)*2-len(df.all_days)
for index, day in enumerate(df.all_days[sum(df.is_cond_day):]):
    
     # leave top panel blank if subplots number is greater than number of training days
# select data for different trial types
#go
    index_go_trials = df.df_trials_lick[day]['trialtype'] == 'OdorReward'
    go_trials = df.df_trials_lick[day][index_go_trials]
    ax[index+offset,1].hist(go_trials[column_name], color= '#599B83',bins = 20,range=(0,5), density=True)
    ax[index+offset,1].set_title('{} {}'.format(day,'condition' if df.is_cond_day[index+sum(df.is_cond_day)] else 'degradation'),y=0.95,fontsize = fsize)
    ax[index+offset,1].spines["top"].set_visible(False)
    ax[index+offset,1].spines["right"].set_visible(False)
    ax[index+offset,1].set_xlabel('Time after odor onset (s)' if index == len(df.all_days)-sum(df.is_cond_day)-1 else '',fontsize = fsize)
    


if is_save:
    try:
        savepath = "{1}/{0}/{2}".format(df.mouse_id,save_dir,date.today())
        os.makedirs(savepath)
    except:
        pass
    

    plt.savefig("{0}/{1}.png".format(savepath,figurename), bbox_inches="tight", dpi = 100)
plt.show() 
#%% plot of licking before water and post water  df.df_trials_lick!!!


column_name1 = 'duration' # actually for NOdorReward trial, it's the last licking before the water delivery
column_name2 = 'latency'
trial_type = 'NOdorReward'
is_save = True
save_dir = 'D:/PhD/Behavior/behavior_20_03/figures'
figurename = 'plot_licking_prior_post_water'
#title fontsize
fsize = 11

#plot

fig,ax = plt.subplots(nrows=1, ncols=len(df.all_days[sum(df.is_cond_day):]), sharex=True,sharey=True,figsize = (12,3))


for index, day in enumerate(df.all_days[sum(df.is_cond_day):]):
    
     # leave top panel blank if subplots number is greater than number of training days
# select data for different trial types
#go
    index_bgrw_trials = df.df_trials_lick[day]['trialtype'] == 'NOdorReward'
    bgrw_trials = df.df_trials_lick[day][index_bgrw_trials]
    y = range(0,sum(index_bgrw_trials),1)
    prior = np.array([-x for x in bgrw_trials[column_name1]])
    post = np.array([x for x in bgrw_trials[column_name2]])
    ax[index].plot(prior,y,'o', color= '#599B83')
    ax[index].plot(post,y,'o',  color= '#500887')
    ax[index].axvline(x = 0,color = 'k')
    ax[index].set_title('{} {}'.format(day,'condition' if df.is_cond_day[index+sum(df.is_cond_day)] else 'degradation'),y=0.95,fontsize = fsize,pad = 20)
    ax[index].spines["top"].set_visible(False)
    ax[index].spines["right"].set_visible(False)
    ax[index].set_xlabel('Time after odor onset (s)',fontsize = fsize)
    


if is_save:
    try:
        savepath = "{1}/{0}/{2}".format(df.mouse_id,save_dir,date.today())
        os.makedirs(savepath)
    except:
        pass
    

    plt.savefig("{0}/{1}.png".format(savepath,figurename), bbox_inches="tight", dpi = 100)
plt.show() 
#%% hist of rate of anti licks of all four trialtypes

# choose a date
is_save = True
all_days = df.all_days
save_dir = 'D:/PhD/Behavior/behavior_20_03/figures'
figurename = 'hist_anti_rate_four_trials'
for index in range(len(all_days)):

    day = all_days[index] 
    hist = df.df_trials_lick[day].hist(figsize = (15,2),column = 'rate', by = 'trialtype',layout = (1,4), bins=20, density = True,
                                        range=(0,6),edgecolor = 'black',facecolor = '#c47898' if df.is_cond_day[index] else '#599B83',
                                        sharex = True,sharey = True)
    hide_spines()
    
    if is_save:
        try:
            savepath = "{1}/{0}/{2}".format(df.mouse_id,save_dir,date.today())
            os.makedirs(savepath)
        except:
            pass
    

    plt.savefig("{0}/{1}_{2}.png".format(savepath,day,figurename), bbox_inches="tight", dpi = 100)
    
    plt.show()



#%% licking hist over delay
is_save= True
save_dir = 'D:/PhD/Behavior/behavior_20_03/figures'
figurename = 'hist_lump_anticipatory_licking'
fig,ax = plt.subplots(nrows = int(max(sum(df.is_cond_day),len(df.all_days)-sum(df.is_cond_day))),ncols = 2,sharex = True, sharey = True, figsize = (10,20),dpi= 80)
fsize = 14
for i in range(len(df.all_days)):
    print(i)
    
    day = df.all_days[i]     
    if df.is_cond_day[i]:
        index_go_trials = df.df_trials_lick[day]['trialtype'] == 'go'
        index_goomit_trials = df.df_trials_lick[day]['trialtype'] == 'go_omit'
        cols = int(0)
        rows = int(i)
    else:
        index_go_trials = df.df_trials_lick[day]['trialtype'] == 'OdorReward'
        index_goomit_trials = df.df_trials_lick[day]['trialtype'] == 'OdorNReward'
        cols = int(1)
        offset = sum(df.is_cond_day)*2-len(df.all_days)
        rows = int(i - sum(df.is_cond_day)+offset )
        
    index_both = index_go_trials + index_goomit_trials
    both_trials = df.df_trials_lick[day][index_both] # a dataframe of a particular day that only contains go and go_omit trials
    
    lump_anti_lick = [] # since all the anti_lick are list structure
    for index, row in both_trials.iterrows():
        lump_anti_lick = lump_anti_lick+row['anti_lick']
    
    array = np.array(lump_anti_lick)-2.5 # set the go odor on as the origin
    
    newbins = np.arange(0,4,8)
    sns.distplot(array, color="dodgerblue", ax=ax[rows,cols],label="{}".format(df.all_days[i]), rug=True, rug_kws={"color": '#c47898' if df.is_cond_day[i] else "#599B83"},
                  kde_kws={"color": "k", "lw": 3},hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": '#c47898' if df.is_cond_day[i] else "#599B83"})
    ax[rows,cols].set_xlim(-0.2,3.7)
    
#    ax[index,0].set_title('{} condition:{}'.format(day,df.is_cond_day[index]),y=0.8)
    #ax[index,0].spines["top"].set_visible(True)
    #ax[index,0].spines["right"].set_visible(True)
    ax[rows,cols].spines["top"].set_visible(False)
    ax[rows,cols].spines["right"].set_visible(False)

    # Decoration
    
    ax[rows,cols].legend(loc = 2,fontsize = fsize)
    ax[rows,cols].set_xlabel('Time after odor onset (s)' if i in [sum(df.is_cond_day)-1,len(df.all_days)-1] else '',fontsize = fsize)
fig.suptitle('Density Plot of number of licking within delay', fontsize=18,y = 0.9)
if is_save:
    try:
        savepath = "{1}/{0}/{2}".format(df.mouse_id,save_dir,date.today())
        os.makedirs(savepath)
    except:
        pass
    plt.savefig("{0}/{1}.png".format(savepath,figurename), bbox_inches="tight", dpi = 100)
    

plt.show()











#%% hit rate, correct rejection rate, licking, latency, and background licking
import itertools
#choose a date
def all_in_5(df, size = 20):
# chunk dataframe     
       
    
    
    Go_lick = []
    Go_latency = []
    Bg_lick = []
    Hit = []
    Rej = []
    
    
    for i in range(len(df.all_days)):
        
        day = df.all_days[i]
        nstep = int(len(df.df_trials_lick[day]['trialtype'])/size)
        res = len(df.df_trials_lick[day]['trialtype'])% size
        print(day)
        bins = [size]*nstep
        if res != 0:
            bins.append(res)
        step_bins = np.cumsum(bins)
        
        list_go = []
        list_latency= []
        list_background = []
        list_hit = []
        list_rej = []
        
        for step in step_bins:
            anti_lick=[]
            chunk = df.df_trials_lick[day][step-size:step]
            no_correct = False
            try:
                chunk_correct = df.df_trials_iscorrect[day][step-size:step]
            except:
                no_correct = True
            
            if no_correct:
                hit_rate = np.nan
                rej_rate = np.nan
            else:   
                # hit
                hit1 = chunk_correct.loc[chunk_correct['trialtype'] == 'go' , 'is_Correct']
                hit2 = chunk_correct.loc[chunk_correct['trialtype'] == 'go_omit' , 'is_Correct']
                hit_rate = (sum(hit1)+sum(hit2))/(len(hit1)+len(hit2))
                
                # rejection
                rej = chunk_correct.loc[chunk_correct['trialtype'] == 'no_go', 'is_Correct']
                
                rej_rate = sum(rej)/len(rej)
            
            # for go trials
            
            if df.is_cond_day[i]:
                
                anti_lick = chunk.loc[chunk['trialtype'] == 'go' , 'anti_lick']
            else:
                anti_lick = chunk.loc[chunk['trialtype'] == 'OdorReward' , 'anti_lick']
            num_go_anti_lick = []
            for lick in anti_lick:
                num_go_anti_lick = num_go_anti_lick + lick
            num_go = len(anti_lick)
            
            anti_lick = []
            # for go_omit trials
            if df.is_cond_day[i]:
                anti_lick = chunk.loc[chunk['trialtype'] == 'go_omit', 'anti_lick']
            else:
                anti_lick = chunk.loc[chunk['trialtype'] == 'OdorNReward', 'anti_lick']
            
            num_goomit_anti_lick = []
            for lick in anti_lick:
                num_goomit_anti_lick = num_go_anti_lick + lick
            num_goomit = len(anti_lick)
            
            anti_lick = []
            # for go lantency 
            if df.is_cond_day[i]:
                latency = chunk.loc[chunk['trialtype'] == 'go', 'latency']
            else:
                latency = chunk.loc[chunk['trialtype'] == 'OdorReward', 'latency']
             
            num_latency = len(latency)
            
            anti_lick=[]
            # for go trials
            if df.is_cond_day[i]:
                anti_lick = chunk.loc[chunk['trialtype'] == 'background', 'anti_lick']
            else:
                anti_lick = chunk.loc[chunk['trialtype'] == 'NOdorNReward', 'anti_lick']
            num_background_anti_lick = []
            for lick in anti_lick:
                num_background_anti_lick = num_background_anti_lick + lick
            num_background = len(anti_lick)
            
            
            
            
            list_go.append((len(num_go_anti_lick)+len(num_goomit_anti_lick))/((num_go+num_goomit)*3.5))
            list_latency.append(sum(latency)/num_latency)
            try:
                list_background.append(len(num_background_anti_lick) / (num_background*11))
            except:
                list_background.append(np.nan)
            list_hit.append(hit_rate)
            list_rej.append(rej_rate)
        
        Go_lick.append(list_go)
        Go_latency.append(list_latency)
        Bg_lick.append(list_background)
        Hit.append(list_hit)
        Rej.append(list_rej)       
                
    return Go_lick,Go_latency,Bg_lick ,Hit,Rej

#%%
block = 20
is_save = True
Go_lick,Go_latency,Bg_lick ,Hit,Rej = all_in_5(df, size = block)
NEWdict = {'Go_lick':Go_lick,'Go_latency' : Go_latency,'Background_lick':Bg_lick ,'Hit':Hit,'Correct_rej':Rej}
Stats_df = pd.DataFrame(NEWdict)


figurename = 'all_in_5_size{}'.format(block)
fig,ax = plt.subplots(nrows=5, ncols=1, sharex=True,figsize = (5,14))
index1 = 0

font = {'size': 14}
markersize = 6
mpl.rc('font', **font)
for index, row in Stats_df.iterrows():
    index2 = index1 + len(row['Go_lick'])
    x  = range(index1,index2)
    
    
    ax[0].plot(x, np.array(row['Hit']),'o-',linewidth = 3,alpha = 0.8,markeredgecolor = 'k',markersize = markersize,color='#a163ab', label = 'hit' if index == 1 else '')
    ax[1].plot(x, row['Correct_rej'],'o-',linewidth = 3,alpha = 0.8,markeredgecolor = 'k',markersize = markersize,color='#e39a62',label = 'correct rejection' if index == 1 else '')
    ax[0].legend(loc='lower right',frameon = False)
    ax[1].legend(loc='lower right',frameon = False)
    ax[0].set_ylabel('Percentage(%)')
    
    ax[1].set_ylabel('Percentage(%)')
    
    ax[2].plot(x, row['Go_lick'],'o-',linewidth = 3,alpha = 0.8,markeredgecolor = 'k',markersize = markersize,color='#c47898' if df.is_cond_day[index] else '#599B83')
    ax[2].set_ylabel('Lick rate to Odor\n over delay')
    ax[3].plot(x, row['Go_latency'],'o-',linewidth = 3,alpha = 0.8,markeredgecolor = 'k',markersize = markersize,color='#c47898' if df.is_cond_day[index] else '#599B83')
    ax[3].set_ylabel('Latency to lick')
    ax[4].plot(x, row['Background_lick'],'o-',linewidth = 3,alpha = 0.8,markeredgecolor = 'k',markersize = markersize,color='#c47898' if df.is_cond_day[index] else '#599B83')
    ax[4].set_ylabel('# Baseline lick rate')
   
   
    
    index1 = index2
ax[0].set_ylim(-0.1,1.1)
ax[1].set_ylim(0,1.1)
ax[2].set_ylim(-0.1,1.5)
ax[3].set_ylim(1,7)
ax[4].set_ylim(-0.1,0.5)
ax[0].spines["top"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[2].spines["top"].set_visible(False)
ax[3].spines["top"].set_visible(False)
ax[4].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)    
ax[1].spines["right"].set_visible(False)    
ax[2].spines["right"].set_visible(False)    
ax[3].spines["right"].set_visible(False)    
ax[4].spines["right"].set_visible(False) 
plt.xticks([])
if is_save:
    try:
        savepath = "{1}/{0}/{2}".format(df.mouse_id,save_dir,date.today())
        os.makedirs(savepath)
    except:
        pass
    plt.savefig("{0}/{1}.png".format(savepath,figurename), bbox_inches="tight", dpi = 200)
    


plt.show()




#%% anti/bg ratio

figurename = 'bg_odor_ratio'
index1 = 0
Ratio = pd.Series([]) 
fig, ax = plt.subplots(figsize = (5,3))
for index, row in Stats_df.iterrows():
    index2 = index1 + len(row['Go_lick'])# anticipatory licking
    x  = range(index1,index2)
    
    bg_lick = np.array(row['Background_lick'])
    bg_lick[bg_lick <= 0] = 0.05
    odor_lick = np.array(row['Go_lick'])
    try:
        ratio = bg_lick/odor_lick
        Ratio[index] = ratio
        
        print(ratio)
    except:
        print('wrong')

    ax.plot(x, ratio,'^-',linewidth = 3,alpha = 0.8,color='#c47898' if df.is_cond_day[index] else '#599B83')
    index1 = index2
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)    
ax.legend(loc='best',frameon = False)
ax.set_ylabel('BG/Odor Ratio')
ax.set_title('ratio of background lick rate and Odor anticipatory lick rate over days', fontsize = 15,y =1.1)
if is_save:
    try:
        savepath = "{1}/{0}/{2}".format(df.mouse_id,save_dir,date.today())
        os.makedirs(savepath)
    except:
        pass
    plt.savefig("{0}/{1}.png".format(savepath,figurename), bbox_inches="tight", dpi = 200)
    
plt.show()
#%%
print(ratio)
#%%

Stats_df.insert(0, "BG/Odor Ratio", Ratio) 
#save data by pickle
#****************
save_path = 'D:/PhD/Behavior/behavior_20_03/parsed_dataframe_pickle'

filename = 'avg_stats_{}'.format(mouse_id)
pickle_dict(Stats_df,save_path,filename)

















