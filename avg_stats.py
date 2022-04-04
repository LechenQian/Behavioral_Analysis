# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:51:14 2020

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
sys.path.insert(0,'D:\PhD\Behavior\behavior_20_03\functions')
from parse_data import Mouse_200310
from parse_data import pickle_dict
from parse_data import load_pickleddata
sys.path.insert(0,'D:\PhD\Behavior\behavior_20_02\functions')
from parse_data_2002 import Mouse_200218

#from licking_stats_plots_2002 import all_in_5
#%%
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
#mouse_id = ['C12','C13','C15','C17','C22','C20']
mouse_id = ['OT-GC-1','OT-GC-3','C12','C13','C15','C17','C22','OT-GC-2','OT-GC-1','OT-GC-3']
for index,mouse in enumerate(mouse_id):
    if mouse in ['C12','C13','C15','C17','C22','C20','OT-GC-2']:
        load_path = 'D:/PhD/Behavior/behavior_20_02/parsed_dataframe_pickle/clean_{0}_stats.pickle'.format(mouse)
    elif mouse in ['OT-GC-1','OT-GC-3']:
        load_path = 'D:/PhD/Behavior/behavior_20_03/parsed_dataframe_pickle/clean_{0}_stats.pickle'.format(mouse)
    print(mouse)
    df = load_pickleddata(load_path)
    Go_lick,Go_latency,Bg_lick ,Hit,Rej = all_in_5(df, size = 160)
    Go_lick = [x[0] for x in Go_lick]
    Go_lick  = Go_lick/np.max(Go_lick)
    Go_latency = [x[0] for x in Go_latency]
    Go_latency = Go_latency/np.max(Go_latency)
    Bg_lick = [x[0] for x in Bg_lick]
    Hit = [x[0] for x in Hit]
    Rej = [x[0] for x in Rej]
    if mouse in ['C12','C13']:
        Group = [0.2]*len(Go_lick)
    elif mouse in ['C15','C17','C22','OT-GC-1','OT-GC-3','OT-GC-2']:
        Group = [0.8]*len(Go_lick)
    else:
        Group = [0]*len(Go_lick)
    
    if mouse == 'OT-GC-1':
        session = ['C1', 'C2', 'C3', 'C4', 'D1','D2', 'D3', 'D4', 'D5']
    elif mouse == 'OT-GC-3':
        session = ['C1', 'C2', 'C3', 'C4', 'C5','D1','D3', 'D5']
    elif mouse in ['C12','C13','C15','C17','C22','C20','OT-GC-2']:
        session = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'D1', 'D2', 'D3', 'D4', 'D5']
        
    NEWdict = {'Session':session,
               'Type':['cond']*sum(df.is_cond_day)+['deg']*(len(df.is_cond_day)-sum(df.is_cond_day)),'Group':Group,
               'Mouse':mouse,'Go_lick':Go_lick,'Go_latency' : Go_latency,
               'Background_lick':Bg_lick ,'Hit':Hit,'Correct_rej':Rej}
    
    temp = pd.DataFrame(NEWdict)
    if index <1:
        df_allmouse = pd.DataFrame(temp)
    else:
        df_allmouse = df_allmouse.append(temp)


#%%
import seaborn as sns
sessions = df_allmouse.Session.unique()   
   
for column in df_allmouse.columns:
    if column not in ['Session','Mouse','Type','Group']:
        Go_lick_mean = []
        Go_lick_std =  []  
        for session in sessions:
            Go_lick = df_allmouse.loc[df_allmouse.Session == session,column]
            
            Go_lick_mean.append(np.nanmean(Go_lick))
            Go_lick_std.append(np.nanstd(Go_lick))
    
    
        fig, ax = plt.subplots(figsize = (10,6))
        x_pos = range(0, len(Go_lick_mean))  
        
        ses_order = ['C1', 'C2', 'C3', 'C4', 'C5','C6','D1','D2', 'D3', 'D4', 'D5']
        bars = sns.barplot(x="Session", y=column,hue="Group", data=df_allmouse,ci = 68,palette = "Set2",order = ses_order)
#        for i in range(6):
 #           bars[i].set_color('pink')
#       for i in range(6,11):
#            bars[i].set_color('green')
        #bars[6:].set_color('green')
        #df_allmouse.plot(kind='scatter',x='Session',c = 'type',y=column,ax = ax)
#        ax = sns.scatterplot(x='Session', y=column, hue='Group',data=df_allmouse)

        
        ax.set_ylabel(column)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ses_order)
        #ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
        ax.yaxis.grid(False)
        ax.legend(loc=1)
        
        # Save the figure and show
        plt.tight_layout()
        plt.savefig('D:/PhD/Behavior/behavior_20_03/figures/avg/{}_{}.png'.format(column,mouse_id),dpi = 200)
        plt.show()

#%%
print(bars[1])












#%% import from avg_stats
#mouse_id = ['C12','C13','C15','C17','C22','C20']
mouse_id = ['OT-GC-1','OT-GC-3','C15','C17','C22','OT-GC-2']
for index,mouse in enumerate(mouse_id):
    if mouse in ['C12','C13','C15','C17','C22','C20','OT-GC-2']:
        load_path = 'D:/PhD/Behavior/behavior_20_02/parsed_dataframe_pickle/avg_stats_{0}.pickle'.format(mouse)
        load_path2 = 'D:/PhD/Behavior/behavior_20_02/parsed_dataframe_pickle/clean_{0}_stats.pickle'.format(mouse)
    elif mouse in ['OT-GC-1','OT-GC-3']:
        load_path = 'D:/PhD/Behavior/behavior_20_03/parsed_dataframe_pickle/avg_stats_{0}.pickle'.format(mouse)
        load_path2 = 'D:/PhD/Behavior/behavior_20_03/parsed_dataframe_pickle/clean_{0}_stats.pickle'.format(mouse)
    print(mouse)
    df = load_pickleddata(load_path)
    df2 = load_pickleddata(load_path2)
    
    Go_lick = [x[0] for x in df.Go_lick]
    Go_lick  = Go_lick/np.max(Go_lick)
    Go_latency = [x[0] for x in df.Go_latency]
    Go_latency = Go_latency/np.max(Go_latency)
    Bg_lick = [x[0] for x in df.Background_lick]
    bg_ratio = [x[0] for x in df['BG/Odor Ratio']]
    Hit = [x[0] for x in df.Hit]
    Rej = [x[0] for x in df.Correct_rej]
    # if mouse in ['C12','C13']:
    #     Group = [0.2]*len(Go_lick)
    # elif mouse in ['C15','C17','C22','OT-GC-1','OT-GC-3','OT-GC-2']:
    #     Group = [0.8]*len(Go_lick)
    # else:
    #     Group = [0]*len(Go_lick)
    
    if mouse in ['C15','C17','C22']:
        Group = ['headplate']*len(Go_lick)
    elif mouse in ['OT-GC-1','OT-GC-3','OT-GC-2']:
        Group = ['headplate+lens']*len(Go_lick)
    
    
    if mouse == 'OT-GC-1':
        session = ['C1', 'C2', 'C3', 'C4', 'D1','D2', 'D3', 'D4', 'D5']
    elif mouse == 'OT-GC-3':
        session = ['C1', 'C2', 'C3', 'C4', 'C5','D1','D3', 'D5']
    elif mouse in ['C12','C13','C15','C17','C22','C20']:
        session = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'D1', 'D2', 'D3', 'D4', 'D5']
        
    NEWdict = {'Session':session,
               'Type':['cond']*sum(df2.is_cond_day)+['deg']*(len(df2.is_cond_day)-sum(df2.is_cond_day)),'Group':Group,
               'Mouse':mouse,'Go_lick':Go_lick,'Go_latency' : Go_latency,
               'Background_lick':Bg_lick ,'Hit':Hit,'Correct_rej':Rej,'BG_odor_ratio':bg_ratio}
    
    temp = pd.DataFrame(NEWdict)
    if index <1:
        df_allmouse = pd.DataFrame(temp)
    else:
        df_allmouse = df_allmouse.append(temp)
#%%
import seaborn as sns
sessions = df_allmouse.Session.unique()   
   
for column in df_allmouse.columns:
    if column not in ['Session','Mouse','Type','Group']:
        Go_lick_mean = []
        Go_lick_std =  []  
        for session in sessions:
            Go_lick = df_allmouse.loc[df_allmouse.Session == session,column]
            
            Go_lick_mean.append(np.nanmean(Go_lick))
            Go_lick_std.append(np.nanstd(Go_lick))
    
    
        fig, ax = plt.subplots(figsize = (10,6))
        x_pos = range(0, len(Go_lick_mean))  
        
        ses_order = ['C1', 'C2', 'C3', 'C4', 'C5','C6','D1','D2', 'D3', 'D4', 'D5']
        bars = sns.barplot(x="Session", y=column,hue="Group", data=df_allmouse,ci = 68,palette = "Set2",order = ses_order)
#        for i in range(6):
 #           bars[i].set_color('pink')
#       for i in range(6,11):
#            bars[i].set_color('green')
        #bars[6:].set_color('green')
        #df_allmouse.plot(kind='scatter',x='Session',c = 'type',y=column,ax = ax)
#        ax = sns.scatterplot(x='Session', y=column, hue='Group',data=df_allmouse)

        
        ax.set_ylabel(column)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ses_order)
        #ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
        ax.yaxis.grid(False)
        ax.legend(loc=1)
        
        # Save the figure and show
        plt.tight_layout()
        plt.savefig('D:/PhD/Behavior/behavior_20_03/figures/avg/comparison_headplate_grinlens_{}_{}_by_avg_stats.png'.format(column,mouse_id),dpi = 200)
        plt.show()













































              #%% plot lick rate over days

cond_date = [ x for i,x in enumerate(df.all_days) if df.is_cond_day[i] == 1]
deg_date = [ x for i,x in enumerate(df.all_days) if df.is_cond_day[i] == 0]

# date
# for condition days: go and go_omit trials is what I care about
ave_cond = []
std_cond = []
for i in range(len(cond_date)):
    index = i
    day = cond_date[index] 
    
    
    index_go_trials = df.df_trials_lick[day]['trialtype'] == 'go'
    index_goomit_trials = df.df_trials_lick[day]['trialtype'] == 'go_omit'
    index_both = index_go_trials + index_goomit_trials
    both_trials = df.df_trials_lick[day][index_both]
    ave_lickrate = both_trials['rate'].mean()
    std_lickrate = both_trials['rate'].std()
    ave_cond.append(ave_lickrate)
    std_cond.append(std_lickrate)

x= np.arange(1,len(cond_date)+1)
y = np.asarray(ave_cond)
se = np.asarray(std_cond)

plt.figure(figsize=(16,10), dpi= 80)
plt.ylabel("lick rate/s", fontsize=16)  

plt.plot(x, y, color="white", lw=2) 
plt.fill_between(x, y-se, y+se, color="#3F5D7D")  

# Lighten borders
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(1)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(1)
plt.xticks(x, [d for d in cond_date] , fontsize=12)
plt.title("{}'s lick rate over cond days".format(df.mouse_id), fontsize=20)
plt.show()

#%%
# for degradation days
ave_cond = []
std_cond = []
for i in range(len(deg_date)):
    index = i
    day = deg_date[index] 
    
    # for condition days: go and go_omit trials is what I care about
    index_go_trials = df.df_trials_lick[day]['trialtype'] == 'OdorReward'
    index_goomit_trials = df.df_trials_lick[day]['trialtype'] == 'OdorNReward'
    index_both = index_go_trials + index_goomit_trials
    both_trials = df.df_trials_lick[day][index_both]
    ave_lickrate = both_trials['rate'].mean()
    std_lickrate = both_trials['rate'].std()
    ave_cond.append(ave_lickrate)
    std_cond.append(std_lickrate)

x= np.arange(1,len(deg_date)+1)
y = np.asarray(ave_cond)
se = np.asarray(std_cond)

plt.figure(figsize=(16,10), dpi= 80)
plt.ylabel("lick rate/s", fontsize=16)  

plt.plot(x, y, color="white", lw=2) 
plt.fill_between(x, y-se, y+se, color="#3F5D70")  

# Decorations
# Lighten borders
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(1)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(1)
plt.xticks(x, [d for d in deg_date] , fontsize=12)
plt.title("{}'s lick rate over deg days".format(df.mouse_id), fontsize=20)
plt.show()




    
        
    
    
    


