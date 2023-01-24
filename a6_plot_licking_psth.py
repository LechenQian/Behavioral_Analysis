# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:24:37 2023

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
deg_mice = ['D1-05-TDT','D1-10','D1-13','D2-05-TDT','D2-02','D2-04','D1-09','D2-16']
control_mice = ['C23','C24','D1-15', 'D2-18','D2-21','D2-23','D2-24'] #还有两只
c_mice = ['M-2','M-4','M-5']
c_extra_mice = ['C29','C30','C32','D1-280','D1-300']
double_a_mice = ['C60','C61','C62'] # not good


#%%
# mice = ['FgDA_01','FrgD1_01','FrgD2_01','FrgD2_02']



def create_binned_licking_for_group(group,TT,path,uplim = 9,lowlim = 0,sample_points = 45):
    licking_data = {}
    
    
    # get trials
    
    
    for mouse_id in group:
        load_path = os.path.join(path,'{0}_stats.pickle'.format(mouse_id))
        mouse = load_pickleddata(load_path)
        
        #event plot with trials and iscorerct data
        
        # assign two df 
        mouse_trials = mouse.df_trials
        
        # choose a date
        all_days = mouse.all_days.copy()
        licking_data[mouse_id] = {}
        for trialtype in TT:
            
            licking_data[mouse_id][trialtype] = np.full([160,sample_points,len(all_days)], np.nan)       
            for index in range(len(all_days)):
                day = all_days[index] 
                dataframe = mouse_trials[day].copy()
                is_x_TT = dataframe['trialtype'] == trialtype #or 'go_omit' # index of go trials
                if len(is_x_TT) == 0:
                    continue
                data_xTT = dataframe[is_x_TT]
                num_trial = np.sum(is_x_TT)
                for trial in range(num_trial):
                    lickings = data_xTT['licking'].values[trial]
                    if len(lickings) == 0:
                        binned_licks = np.histogram(lickings,bins = sample_points,range = (lowlim,uplim))[0]/((uplim-lowlim)/sample_points)               
                    else:
                        if trialtype in ['go','go_omit']:#c-reward, c-omit
                            lickings = [i + 2 - data_xTT['go_odor'].values[trial][0] for i in lickings if i <uplim]
                        elif trialtype in ['no_go']:
                            lickings = [i + 2 - data_xTT['nogo_odor'].values[trial][0] for i in lickings if i <uplim]
                        elif trialtype == 'UnpredReward':
                            lickings = [i + 5.5 - data_xTT['water_on'].values[trial] for i in lickings if i <uplim]
                        else:
                            lickings = [i for i in lickings if i <uplim]
                        binned_licks = np.histogram(lickings,bins = sample_points,range = (lowlim,uplim))[0]/((uplim-lowlim)/sample_points) 
                  
                    licking_data[mouse_id][trialtype][trial,:,index] = binned_licks
    return licking_data

def created_average_licking_psth_for_group(licking_data, TTs, session_num = 10,sample_points = 45):
    num_mice = len(licking_data)
    
    mean_licking_bymouse_dict = {'go':np.zeros([num_mice,sample_points,session_num]),
                                 'no_go':np.zeros([num_mice,sample_points,session_num]), 
                                'go_omit':np.zeros([num_mice,sample_points,session_num]), 
                                 'unpred_water':np.zeros([num_mice,sample_points,session_num]),
                                 'background':np.zeros([num_mice,sample_points,session_num])}    
    for i,mouse_name in enumerate(licking_data.keys()):
        print(mouse_name)
        for TT in TTs:
            lick_trace = licking_data[mouse_name][TT]
            mean_lick_trace = np.nanmean(lick_trace[:,:,:session_num],axis = 0)
            mean_licking_bymouse_dict[TT][i,:,:] = mean_lick_trace
    return mean_licking_bymouse_dict
#%% create bined licking data for groups

path = 'D:/PhD/Behavior/Behavior_Analysis/batch_clean_database/parsed_dataframe_pickle'
TT = ['go', 'no_go', 'go_omit', 'background','unpred_water']

deg_mice_lick = create_binned_licking_for_group(deg_mice,TT,path,uplim = 9,lowlim = 0,sample_points = 45)
control_mice_lick = create_binned_licking_for_group(control_mice,TT,path,uplim = 9,lowlim = 0,sample_points = 45)
c_mice_lick = create_binned_licking_for_group(c_mice,TT,path,uplim = 9,lowlim = 0,sample_points = 45)
# c_extra_mice_lick = create_binned_licking_for_group(c_extra_mice,TT,path,uplim = 9,lowlim = 0,sample_points = 45)


# average_licking by group
deg_mice_lick_dict = created_average_licking_psth_for_group(deg_mice_lick, TT, session_num = 10,sample_points = 45)
control_mice_lick_dict = created_average_licking_psth_for_group(control_mice_lick, TT, session_num = 10,sample_points = 45)
c_mice_lick_dict = created_average_licking_psth_for_group(c_mice_lick, TT, session_num = 10,sample_points = 45)
# c_extra_mice_lick = created_average_licking_psth_for_group(c_extra_mice_lick, TT, session_num = 10,sample_points = 45)
   #%%         
# save_path = 'D:/PhD/Photometry/results/lickings'
# filename = 'licking_data'
# pickle_dict(licking_data,save_path,filename)           
#%% single plot average licking trace

fig = plt.figure()


session_of_phase = 4

length = 45
trialtypes_full = ['go','no_go','unpred_water','go_omit','background']

def plot_single_session_average_licking(data, color,legends,CS,US,std_on = False,figsize = (5,4),ylim=[-1,7],savename = 'temp'):
    
    fig,ax = plt.subplots(figsize = (figsize[0],figsize[1]))
    
    plt.xticks(np.arange(0,45,5),np.arange(-2,7,1))
    plt.xlabel('Time from odor onset(s)')
    plt.ylabel('licking rate (/s)')
    ax.set_ylim([ylim[0],ylim[1]])
    
    for item,c,l in zip(data,color,legends):
        # filled area
        num = item.shape[0]
        aa = np.nanmean(item,axis = 0)
        ax.plot(aa,color = c,label = l,lw = 1.7)
        if std_on:
            std = np.nanstd(item,axis = 0)
            ax.fill_between(np.arange(0,45,1), aa-std/num, aa+std/num,alpha = 0.1,color = 'grey') ## shall I devide this by num?
        ymin, ymax = ax.get_ylim()
    if CS:       
        # vertical lines
        ax.vlines(x=10, ymin=ymin, ymax=ymax, colors='tab:grey', ls='--', lw=2)
        ax.vlines(x=15, ymin=ymin, ymax=ymax, colors='tab:grey', ls='--', lw=2)
    if US:     
        # vertical lines
        ax.vlines(x=27.5, ymin=ymin, ymax=ymax, colors='tab:grey', ls='--', lw=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.legend(frameon=False,fancybox=False,bbox_to_anchor=(1, 0.8),)
    savepath = 'D:/PhD/Figures/lickings'
    plt.savefig("{0}/{1}_mean_licking_by_group_{2}.png".format(savepath,savename,date.today()), bbox_inches="tight", dpi = 300)
    plt.savefig("{0}/{1}_mean_licking_by_group_{2}.eps".format(savepath,savename,date.today()), bbox_inches="tight", dpi = 300)
    
    plt.show()
    
ymax = 10
# plot_single_session_average_licking(data = [deg_mice_lick_dict['go'],deg_mice_lick_dict['no_go'],deg_mice_lick_dict['go_omit']],
#                                     session_id = 4,color = ['#ffcb0d','#2e2e2d','#ffe28a'],
#                                   CS = True,US = True,
#                                   figsize = (5,4),ylim=[-1,ymax])  

plot_single_session_average_licking(data = [deg_mice_lick_dict['go'][:,:,4],deg_mice_lick_dict['go'][:,:,9],
                                            control_mice_lick_dict['go'][:,:,4],control_mice_lick_dict['go'][:,:,9],
                                            c_mice_lick_dict['go'][:,:,4],c_mice_lick_dict['go'][:,:,8],],
                                    color = ['#ff9203','#fcb85d','#54e3a3','#a1e3c5','#7786f7','#b1b9f0'],
                                    legends = ['deg group after conditioning','deg group after degradation',
                                               'cond group after conditioning','cond group after degradation',
                                               'C odor group after conditioning','C odor group after degradation'],
                                  CS = True,US = True,
                                  figsize = (7,4),ylim=[-1,ymax],
                                  std_on = False,
                                  savename = 'deg cond c before and after degradation')  

plot_single_session_average_licking(data = [deg_mice_lick_dict['go'][:,:,4],deg_mice_lick_dict['go'][:,:,9],
                                            control_mice_lick_dict['go'][:,:,4],control_mice_lick_dict['go'][:,:,9],
                                    deg_mice_lick_dict['no_go'][:,:,4],deg_mice_lick_dict['no_go'][:,:,9],
                                            control_mice_lick_dict['no_go'][:,:,4],control_mice_lick_dict['no_go'][:,:,9]],
                                    color = ['#ff9203','#fcb85d','#54e3a3','#a1e3c5','grey','grey','grey','grey'],
                                    legends = ['deg go after conditioning','deg go after degradation',
                                               'cond go after conditioning','cond go after degradation',
                                               'no go','','',''],
                                  CS = True,US = True,
                                  figsize = (7,4),ylim=[-1,ymax],
                                  std_on = False,
                                  savename = 'deg cond go and nogo before and after degradation')  

plot_single_session_average_licking(data = [control_mice_lick_dict['go'][:,:,4],control_mice_lick_dict['go_omit'][:,:,4],
                                            control_mice_lick_dict['no_go'][:,:,4]],
                                    color = ['#54e3a3','#54d0e3','grey',],
                                    legends = ['cond go after conditioning','cond go omission','cond no go'],
                                  CS = True,US = True,
                                  figsize = (7,4),ylim=[-1,ymax],
                                  std_on = True,
                                  savename = 'cond go go omit and nogo after conditioning')  

plot_single_session_average_licking(data = [deg_mice_lick_dict['go'][:,:,4],deg_mice_lick_dict['go'][:,:,9],
                                            deg_mice_lick_dict['no_go'][:,:,4],deg_mice_lick_dict['unpred_water'][:,:,9]],
                                    color = ['#ff9203','#fcb85d','grey','#c074f2'],
                                    legends = ['deg go after conditioning','deg go after degradation',
                                               'deg no go','deg unpredicted water'],
                                  CS = True,US = True,
                                  figsize = (7,4),ylim=[-1,ymax],
                                  std_on = False,
                                  savename = 'def go and nogo unpred before and after conditioning')  


#%% not being used below, could use for supplementary
#%%
import matplotlib.cm as cm
import matplotlib as mpl
rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
}
plt.rcParams.update(rc)

mpl.rcParams['lines.linewidth'] = 4

cond_days = [0,1,2,3,4]
deg_days = [5,6,7,8,9]
rec_days = [10,11,12,13]
ext_days = [14,15,16]
finalrec_days = [17,18]

sample_points = 45
water_point = int(sample_points/9*5.5)
odor_on_point = int(sample_points/9*2)
odor_off_point = int(sample_points/9*3)

gap = 15

x = np.linspace(0,1,10)
knots = sb.get_knots_bsplines(x,df = 7)#spacing = 'equal')

basis = sb._eval_bspline_basis(x,degree = 3,knots=knots)[0][:,1]

# mice = ['FgDA_01','FrgD1_01','FrgD2_01','FrgD2_02']
mice = ['FgDA_02','FgDA_03','FgDA_04','FgDA_05','FgDA_01']
trialtypes = ['go', 'no_go', 'go_omit', 'UnpredReward','background']

for mouse_name in mice:
    for TT in trialtypes:
        if TT == 'go':
            water_point = int(sample_points/9*5.5)
            odor_on_point = int(sample_points/9*2)
            odor_off_point = int(sample_points/9*3)
        elif TT in ['go_omit','no_go']:
            odor_on_point = int(sample_points/9*2)
            odor_off_point = int(sample_points/9*3)
            water_point = np.nan
        elif TT == 'UnpredReward':
            water_point = int(sample_points/9*5.5)
            odor_on_point = np.nan
            odor_off_point = np.nan
        else:
            water_point = np.nan
            odor_on_point = np.nan
            odor_off_point = np.nan
                    
        
        fig,ax = plt.subplots(1,5,sharex = True, sharey = True,figsize = (12,7))
        
        plt.setp(ax, xticks=np.arange(0,sample_points,int(sample_points/9)), xticklabels=np.arange(0,9,1),
                yticks=np.arange(0,15,10))
        
        
        across_day_licking = np.nanmean(licking_data[mouse_name][TT],axis = 0)
        
        if TT =='go':
            across_day_licking_omit = np.nanmean(licking_data[mouse_name]['go_omit'],axis = 0)
        else:
            across_day_licking_omit = np.nanmean(licking_data[mouse_name][TT],axis = 0)
            
        for index,i in enumerate(cond_days):
            licking_conv = np.convolve(basis,across_day_licking[:,i],mode = 'full')[0:sample_points]
            ax[0].plot(licking_conv+gap*index,color =cm.spring(index/float(5)),label = all_days[i])
            ax[0].legend(frameon=False)
            ax[0].set_ylim(-2,gap*5+20)
            ax[0].set_title('conditioning',fontsize = 20)
            ax[0].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[0].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[0].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))
            ax[0].set_ylabel('licking rate (/s)',fontsize = 35)
            
        for index,i in enumerate(deg_days):
            licking_conv = np.convolve(basis,across_day_licking[:,i],mode = 'full')[0:sample_points]
            ax[1].plot(licking_conv+gap*index,color =cm.spring(index/float(5)),label = all_days[i])
            ax[1].legend(frameon=False)
            ax[1].set_title('degradation',fontsize = 20)
            ax[1].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[1].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[1].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))
        for index,i in enumerate(rec_days):
            licking_conv = np.convolve(basis,across_day_licking[:,i],mode = 'full')[0:sample_points]
            ax[2].plot(licking_conv+gap*index,color =cm.spring(index/float(5)),label = all_days[i])
            ax[2].legend(frameon=False)
            ax[2].set_title('recovery',fontsize = 20)
            ax[2].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[2].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[2].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))
        for index,i in enumerate(ext_days):
            licking_conv = np.convolve(basis,across_day_licking_omit[:,i],mode = 'full')[0:sample_points]
            ax[3].plot(licking_conv+gap*index,color =cm.spring(index/float(5)),label = all_days[i])
            ax[3].legend(frameon=False)
            ax[3].set_title('extinction',fontsize = 20)
            ax[3].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[3].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[3].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))
        for index,i in enumerate(finalrec_days):
            licking_conv = np.convolve(basis,across_day_licking[:,i],mode = 'full')[0:sample_points]
            ax[4].plot(licking_conv+gap*index,color =cm.spring(index/float(5)),label = all_days[i])
            ax[4].legend(frameon=False)
            ax[4].set_title('final recovery',fontsize = 20)
            ax[4].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[4].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[4].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))
    
        
        plt.suptitle('trial type{}'.format(TT))
        savepath = 'D:/PhD/Photometry/results/plots/lickings'
        # plt.savefig("{0}/{1}_{2}_licking_{3}.png".format(savepath,mouse_name,TT,date.today()), bbox_inches="tight", dpi = 72)
        plt.show()

#%% averaged licking plot



#%% plot averaged licking traces
def plot_averaged_licking_traces_across_days(licking_data,group_name):
    import matplotlib.cm as cm
    import matplotlib as mpl
    rc = {"axes.spines.left" : False,
          "axes.spines.right" : False,
          "axes.spines.bottom" : False,
          "axes.spines.top" : False,
    }
    plt.rcParams.update(rc)
    
    mpl.rcParams['lines.linewidth'] = 4
    
    cond_days = [0,1,2,3,4]
    deg_days = [5,6,7,8,9]
    
    
    sample_points = 45
    water_point = int(sample_points/9*5.5)
    odor_on_point = int(sample_points/9*2)
    odor_off_point = int(sample_points/9*3)
    
    gap = 15
    
    x = np.linspace(0,1,10)
    knots = sb.get_knots_bsplines(x,df = 7)#spacing = 'equal')
    
    basis = sb._eval_bspline_basis(x,degree = 3,knots=knots)[0][:,1]
    
    trialtypes = ['go', 'no_go', 'go_omit', 'unpred_water','background']
    sessions = ['session {}'.format(x+1) for x in range(10)]
    
    
    for TT in trialtypes:
        if TT == 'go':
            water_point = int(sample_points/9*5.5)
            odor_on_point = int(sample_points/9*2)
            odor_off_point = int(sample_points/9*3)
        elif TT in ['go_omit','no_go']:
            odor_on_point = int(sample_points/9*2)
            odor_off_point = int(sample_points/9*3)
            water_point = np.nan
        elif TT == 'unpred_water':
            water_point = int(sample_points/9*5.5)
            odor_on_point = np.nan
            odor_off_point = np.nan
        else:
            water_point = np.nan
            odor_on_point = np.nan
            odor_off_point = np.nan
                    
        
        fig,ax = plt.subplots(1,2,sharex = True, sharey = True,figsize = (5,10))
        
        plt.setp(ax, xticks=np.arange(0,sample_points,int(sample_points/9)), xticklabels=np.arange(0,9,1),
                yticks=np.arange(0,15,10))
        
        
        across_day_licking = np.nanmean(licking_data[TT],axis = 0)
        

                
        for index,i in enumerate(cond_days):
            licking_conv = np.convolve(basis,across_day_licking[:,i],mode = 'full')[0:sample_points]
            ax[0].plot(licking_conv+gap*index,color =cm.spring(index/float(5)),label = sessions[i])
            ax[0].legend(frameon=False)
            ax[0].set_ylim(-2,gap*5+20)
            ax[0].set_title('conditioning',fontsize = 20)
            ax[0].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[0].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[0].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))
            ax[0].set_ylabel('licking rate (/s)',fontsize = 35)
            
        for index,i in enumerate(deg_days):
            licking_conv = np.convolve(basis,across_day_licking[:,i],mode = 'full')[0:sample_points]
            ax[1].plot(licking_conv+gap*index,color =cm.spring(index/float(5)),label = sessions[i])
            ax[1].legend(frameon=False)
            ax[1].set_title('degradation',fontsize = 20)
            ax[1].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[1].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
            ax[1].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))

        plt.suptitle('trialtype: {} group: {}'.format(TT, group_name))
        # savepath = 'D:/PhD/Photometry/results/plots/lickings'
        # plt.savefig("{0}/{1}_mean_licking_{2}.png".format(savepath,TT,date.today()), bbox_inches="tight", dpi = 72)
        plt.show()
#%%

plot_averaged_licking_traces_across_days(deg_mice_lick_dict, 'deg_group') 
plot_averaged_licking_traces_across_days(control_mice_lick_dict, 'controal_group') 
plot_averaged_licking_traces_across_days(c_mice_lick_dict, 'c_group') 

    
#%% plot single date average licking trace

fig = plt.figure()


session_of_phase = 4

length = 45
trialtypes_full = ['go','no_go','unpred_water','go_omit','background']

def plot_single_session_average_licking(data, color,legends,CS,US,std_on = False,figsize = (5,4),ylim=[-1,7],savename = 'temp'):
    
    fig,ax = plt.subplots(figsize = (figsize[0],figsize[1]))
    
    plt.xticks(np.arange(0,45,5),np.arange(-2,7,1))
    plt.xlabel('Time from odor onset(s)')
    plt.ylabel('licking rate (/s)')
    ax.set_ylim([ylim[0],ylim[1]])
    
    for item,c,l in zip(data,color,legends):
        # filled area
        num = item.shape[0]
        aa = np.nanmean(item,axis = 0)
        ax.plot(aa,color = c,label = l,lw = 1.7)
        if std_on:
            std = np.nanstd(item,axis = 0)
            ax.fill_between(np.arange(0,45,1), aa-std/num, aa+std/num,alpha = 0.1,color = 'grey') ## shall I devide this by num?
        ymin, ymax = ax.get_ylim()
    if CS:       
        # vertical lines
        ax.vlines(x=10, ymin=ymin, ymax=ymax, colors='tab:grey', ls='--', lw=2)
        ax.vlines(x=15, ymin=ymin, ymax=ymax, colors='tab:grey', ls='--', lw=2)
    if US:     
        # vertical lines
        ax.vlines(x=27.5, ymin=ymin, ymax=ymax, colors='tab:grey', ls='--', lw=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.legend(frameon=False,fancybox=False,bbox_to_anchor=(1, 0.8),)
    savepath = 'D:/PhD/Figures/lickings'
    plt.savefig("{0}/{1}_mean_licking_by_group_{2}.png".format(savepath,savename,date.today()), bbox_inches="tight", dpi = 300)
    plt.savefig("{0}/{1}_mean_licking_by_group_{2}.eps".format(savepath,savename,date.today()), bbox_inches="tight", dpi = 300)
    
    plt.show()
    
ymax = 10
# plot_single_session_average_licking(data = [deg_mice_lick_dict['go'],deg_mice_lick_dict['no_go'],deg_mice_lick_dict['go_omit']],
#                                     session_id = 4,color = ['#ffcb0d','#2e2e2d','#ffe28a'],
#                                   CS = True,US = True,
#                                   figsize = (5,4),ylim=[-1,ymax])  

plot_single_session_average_licking(data = [deg_mice_lick_dict['go'][:,:,4],deg_mice_lick_dict['go'][:,:,9],
                                            control_mice_lick_dict['go'][:,:,4],control_mice_lick_dict['go'][:,:,9],
                                            c_mice_lick_dict['go'][:,:,4],c_mice_lick_dict['go'][:,:,8],],
                                    color = ['#ff9203','#fcb85d','#54e3a3','#a1e3c5','#7786f7','#b1b9f0'],
                                    legends = ['deg group after conditioning','deg group after degradation',
                                               'cond group after conditioning','cond group after degradation',
                                               'C odor group after conditioning','C odor group after degradation'],
                                  CS = True,US = True,
                                  figsize = (7,4),ylim=[-1,ymax],
                                  std_on = False,
                                  savename = 'deg cond c before and after degradation')  

plot_single_session_average_licking(data = [deg_mice_lick_dict['go'][:,:,4],deg_mice_lick_dict['go'][:,:,9],
                                            control_mice_lick_dict['go'][:,:,4],control_mice_lick_dict['go'][:,:,9],
                                    deg_mice_lick_dict['no_go'][:,:,4],deg_mice_lick_dict['no_go'][:,:,9],
                                            control_mice_lick_dict['no_go'][:,:,4],control_mice_lick_dict['no_go'][:,:,9]],
                                    color = ['#ff9203','#fcb85d','#54e3a3','#a1e3c5','grey','grey','grey','grey'],
                                    legends = ['deg go after conditioning','deg go after degradation',
                                               'cond go after conditioning','cond go after degradation',
                                               'no go','','',''],
                                  CS = True,US = True,
                                  figsize = (7,4),ylim=[-1,ymax],
                                  std_on = False,
                                  savename = 'deg cond go and nogo before and after degradation')  

plot_single_session_average_licking(data = [control_mice_lick_dict['go'][:,:,4],control_mice_lick_dict['go_omit'][:,:,4],
                                            control_mice_lick_dict['no_go'][:,:,4]],
                                    color = ['#54e3a3','#54d0e3','grey',],
                                    legends = ['cond go after conditioning','cond go omission','cond no go'],
                                  CS = True,US = True,
                                  figsize = (7,4),ylim=[-1,ymax],
                                  std_on = True,
                                  savename = 'cond go go omit and nogo after conditioning')  

plot_single_session_average_licking(data = [deg_mice_lick_dict['go'][:,:,4],deg_mice_lick_dict['go'][:,:,9],
                                            deg_mice_lick_dict['no_go'][:,:,4],deg_mice_lick_dict['unpred_water'][:,:,9]],
                                    color = ['#ff9203','#fcb85d','grey','#c074f2'],
                                    legends = ['deg go after conditioning','deg go after degradation',
                                               'deg no go','deg unpredicted water'],
                                  CS = True,US = True,
                                  figsize = (7,4),ylim=[-1,ymax],
                                  std_on = False,
                                  savename = 'def go and nogo unpred before and after conditioning')  