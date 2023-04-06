# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2023. Simon J. Guillot. All rights reserved.                            +
#  Redistribution and use in source and binary forms, with or without modification, are strictly prohibited.
#                                                                                        +
#  THIS CODE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
#  BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#  IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
#  OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
#  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS CODE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import mne
import glob
import math
import yasa
import time
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import statistics as st
import PySimpleGUI as sg
import scipy.stats as stats
import matplotlib.pyplot as plt
from fooof import FOOOF
from scipy import signal
from tensorpac import Pac
from scipy.fft import fft
from scipy.stats import sem
from scipy.stats import zscore
from mne.preprocessing import ICA
from scipy.integrate import simpson
from tkinter import filedialog as fd
from coffeine import compute_features
from scipy.signal import welch,hilbert
from sklearn.metrics import accuracy_score
from mne_features.utils import power_spectrum
from matplotlib.collections import LineCollection
from mne.time_frequency import psd_array_multitaper
from mne_features.univariate import compute_spect_slope
from matplotlib.colors import ListedColormap,BoundaryNorm
from fooof.plts.spectra import plot_spectrum,plot_spectra


#region SOD-WT
#region Dir path SOD-wt
path=fd.askdirectory(title='SELECT WHERE TO SAVE PLOTS')
inpath=fd.askdirectory(title='SELECT DATA DIRECTORY')
dir_edf=sorted(glob.glob(os.path.join(inpath,'*.edf')))
dname=[]
EEG_wt=pd.DataFrame()
#endregion
for file in dir_edf:
	fname=os.path.basename(file).split('.')[0]
	dname.append(fname)
	raw=mne.io.read_raw_edf(file,preload=True,verbose=False)
	print('Data loaded.')
	print(raw.info)
	raw
	print(raw.ch_names)
	#region Drop unused channels
	if "Channel 2" in raw.ch_names:
		raw.drop_channels(['Channel 2'])
	if "Channel 3" in raw.ch_names:
		raw.drop_channels(['Channel 3'])
	if "Channel 4" in raw.ch_names:
		raw.drop_channels(['Channel 4'])
	#endregion
	chan=raw.ch_names
	print(chan)
	raw.compute_psd(n_fft=500).plot()
	print(raw.info['sfreq'])
	sf=raw.info['sfreq']
	print(sf)
	raw_notch=raw.copy()
	raw_notch=raw_notch.notch_filter(sf,filter_length='auto',method='spectrum_fit',fir_window='hamm')
	raw_notch.compute_psd().plot()
	raw_notch.compute_psd(fmax=55,n_overlap=5,n_fft=1000).plot()
	mne_sigs=raw_notch.get_data()
	print('Data in mV:',np.sum(np.abs(mne_sigs))/mne_sigs.size)
	data=raw_notch.get_data(units="uV")
	print(data.shape)
	data=data.reshape(-1)
	print(data.shape)
	data=pd.DataFrame(data.T)
	EEG_wt=pd.concat([EEG_wt,data],ignore_index=True,axis=1)
else:
	#region Data export
	EEG_wt=EEG_wt.reset_index(drop=True)
	EEG_wt.index=range(1,EEG_wt.shape[0]+1)
	EEG_wt.columns=['wt43','wt63','wt66','wt70','wt71']

	#region wt43
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_wt=len(EEG_wt.index)
	EEG_wt_baseline=EEG_wt.drop(range(baseline_stop,maxlen_wt))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(6*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_wt_PTZ=EEG_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_wt+1)))
	EEG_wt_baseline=EEG_wt_baseline.reset_index()
	EEG_wt_PTZ=EEG_wt_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_wt43=EEG_wt_baseline['wt43'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_wt_PTZ['wt43']>mean_wt43)
	EEG_comp["wt43"]=EEG_comp["wt43"].astype(int)
	#Counting the values strictly above
	count_wt43=EEG_comp['wt43'].sum()
	EEG_counts_wt43=EEG_comp.loc[EEG_comp['wt43']==True]
	#Percentage of event above mean of baseline
	perc_wt43=(count_wt43/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['wt43'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region wt63
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_wt=len(EEG_wt.index)
	EEG_wt_baseline=EEG_wt.drop(range(baseline_stop,maxlen_wt))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.51*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_wt_PTZ=EEG_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_wt+1)))
	EEG_wt_baseline=EEG_wt_baseline.reset_index()
	EEG_wt_PTZ=EEG_wt_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_wt63=EEG_wt_baseline['wt63'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_wt_PTZ['wt63']>mean_wt63)
	EEG_comp["wt63"]=EEG_comp["wt63"].astype(int)
	#Counting the values strictly above
	count_wt63=EEG_comp['wt63'].sum()
	EEG_counts_wt63=EEG_comp.loc[EEG_comp['wt63']==True]
	#Percentage of event above mean of baseline
	perc_wt63=(count_wt63/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['wt63'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region wt66
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_wt=len(EEG_wt.index)
	EEG_wt_baseline=EEG_wt.drop(range(baseline_stop,maxlen_wt))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.48*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_wt_PTZ=EEG_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_wt+1)))
	EEG_wt_baseline=EEG_wt_baseline.reset_index()
	EEG_wt_PTZ=EEG_wt_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_wt66=EEG_wt_baseline['wt66'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_wt_PTZ['wt66']>mean_wt66)
	EEG_comp["wt66"]=EEG_comp["wt66"].astype(int)
	#Counting the values strictly above
	count_wt66=EEG_comp['wt66'].sum()
	EEG_counts_wt66=EEG_comp.loc[EEG_comp['wt66']==True]
	#Percentage of event above mean of baseline
	perc_wt66=(count_wt66/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['wt66'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region wt70
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_wt=len(EEG_wt.index)
	EEG_wt_baseline=EEG_wt.drop(range(baseline_stop,maxlen_wt))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(6.04*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_wt_PTZ=EEG_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_wt+1)))
	EEG_wt_baseline=EEG_wt_baseline.reset_index()
	EEG_wt_PTZ=EEG_wt_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_wt70=EEG_wt_baseline['wt70'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_wt_PTZ['wt70']>mean_wt70)
	EEG_comp["wt70"]=EEG_comp["wt70"].astype(int)
	#Counting the values strictly above
	count_wt70=EEG_comp['wt70'].sum()
	EEG_counts_wt70=EEG_comp.loc[EEG_comp['wt70']==True]
	#Percentage of event above mean of baseline
	perc_wt70=(count_wt70/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['wt70'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region wt71
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_wt=len(EEG_wt.index)
	EEG_wt_baseline=EEG_wt.drop(range(baseline_stop,maxlen_wt))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.56*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_wt_PTZ=EEG_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_wt+1)))
	EEG_wt_baseline=EEG_wt_baseline.reset_index()
	EEG_wt_PTZ=EEG_wt_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_wt71=EEG_wt_baseline['wt71'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_wt_PTZ['wt71']>mean_wt71)
	EEG_comp["wt71"]=EEG_comp["wt71"].astype(int)
	#Counting the values strictly above
	count_wt71=EEG_comp['wt71'].sum()
	EEG_counts_wt71=EEG_comp.loc[EEG_comp['wt71']==True]
	#Percentage of event above mean of baseline
	perc_wt71=(count_wt71/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['wt71'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)   #endregion
#endregion
#region SOD
#region Dir path SOD
inpath=fd.askdirectory(title='SELECT DATA DIRECTORY')
dir_edf=sorted(glob.glob(os.path.join(inpath,'*.edf')))
dname=[]
EEG_sod=pd.DataFrame()
#endregion
for file in dir_edf:
	fname=os.path.basename(file).split('.')[0]
	dname.append(fname)
	raw=mne.io.read_raw_edf(file,preload=True,verbose=False)
	print('Data loaded.')
	print(raw.info)
	raw
	print(raw.ch_names)
	#region Drop unused channels
	if "Channel 2" in raw.ch_names:
		raw.drop_channels(['Channel 2'])
	if "Channel 3" in raw.ch_names:
		raw.drop_channels(['Channel 3'])
	if "Channel 4" in raw.ch_names:
		raw.drop_channels(['Channel 4'])
	#endregion
	chan=raw.ch_names
	print(chan)
	raw.compute_psd(n_fft=500).plot()
	print(raw.info['sfreq'])
	sf=raw.info['sfreq']
	print(sf)
	raw_notch=raw.copy()
	raw_notch=raw_notch.notch_filter(sf,filter_length='auto',method='spectrum_fit',fir_window='hamm')
	raw_notch.compute_psd().plot()
	raw_notch.compute_psd(fmax=55,n_overlap=5,n_fft=1000).plot()
	mne_sigs=raw_notch.get_data()
	print('Data in mV:',np.sum(np.abs(mne_sigs))/mne_sigs.size)
	data=raw_notch.get_data(units="uV")
	print(data.shape)
	data=data.reshape(-1)
	print(data.shape)
	data=pd.DataFrame(data.T)
	EEG_sod=pd.concat([EEG_sod,data],ignore_index=True,axis=1)
else:
	#region Data export
	EEG_sod=EEG_sod.reset_index(drop=True)
	EEG_sod.index=range(1,EEG_sod.shape[0]+1)
	EEG_sod.columns=['sod45','sod61','sod64','sod65','sod69']

	#region SOD45
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_sod=len(EEG_sod.index)
	EEG_sod_baseline=EEG_sod.drop(range(baseline_stop,maxlen_sod))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(6*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_sod_PTZ=EEG_sod.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod+1)))
	EEG_sod_baseline=EEG_sod_baseline.reset_index()
	EEG_sod_PTZ=EEG_sod_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_sod45=EEG_sod_baseline['sod45'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_sod_PTZ['sod45']>mean_sod45)
	EEG_comp["sod45"]=EEG_comp["sod45"].astype(int)
	#Counting the values strictly above
	count_sod45=EEG_comp['sod45'].sum()
	EEG_counts_sod45=EEG_comp.loc[EEG_comp['sod45']==True]
	#Percentage of event above mean of baseline
	perc_sod45=(count_sod45/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['sod45'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region SOD61
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_sod=len(EEG_sod.index)
	EEG_sod_baseline=EEG_sod.drop(range(baseline_stop,maxlen_sod))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.51*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_sod_PTZ=EEG_sod.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod+1)))
	EEG_sod_baseline=EEG_sod_baseline.reset_index()
	EEG_sod_PTZ=EEG_sod_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_sod61=EEG_sod_baseline['sod61'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_sod_PTZ['sod61']>mean_sod61)
	EEG_comp["sod61"]=EEG_comp["sod61"].astype(int)
	#Counting the values strictly above
	count_sod61=EEG_comp['sod61'].sum()
	EEG_counts_sod61=EEG_comp.loc[EEG_comp['sod61']==True]
	#Percentage of event above mean of baseline
	perc_sod61=(count_sod61/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['sod61'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region SOD64
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_sod=len(EEG_sod.index)
	EEG_sod_baseline=EEG_sod.drop(range(baseline_stop,maxlen_sod))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.48*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_sod_PTZ=EEG_sod.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod+1)))
	EEG_sod_baseline=EEG_sod_baseline.reset_index()
	EEG_sod_PTZ=EEG_sod_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_sod64=EEG_sod_baseline['sod64'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_sod_PTZ['sod64']>mean_sod64)
	EEG_comp["sod64"]=EEG_comp["sod64"].astype(int)
	#Counting the values strictly above
	count_sod64=EEG_comp['sod64'].sum()
	EEG_counts_sod64=EEG_comp.loc[EEG_comp['sod64']==True]
	#Percentage of event above mean of baseline
	perc_sod64=(count_sod64/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['sod64'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region SOD65
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_sod=len(EEG_sod.index)
	EEG_sod_baseline=EEG_sod.drop(range(baseline_stop,maxlen_sod))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(6.04*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_sod_PTZ=EEG_sod.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod+1)))
	EEG_sod_baseline=EEG_sod_baseline.reset_index()
	EEG_sod_PTZ=EEG_sod_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_sod65=EEG_sod_baseline['sod65'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_sod_PTZ['sod65']>mean_sod65)
	EEG_comp["sod65"]=EEG_comp["sod65"].astype(int)
	#Counting the values strictly above
	count_sod65=EEG_comp['sod65'].sum()
	EEG_counts_sod65=EEG_comp.loc[EEG_comp['sod65']==True]
	#Percentage of event above mean of baseline
	perc_sod65=(count_sod65/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['sod65'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region SOD69
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_sod=len(EEG_sod.index)
	EEG_sod_baseline=EEG_sod.drop(range(baseline_stop,maxlen_sod))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.56*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_sod_PTZ=EEG_sod.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod+1)))
	EEG_sod_baseline=EEG_sod_baseline.reset_index()
	EEG_sod_PTZ=EEG_sod_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_sod69=EEG_sod_baseline['sod69'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_sod_PTZ['sod69']>mean_sod69)
	EEG_comp["sod69"]=EEG_comp["sod69"].astype(int)
	#Counting the values strictly above
	count_sod69=EEG_comp['sod69'].sum()
	EEG_counts_sod69=EEG_comp.loc[EEG_comp['sod69']==True]
	#Percentage of event above mean of baseline
	perc_sod69=(count_sod69/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['sod69'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)   #endregion
#endregion
#region FUS-WT
#region Dir path FUS-WT
inpath=fd.askdirectory(title='SELECT DATA DIRECTORY')
dir_edf=sorted(glob.glob(os.path.join(inpath,'*.edf')))
dname=[]
EEG_fus_wt=pd.DataFrame()
#endregion
for file in dir_edf:
	fname=os.path.basename(file).split('.')[0]
	dname.append(fname)
	raw=mne.io.read_raw_edf(file,preload=True,verbose=False)
	print('Data loaded.')
	print(raw.info)
	raw
	print(raw.ch_names)
	#region Drop unused channels
	if "Channel 2" in raw.ch_names:
		raw.drop_channels(['Channel 2'])
	if "Channel 3" in raw.ch_names:
		raw.drop_channels(['Channel 3'])
	if "Channel 4" in raw.ch_names:
		raw.drop_channels(['Channel 4'])
	#endregion
	chan=raw.ch_names
	print(chan)
	raw.compute_psd(n_fft=500).plot()
	print(raw.info['sfreq'])
	sf=raw.info['sfreq']
	print(sf)
	raw_notch=raw.copy()
	raw_notch=raw_notch.notch_filter(sf,filter_length='auto',method='spectrum_fit',fir_window='hamm')
	raw_notch.compute_psd().plot()
	raw_notch.compute_psd(fmax=55,n_overlap=5,n_fft=1000).plot()
	mne_sigs=raw_notch.get_data()
	print('Data in mV:',np.sum(np.abs(mne_sigs))/mne_sigs.size)
	data=raw_notch.get_data(units="uV")
	print(data.shape)
	data=data.reshape(-1)
	print(data.shape)
	data=pd.DataFrame(data.T)
	EEG_fus_wt=pd.concat([EEG_fus_wt,data],ignore_index=True,axis=1)
else:
	#region Data export
	EEG_fus_wt=EEG_fus_wt.reset_index(drop=True)
	EEG_fus_wt.index=range(1,EEG_fus_wt.shape[0]+1)
	EEG_fus_wt.columns=['fus_wt358','fus_wt360','fus_wt361','fus_wt363']

	#region FUS_wt358
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_fus_wt=len(EEG_fus_wt.index)
	EEG_fus_wt_baseline=EEG_fus_wt.drop(range(baseline_stop,maxlen_fus_wt))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.56*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_fus_wt_PTZ=EEG_fus_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus_wt+1)))
	EEG_fus_wt_baseline=EEG_fus_wt_baseline.reset_index()
	EEG_fus_wt_PTZ=EEG_fus_wt_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_fus_wt358=EEG_fus_wt_baseline['fus_wt358'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_fus_wt_PTZ['fus_wt358']>mean_fus_wt358)
	EEG_comp["fus_wt358"]=EEG_comp["fus_wt358"].astype(int)
	#Counting the values strictly above
	count_fus_wt358=EEG_comp['fus_wt358'].sum()
	EEG_counts_fus_wt358=EEG_comp.loc[EEG_comp['fus_wt358']==True]
	#Percentage of event above mean of baseline
	perc_fus_wt358=(count_fus_wt358/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['fus_wt358'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region FUS_wt360
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_fus_wt=len(EEG_fus_wt.index)
	EEG_fus_wt_baseline=EEG_fus_wt.drop(range(baseline_stop,maxlen_fus_wt))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.56*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_fus_wt_PTZ=EEG_fus_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus_wt+1)))
	EEG_fus_wt_baseline=EEG_fus_wt_baseline.reset_index()
	EEG_fus_wt_PTZ=EEG_fus_wt_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_fus_wt360=EEG_fus_wt_baseline['fus_wt360'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_fus_wt_PTZ['fus_wt360']>mean_fus_wt360)
	EEG_comp["fus_wt360"]=EEG_comp["fus_wt360"].astype(int)
	#Counting the values strictly above
	count_fus_wt360=EEG_comp['fus_wt360'].sum()
	EEG_counts_fus_wt360=EEG_comp.loc[EEG_comp['fus_wt360']==True]
	#Percentage of event above mean of baseline
	perc_fus_wt360=(count_fus_wt360/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['fus_wt360'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region FUS_wt361
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_fus_wt=len(EEG_fus_wt.index)
	EEG_fus_wt_baseline=EEG_fus_wt.drop(range(baseline_stop,maxlen_fus_wt))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.56*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_fus_wt_PTZ=EEG_fus_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus_wt+1)))
	EEG_fus_wt_baseline=EEG_fus_wt_baseline.reset_index()
	EEG_fus_wt_PTZ=EEG_fus_wt_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_fus_wt361=EEG_fus_wt_baseline['fus_wt361'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_fus_wt_PTZ['fus_wt361']>mean_fus_wt361)
	EEG_comp["fus_wt361"]=EEG_comp["fus_wt361"].astype(int)
	#Counting the values strictly above
	count_fus_wt361=EEG_comp['fus_wt361'].sum()
	EEG_counts_fus_wt361=EEG_comp.loc[EEG_comp['fus_wt361']==True]
	#Percentage of event above mean of baseline
	perc_fus_wt361=(count_fus_wt361/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['fus_wt361'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region FUS_wt363
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_fus_wt=len(EEG_fus_wt.index)
	EEG_fus_wt_baseline=EEG_fus_wt.drop(range(baseline_stop,maxlen_fus_wt))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.56*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_fus_wt_PTZ=EEG_fus_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus_wt+1)))
	EEG_fus_wt_baseline=EEG_fus_wt_baseline.reset_index()
	EEG_fus_wt_PTZ=EEG_fus_wt_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_fus_wt363=EEG_fus_wt_baseline['fus_wt363'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_fus_wt_PTZ['fus_wt363']>mean_fus_wt363)
	EEG_comp["fus_wt363"]=EEG_comp["fus_wt363"].astype(int)
	#Counting the values strictly above
	count_fus_wt363=EEG_comp['fus_wt363'].sum()
	EEG_counts_fus_wt363=EEG_comp.loc[EEG_comp['fus_wt363']==True]
	#Percentage of event above mean of baseline
	perc_fus_wt363=(count_fus_wt363/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['fus_wt363'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)  #endregion
#endregion
#region FUS
#region Dir path FUS
inpath=fd.askdirectory(title='SELECT DATA DIRECTORY')
dir_edf=sorted(glob.glob(os.path.join(inpath,'*.edf')))
dname=[]
EEG_fus=pd.DataFrame()
#endregion
for file in dir_edf:
	fname=os.path.basename(file).split('.')[0]
	dname.append(fname)
	raw=mne.io.read_raw_edf(file,preload=True,verbose=False)
	print('Data loaded.')
	print(raw.info)
	raw
	print(raw.ch_names)
	#region Drop unused channels
	if "Channel 2" in raw.ch_names:
		raw.drop_channels(['Channel 2'])
	if "Channel 3" in raw.ch_names:
		raw.drop_channels(['Channel 3'])
	if "Channel 4" in raw.ch_names:
		raw.drop_channels(['Channel 4'])
	#endregion
	chan=raw.ch_names
	print(chan)
	raw.compute_psd(n_fft=500).plot()
	print(raw.info['sfreq'])
	sf=raw.info['sfreq']
	print(sf)
	raw_notch=raw.copy()
	raw_notch=raw_notch.notch_filter(sf,filter_length='auto',method='spectrum_fit',fir_window='hamm')
	raw_notch.compute_psd().plot()
	raw_notch.compute_psd(fmax=55,n_overlap=5,n_fft=1000).plot()
	mne_sigs=raw_notch.get_data()
	print('Data in mV:',np.sum(np.abs(mne_sigs))/mne_sigs.size)
	data=raw_notch.get_data(units="uV")
	print(data.shape)
	data=data.reshape(-1)
	print(data.shape)
	data=pd.DataFrame(data.T)
	EEG_fus=pd.concat([EEG_fus,data],ignore_index=True,axis=1)
else:
	#region Data export
	EEG_fus=EEG_fus.reset_index(drop=True)
	EEG_fus.index=range(1,EEG_fus.shape[0]+1)
	EEG_fus.columns=['fus362','fus364','fus365','fus372','fus373']

	#region FUS362
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_fus=len(EEG_fus.index)
	EEG_fus_baseline=EEG_fus.drop(range(baseline_stop,maxlen_fus))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(6*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_fus_PTZ=EEG_fus.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus+1)))
	EEG_fus_baseline=EEG_fus_baseline.reset_index()
	EEG_fus_PTZ=EEG_fus_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_fus362=EEG_fus_baseline['fus362'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_fus_PTZ['fus362']>mean_fus362)
	EEG_comp["fus362"]=EEG_comp["fus362"].astype(int)
	#Counting the values strictly above
	count_fus362=EEG_comp['fus362'].sum()
	EEG_counts_fus362=EEG_comp.loc[EEG_comp['fus362']==True]
	#Percentage of event above mean of baseline
	perc_fus362=(count_fus362/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['FUS362'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region FUS364
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_fus=len(EEG_fus.index)
	EEG_fus_baseline=EEG_fus.drop(range(baseline_stop,maxlen_fus))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.51*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_fus_PTZ=EEG_fus.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus+1)))
	EEG_fus_baseline=EEG_fus_baseline.reset_index()
	EEG_fus_PTZ=EEG_fus_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_fus364=EEG_fus_baseline['fus364'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_fus_PTZ['fus364']>mean_fus364)
	EEG_comp["fus364"]=EEG_comp["fus364"].astype(int)
	#Counting the values strictly above
	count_fus364=EEG_comp['fus364'].sum()
	EEG_counts_fus364=EEG_comp.loc[EEG_comp['fus364']==True]
	#Percentage of event above mean of baseline
	perc_fus364=(count_fus364/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['FUS364'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region FUS365
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_fus=len(EEG_fus.index)
	EEG_fus_baseline=EEG_fus.drop(range(baseline_stop,maxlen_fus))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.48*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_fus_PTZ=EEG_fus.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus+1)))
	EEG_fus_baseline=EEG_fus_baseline.reset_index()
	EEG_fus_PTZ=EEG_fus_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_fus365=EEG_fus_baseline['fus365'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_fus_PTZ['fus365']>mean_fus365)
	EEG_comp["fus365"]=EEG_comp["fus365"].astype(int)
	#Counting the values strictly above
	count_fus365=EEG_comp['fus365'].sum()
	EEG_counts_fus365=EEG_comp.loc[EEG_comp['fus365']==True]
	#Percentage of event above mean of baseline
	perc_fus365=(count_fus365/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['FUS365'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region FUS372
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_fus=len(EEG_fus.index)
	EEG_fus_baseline=EEG_fus.drop(range(baseline_stop,maxlen_fus))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(6.04*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_fus_PTZ=EEG_fus.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus+1)))
	EEG_fus_baseline=EEG_fus_baseline.reset_index()
	EEG_fus_PTZ=EEG_fus_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_fus372=EEG_fus_baseline['fus372'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_fus_PTZ['fus372']>mean_fus372)
	EEG_comp["fus372"]=EEG_comp["fus372"].astype(int)
	#Counting the values strictly above
	count_fus372=EEG_comp['fus372'].sum()
	EEG_counts_fus372=EEG_comp.loc[EEG_comp['fus372']==True]
	#Percentage of event above mean of baseline
	perc_fus372=(count_fus372/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['FUS372'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)
	#endregion
	#region FUS373
	#Baseline window 10mn
	baseline_stop=10*60000
	maxlen_fus=len(EEG_fus.index)
	EEG_fus_baseline=EEG_fus.drop(range(baseline_stop,maxlen_fus))
	#PTZ injection window 10mn
	inject_t0=baseline_stop+(5.56*60000)
	inject_t1=inject_t0+(10*60000)
	EEG_fus_PTZ=EEG_fus.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus+1)))
	EEG_fus_baseline=EEG_fus_baseline.reset_index()
	EEG_fus_PTZ=EEG_fus_PTZ.reset_index()
	#Calculate baseline mean for comparison with PTZ
	mean_fus373=EEG_fus_baseline['fus373'].mean(axis=0)
	EEG_comp=pd.DataFrame(EEG_fus_PTZ['fus373']>mean_fus373)
	EEG_comp["fus373"]=EEG_comp["fus373"].astype(int)
	#Counting the values strictly above
	count_fus373=EEG_comp['fus373'].sum()
	EEG_counts_fus373=EEG_comp.loc[EEG_comp['fus373']==True]
	#Percentage of event above mean of baseline
	perc_fus373=(count_fus373/len(EEG_comp))*100
	#Latency to first event
	latency=EEG_comp['FUS373'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print(latency)  #endregion
pass
#endregion











	#Envelope + plot baseline
	windowsize=20
	env_wt=pd.DataFrame()
	env_wt['wt_upperEnv']=EEG_wt_baseline['avzscore'].rolling(window=windowsize).max().shift(int(-windowsize/2))
	env_wt['wt_lowerEnv']=EEG_wt_baseline['avzscore'].rolling(window=windowsize).min().shift(int(-windowsize/2))
	print("Envelope created.")
	#Plot envelope
	x=range(8031)
	y=EEG_wt_baseline['avzscore']
	df=pd.DataFrame(data={"y":y},index=x)
	extract_col=EEG_wt_baseline['WT43z']
	df=df.join(extract_col)
	windowsize=20
	df['y_upperEnv']=env_wt['wt_upperEnv']
	df['y_lowerEnv']=env_wt['wt_lowerEnv']
	cmap=ListedColormap(['none','red','black'])
	df.plot(cmap=cmap,linewidth=0.75)
	plt.tight_layout()
	plt.show()

	mean_EEG=EEG_wt_baseline['WT43z']
	EEG_time=range(len(EEG_wt_baseline.index))
	Nfine=500
	x=np.linspace(EEG_time[0],EEG_time[-1],len(EEG_time)*Nfine)
	y=np.interp(x,EEG_time,mean_EEG)
	cmap=ListedColormap(['grey','red'])
	norm=BoundaryNorm([0,sd_pos_wt43,np.inf],cmap.N)
	points=np.array([x,y]).T.reshape(-1,1,2)
	segments=np.concatenate([points[:-1],points[1:]],axis=1)
	lc=LineCollection(segments,cmap=cmap,norm=norm)
	lc.set_array(y)
	lc.set_linewidth(0.75)
	plt.gca().add_collection(lc)
	plt.xlim(x.min(),x.max())
	#plt.ylim(y.min(),y.max())
	#plt.ylim(-60,60)
	plt.tight_layout()
	plt.show()


	name00="EEG_SOD-WT_env.png"
	path00=os.path.join(path,name00)
	plt.savefig(path00,dpi=1200,bbox_inches='tight',transparent=True,pad_inches=0)
	plt.show()
	print("Plot done.")



	name1=('EEG_WT.csv')
	path1=os.path.join(path,name1)
	EEG_wt.to_csv(path1)	#endregion
