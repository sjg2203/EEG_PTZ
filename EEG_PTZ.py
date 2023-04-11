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
EEG_sod_wt=pd.DataFrame()
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
	EEG_sod_wt=pd.concat([EEG_sod_wt,data],ignore_index=True,axis=1)
else:
	#region Data export
	EEG_sod_wt=EEG_sod_wt.reset_index(drop=True)
	EEG_sod_wt.index=range(1,EEG_sod_wt.shape[0]+1)
	EEG_sod_wt.columns=['sod_wt43','sod_wt63','sod_wt66','sod_wt70','sod_wt71']

	#region sod_wt43
	baseline_stop=10*60000
	maxlen_sod_wt=len(EEG_sod_wt.index)
	EEG_sod_wt_baseline=EEG_sod_wt.drop(range(baseline_stop,maxlen_sod_wt))
	inject_t0=int(baseline_stop+(6*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_sod_wt_PTZ=EEG_sod_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod_wt+1)))
	EEG_sod_wt_baseline=EEG_sod_wt_baseline.reset_index()
	EEG_sod_wt_PTZ=EEG_sod_wt_PTZ.reset_index()
	z=pd.DataFrame(EEG_sod_wt_baseline['sod_wt43'])
	y=pd.DataFrame(EEG_sod_wt_PTZ['sod_wt43'])
	sod_wt43z=abs(zscore(z,axis=None,nan_policy='omit'))
	sod_wt43zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(sod_wt43zptz>4*sod_wt43z)
	EEG_comp["sod_wt43"]=EEG_comp["sod_wt43"].astype(int)
	count_sod_wt43=EEG_comp['sod_wt43'].sum()
	EEG_counts_sod_wt43=EEG_comp.loc[EEG_comp['sod_wt43']==True]
	latency=EEG_comp['sod_wt43'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region sod_wt63
	baseline_stop=10*60000
	maxlen_sod_wt=len(EEG_sod_wt.index)
	EEG_sod_wt_baseline=EEG_sod_wt.drop(range(baseline_stop,maxlen_sod_wt))
	inject_t0=int(baseline_stop+(5.51*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_sod_wt_PTZ=EEG_sod_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod_wt+1)))
	EEG_sod_wt_baseline=EEG_sod_wt_baseline.reset_index()
	EEG_sod_wt_PTZ=EEG_sod_wt_PTZ.reset_index()
	z=pd.DataFrame(EEG_sod_wt_baseline['sod_wt63'])
	y=pd.DataFrame(EEG_sod_wt_PTZ['sod_wt63'])
	sod_wt63z=abs(zscore(z,axis=None,nan_policy='omit'))
	sod_wt63zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(sod_wt63zptz>4*sod_wt63z)
	EEG_comp["sod_wt63"]=EEG_comp["sod_wt63"].astype(int)
	count_sod_wt63=EEG_comp['sod_wt63'].sum()
	EEG_counts_sod_wt63=EEG_comp.loc[EEG_comp['sod_wt63']==True]
	latency=EEG_comp['sod_wt63'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region sod_wt66
	baseline_stop=10*60000
	maxlen_sod_wt=len(EEG_sod_wt.index)
	EEG_sod_wt_baseline=EEG_sod_wt.drop(range(baseline_stop,maxlen_sod_wt))
	inject_t0=int(baseline_stop+(5.48*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_sod_wt_PTZ=EEG_sod_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod_wt+1)))
	EEG_sod_wt_baseline=EEG_sod_wt_baseline.reset_index()
	EEG_sod_wt_PTZ=EEG_sod_wt_PTZ.reset_index()
	z=pd.DataFrame(EEG_sod_wt_baseline['sod_wt66'])
	y=pd.DataFrame(EEG_sod_wt_PTZ['sod_wt66'])
	sod_wt66z=abs(zscore(z,axis=None,nan_policy='omit'))
	sod_wt66zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(sod_wt66zptz>4*sod_wt66z)
	EEG_comp["sod_wt66"]=EEG_comp["sod_wt66"].astype(int)
	count_sod_wt66=EEG_comp['sod_wt66'].sum()
	EEG_counts_sod_wt66=EEG_comp.loc[EEG_comp['sod_wt66']==True]
	latency=EEG_comp['sod_wt66'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region sod_wt70
	baseline_stop=10*60000
	maxlen_sod_wt=len(EEG_sod_wt.index)
	EEG_sod_wt_baseline=EEG_sod_wt.drop(range(baseline_stop,maxlen_sod_wt))
	inject_t0=int(baseline_stop+(6.04*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_sod_wt_PTZ=EEG_sod_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod_wt+1)))
	EEG_sod_wt_baseline=EEG_sod_wt_baseline.reset_index()
	EEG_sod_wt_PTZ=EEG_sod_wt_PTZ.reset_index()
	z=pd.DataFrame(EEG_sod_wt_baseline['sod_wt70'])
	y=pd.DataFrame(EEG_sod_wt_PTZ['sod_wt70'])
	sod_wt70z=abs(zscore(z,axis=None,nan_policy='omit'))
	sod_wt70zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(sod_wt70zptz>4*sod_wt70z)
	EEG_comp["sod_wt70"]=EEG_comp["sod_wt70"].astype(int)
	count_sod_wt70=EEG_comp['sod_wt70'].sum()
	EEG_counts_sod_wt70=EEG_comp.loc[EEG_comp['sod_wt70']==True]
	latency=EEG_comp['sod_wt70'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region sod_wt71
	baseline_stop=10*60000
	maxlen_sod_wt=len(EEG_sod_wt.index)
	EEG_sod_wt_baseline=EEG_sod_wt.drop(range(baseline_stop,maxlen_sod_wt))
	inject_t0=int(baseline_stop+(5.56*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_sod_wt_PTZ=EEG_sod_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod_wt+1)))
	EEG_sod_wt_baseline=EEG_sod_wt_baseline.reset_index()
	EEG_sod_wt_PTZ=EEG_sod_wt_PTZ.reset_index()
	z=pd.DataFrame(EEG_sod_wt_baseline['sod_wt71'])
	y=pd.DataFrame(EEG_sod_wt_PTZ['sod_wt71'])
	sod_wt71z=abs(zscore(z,axis=None,nan_policy='omit'))
	sod_wt71zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(sod_wt71zptz>4*sod_wt71z)
	EEG_comp["sod_wt71"]=EEG_comp["sod_wt71"].astype(int)
	count_sod_wt71=EEG_comp['sod_wt71'].sum()
	EEG_counts_sod_wt71=EEG_comp.loc[EEG_comp['sod_wt71']==True]
	latency=EEG_comp['sod_wt71'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)    #endregion
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

	#region sod45
	baseline_stop=10*60000
	maxlen_sod=len(EEG_sod.index)
	EEG_sod_baseline=EEG_sod.drop(range(baseline_stop,maxlen_sod))
	inject_t0=int(baseline_stop+(6.26*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_sod_PTZ=EEG_sod.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod+1)))
	EEG_sod_baseline=EEG_sod_baseline.reset_index()
	EEG_sod_PTZ=EEG_sod_PTZ.reset_index()
	z=pd.DataFrame(EEG_sod_baseline['sod45'])
	y=pd.DataFrame(EEG_sod_PTZ['sod45'])
	sod45z=abs(zscore(z,axis=None,nan_policy='omit'))
	sod45zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(sod45zptz>4*sod45z)
	EEG_comp["sod45"]=EEG_comp["sod45"].astype(int)
	count_sod45=EEG_comp['sod45'].sum()
	EEG_counts_sod45=EEG_comp.loc[EEG_comp['sod45']==True]
	latency=EEG_comp['sod45'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region sod61
	baseline_stop=10*60000
	maxlen_sod=len(EEG_sod.index)
	EEG_sod_baseline=EEG_sod.drop(range(baseline_stop,maxlen_sod))
	inject_t0=int(baseline_stop+(6.08*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_sod_PTZ=EEG_sod.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod+1)))
	EEG_sod_baseline=EEG_sod_baseline.reset_index()
	EEG_sod_PTZ=EEG_sod_PTZ.reset_index()
	z=pd.DataFrame(EEG_sod_baseline['sod61'])
	y=pd.DataFrame(EEG_sod_PTZ['sod61'])
	sod61z=abs(zscore(z,axis=None,nan_policy='omit'))
	sod61zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(sod61zptz>4*sod61z)
	EEG_comp["sod61"]=EEG_comp["sod61"].astype(int)
	count_sod61=EEG_comp['sod61'].sum()
	EEG_counts_sod61=EEG_comp.loc[EEG_comp['sod61']==True]
	latency=EEG_comp['sod61'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region sod64
	baseline_stop=10*60000
	maxlen_sod=len(EEG_sod.index)
	EEG_sod_baseline=EEG_sod.drop(range(baseline_stop,maxlen_sod))
	inject_t0=int(baseline_stop+(6.12*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_sod_PTZ=EEG_sod.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod+1)))
	EEG_sod_baseline=EEG_sod_baseline.reset_index()
	EEG_sod_PTZ=EEG_sod_PTZ.reset_index()
	z=pd.DataFrame(EEG_sod_baseline['sod64'])
	y=pd.DataFrame(EEG_sod_PTZ['sod64'])
	sod64z=abs(zscore(z,axis=None,nan_policy='omit'))
	sod64zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(sod64zptz>4*sod64z)
	EEG_comp["sod64"]=EEG_comp["sod64"].astype(int)
	count_sod64=EEG_comp['sod64'].sum()
	EEG_counts_sod64=EEG_comp.loc[EEG_comp['sod64']==True]
	latency=EEG_comp['sod64'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region sod65
	baseline_stop=10*60000
	maxlen_sod=len(EEG_sod.index)
	EEG_sod_baseline=EEG_sod.drop(range(baseline_stop,maxlen_sod))
	inject_t0=int(baseline_stop+(6.11*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_sod_PTZ=EEG_sod.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod+1)))
	EEG_sod_baseline=EEG_sod_baseline.reset_index()
	EEG_sod_PTZ=EEG_sod_PTZ.reset_index()
	z=pd.DataFrame(EEG_sod_baseline['sod65'])
	y=pd.DataFrame(EEG_sod_PTZ['sod65'])
	sod65z=abs(zscore(z,axis=None,nan_policy='omit'))
	sod65zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(sod65zptz>4*sod65z)
	EEG_comp["sod65"]=EEG_comp["sod65"].astype(int)
	count_sod65=EEG_comp['sod65'].sum()
	EEG_counts_sod65=EEG_comp.loc[EEG_comp['sod65']==True]
	latency=EEG_comp['sod65'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region sod69
	baseline_stop=10*60000
	maxlen_sod=len(EEG_sod.index)
	EEG_sod_baseline=EEG_sod.drop(range(baseline_stop,maxlen_sod))
	inject_t0=int(baseline_stop+(5.45*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_sod_PTZ=EEG_sod.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_sod+1)))
	EEG_sod_baseline=EEG_sod_baseline.reset_index()
	EEG_sod_PTZ=EEG_sod_PTZ.reset_index()
	z=pd.DataFrame(EEG_sod_baseline['sod69'])
	y=pd.DataFrame(EEG_sod_PTZ['sod69'])
	sod69z=abs(zscore(z,axis=None,nan_policy='omit'))
	sod69zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(sod69zptz>4*sod69z)
	EEG_comp["sod69"]=EEG_comp["sod69"].astype(int)
	count_sod69=EEG_comp['sod69'].sum()
	EEG_counts_sod69=EEG_comp.loc[EEG_comp['sod69']==True]
	latency=EEG_comp['sod69'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)	#endregion
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

	#region fus_wt358
	baseline_stop=10*60000
	maxlen_fus_wt=len(EEG_fus_wt.index)
	EEG_fus_wt_baseline=EEG_fus_wt.drop(range(baseline_stop,maxlen_fus_wt))
	inject_t0=int(baseline_stop+(6.33*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_fus_wt_PTZ=EEG_fus_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus_wt+1)))
	EEG_fus_wt_baseline=EEG_fus_wt_baseline.reset_index()
	EEG_fus_wt_PTZ=EEG_fus_wt_PTZ.reset_index()
	z=pd.DataFrame(EEG_fus_wt_baseline['fus_wt358'])
	y=pd.DataFrame(EEG_fus_wt_PTZ['fus_wt358'])
	fus_wt358z=abs(zscore(z,axis=None,nan_policy='omit'))
	fus_wt358zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(fus_wt358zptz>6*fus_wt358z)
	EEG_comp["fus_wt358"]=EEG_comp["fus_wt358"].astype(int)
	count_fus_wt358=EEG_comp['fus_wt358'].sum()
	EEG_counts_fus_wt358=EEG_comp.loc[EEG_comp['fus_wt358']==True]
	latency=EEG_comp['fus_wt358'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region fus_wt360
	baseline_stop=10*60000
	maxlen_fus_wt=len(EEG_fus_wt.index)
	EEG_fus_wt_baseline=EEG_fus_wt.drop(range(baseline_stop,maxlen_fus_wt))
	inject_t0=int(baseline_stop+(7.17*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_fus_wt_PTZ=EEG_fus_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus_wt+1)))
	EEG_fus_wt_baseline=EEG_fus_wt_baseline.reset_index()
	EEG_fus_wt_PTZ=EEG_fus_wt_PTZ.reset_index()
	z=pd.DataFrame(EEG_fus_wt_baseline['fus_wt360'])
	y=pd.DataFrame(EEG_fus_wt_PTZ['fus_wt360'])
	fus_wt360z=abs(zscore(z,axis=None,nan_policy='omit'))
	fus_wt360zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(fus_wt360zptz>6*fus_wt360z)
	EEG_comp["fus_wt360"]=EEG_comp["fus_wt360"].astype(int)
	count_fus_wt360=EEG_comp['fus_wt360'].sum()
	EEG_counts_fus_wt360=EEG_comp.loc[EEG_comp['fus_wt360']==True]
	latency=EEG_comp['fus_wt360'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region fus_wt361
	baseline_stop=10*60000
	maxlen_fus_wt=len(EEG_fus_wt.index)
	EEG_fus_wt_baseline=EEG_fus_wt.drop(range(baseline_stop,maxlen_fus_wt))
	inject_t0=int(baseline_stop+(7.04*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_fus_wt_PTZ=EEG_fus_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus_wt+1)))
	EEG_fus_wt_baseline=EEG_fus_wt_baseline.reset_index()
	EEG_fus_wt_PTZ=EEG_fus_wt_PTZ.reset_index()
	z=pd.DataFrame(EEG_fus_wt_baseline['fus_wt361'])
	y=pd.DataFrame(EEG_fus_wt_PTZ['fus_wt361'])
	fus_wt361z=abs(zscore(z,axis=None,nan_policy='omit'))
	fus_wt361zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(fus_wt361zptz>6*fus_wt361z)
	EEG_comp["fus_wt361"]=EEG_comp["fus_wt361"].astype(int)
	count_fus_wt361=EEG_comp['fus_wt361'].sum()
	EEG_counts_fus_wt361=EEG_comp.loc[EEG_comp['fus_wt361']==True]
	latency=EEG_comp['fus_wt361'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region fus_wt363
	baseline_stop=10*60000
	maxlen_fus_wt=len(EEG_fus_wt.index)
	EEG_fus_wt_baseline=EEG_fus_wt.drop(range(baseline_stop,maxlen_fus_wt))
	inject_t0=int(baseline_stop+(6.58*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_fus_wt_PTZ=EEG_fus_wt.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus_wt+1)))
	EEG_fus_wt_baseline=EEG_fus_wt_baseline.reset_index()
	EEG_fus_wt_PTZ=EEG_fus_wt_PTZ.reset_index()
	z=pd.DataFrame(EEG_fus_wt_baseline['fus_wt363'])
	y=pd.DataFrame(EEG_fus_wt_PTZ['fus_wt363'])
	fus_wt363z=abs(zscore(z,axis=None,nan_policy='omit'))
	fus_wt363zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(fus_wt363zptz>6*fus_wt363z)
	EEG_comp["fus_wt363"]=EEG_comp["fus_wt363"].astype(int)
	count_fus_wt363=EEG_comp['fus_wt363'].sum()
	EEG_counts_fus_wt363=EEG_comp.loc[EEG_comp['fus_wt363']==True]
	latency=EEG_comp['fus_wt363'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)	#endregion
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

	#region fus362
	baseline_stop=10*60000
	maxlen_fus=len(EEG_fus.index)
	EEG_fus_baseline=EEG_fus.drop(range(baseline_stop,maxlen_fus))
	inject_t0=int(baseline_stop+(7.11*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_fus_PTZ=EEG_fus.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus+1)))
	EEG_fus_baseline=EEG_fus_baseline.reset_index()
	EEG_fus_PTZ=EEG_fus_PTZ.reset_index()
	z=pd.DataFrame(EEG_fus_baseline['fus362'])
	y=pd.DataFrame(EEG_fus_PTZ['fus362'])
	fus362z=abs(zscore(z,axis=None,nan_policy='omit'))
	fus362zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(fus362zptz>6*fus362z)
	EEG_comp["fus362"]=EEG_comp["fus362"].astype(int)
	count_fus362=EEG_comp['fus362'].sum()
	EEG_counts_fus362=EEG_comp.loc[EEG_comp['fus362']==True]
	latency=EEG_comp['fus362'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region fus364
	baseline_stop=10*60000
	maxlen_fus=len(EEG_fus.index)
	EEG_fus_baseline=EEG_fus.drop(range(baseline_stop,maxlen_fus))
	inject_t0=int(baseline_stop+(7.08*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_fus_PTZ=EEG_fus.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus+1)))
	EEG_fus_baseline=EEG_fus_baseline.reset_index()
	EEG_fus_PTZ=EEG_fus_PTZ.reset_index()
	z=pd.DataFrame(EEG_fus_baseline['fus364'])
	y=pd.DataFrame(EEG_fus_PTZ['fus364'])
	fus364z=abs(zscore(z,axis=None,nan_policy='omit'))
	fus364zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(fus364zptz>6*fus364z)
	EEG_comp["fus364"]=EEG_comp["fus364"].astype(int)
	count_fus364=EEG_comp['fus364'].sum()
	EEG_counts_fus364=EEG_comp.loc[EEG_comp['fus364']==True]
	latency=EEG_comp['fus364'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region fus365
	baseline_stop=10*60000
	maxlen_fus=len(EEG_fus.index)
	EEG_fus_baseline=EEG_fus.drop(range(baseline_stop,maxlen_fus))
	inject_t0=int(baseline_stop+(6.32*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_fus_PTZ=EEG_fus.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus+1)))
	EEG_fus_baseline=EEG_fus_baseline.reset_index()
	EEG_fus_PTZ=EEG_fus_PTZ.reset_index()
	z=pd.DataFrame(EEG_fus_baseline['fus365'])
	y=pd.DataFrame(EEG_fus_PTZ['fus365'])
	fus365z=abs(zscore(z,axis=None,nan_policy='omit'))
	fus365zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(fus365zptz>6*fus365z)
	EEG_comp["fus365"]=EEG_comp["fus365"].astype(int)
	count_fus365=EEG_comp['fus365'].sum()
	EEG_counts_fus365=EEG_comp.loc[EEG_comp['fus365']==True]
	latency=EEG_comp['fus365'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region fus372
	baseline_stop=10*60000
	maxlen_fus=len(EEG_fus.index)
	EEG_fus_baseline=EEG_fus.drop(range(baseline_stop,maxlen_fus))
	inject_t0=int(baseline_stop+(6.22*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_fus_PTZ=EEG_fus.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus+1)))
	EEG_fus_baseline=EEG_fus_baseline.reset_index()
	EEG_fus_PTZ=EEG_fus_PTZ.reset_index()
	z=pd.DataFrame(EEG_fus_baseline['fus372'])
	y=pd.DataFrame(EEG_fus_PTZ['fus372'])
	fus372z=abs(zscore(z,axis=None,nan_policy='omit'))
	fus372zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(fus372zptz>6*fus372z)
	EEG_comp["fus372"]=EEG_comp["fus372"].astype(int)
	count_fus372=EEG_comp['fus372'].sum()
	EEG_counts_fus372=EEG_comp.loc[EEG_comp['fus372']==True]
	latency=EEG_comp['fus372'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)
	#endregion
	#region fus373
	baseline_stop=10*60000
	maxlen_fus=len(EEG_fus.index)
	EEG_fus_baseline=EEG_fus.drop(range(baseline_stop,maxlen_fus))
	inject_t0=int(baseline_stop+(6.19*60000))
	inject_t1=int(inject_t0+(10*60000))
	EEG_fus_PTZ=EEG_fus.drop(list(range(1,inject_t0))+list(range(inject_t1,maxlen_fus+1)))
	EEG_fus_baseline=EEG_fus_baseline.reset_index()
	EEG_fus_PTZ=EEG_fus_PTZ.reset_index()
	z=pd.DataFrame(EEG_fus_baseline['fus373'])
	y=pd.DataFrame(EEG_fus_PTZ['fus373'])
	fus373z=abs(zscore(z,axis=None,nan_policy='omit'))
	fus373zptz=abs(zscore(y,axis=None,nan_policy='omit'))
	EEG_comp=pd.DataFrame(fus373zptz>6*fus373z)
	EEG_comp["fus373"]=EEG_comp["fus373"].astype(int)
	count_fus373=EEG_comp['fus373'].sum()
	EEG_counts_fus373=EEG_comp.loc[EEG_comp['fus373']==True]
	latency=EEG_comp['fus373'].idxmax()+60
	latency=latency/60
	frac,whole=math.modf(latency)
	frac=0.6*frac
	latency=whole+frac
	print('%.3fmin'%latency)	#endregion
#endregion
