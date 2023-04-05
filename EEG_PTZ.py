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


#region Envelope analysis SOD
#region EEG WT upload

#region Dir path
path=fd.askdirectory(title='SELECT WHERE TO SAVE PLOTS')
inpath=fd.askdirectory(title='SELECT DATA DIRECTORY')
dir_edf=sorted(glob.glob(os.path.join(inpath,'*.edf')))
dname=[]
EEG_wt=pd.DataFrame()
start=time.process_time()
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
	raw_hil=raw_notch.apply_hilbert(envelope=True)
	raw_hil.compute_psd().plot()
	data=raw_notch.get_data(units="uV")
	print(data.shape)
	data=data.reshape(-1)
	print(data.shape)
	data=pd.DataFrame(data.T)
	EEG_wt=pd.concat([EEG_wt,data],ignore_index=True,axis=1)
	#EEG_wt.append(data,ignore_index=True)


else:
	#region Data export
	EEG_wt=EEG_wt.reset_index(drop=True)
	EEG_wt.index=range(1,EEG_wt.shape[0]+1)
	EEG_wt.columns=['WT43','WT63','WT66','WT70','WT71']

	#Baseline window 10mn
	maxlen_wt=len(EEG_wt.index)
	EEG_wt_baseline=EEG_wt.drop(range(600000,maxlen_wt))
	EEG_wt_baseline['avEEG_WT']=EEG_wt_baseline[['WT43','WT63','WT66','WT70','WT71']].mean(axis=1)
	EEG_wt_baseline['avzscore']=zscore(EEG_wt_baseline['avEEG_WT'])
	EEG_wt_baseline['WT43z']=zscore(EEG_wt_baseline['WT43'])


	#TEST
	maxval_wt1=max(EEG_wt_baseline['avzscore'])
	minval_wt1=min(EEG_wt_baseline['avzscore'])
	sd_pos_wt1=np.std(EEG_wt_baseline['avzscore'])*5
	sd_neg_wt1=np.std(EEG_wt_baseline['avzscore'])*-5
	wt1_thres_pos=np.array([np.NaN if diff_wt1_<sd_pos_wt1 else diff_wt1_ for diff_wt1_ in EEG_wt_baseline['WT43z']])
	wt1_thres_pos=wt1_thres_pos[~np.isnan(wt1_thres_pos)]
	#Scanning the whole recording to pin values that are strictly above the threshold
	EEG_wt_baseline['Test']=EEG_wt_baseline['WT43z']>sd_pos_wt1
	#Counting the values strictly above the threshold
	count_wt1=EEG_wt_baseline['Test'].sum()
	EEG_counts_wt1=EEG_wt_baseline.loc[EEG_wt_baseline['Test']==True]
	#ENDTEST

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

	name00="EEG_SOD-WT_env.png"
	path00=os.path.join(path,name00)
	plt.savefig(path00,dpi=1200,bbox_inches='tight',transparent=True,pad_inches=0)
	plt.show()
	print("Plot done.")

	#PTZ injection window 10mn
	EEG_wt_PTZ=EEG_wt.drop(list(range(0,1020000))+list(range(1620000,maxlen_wt)))
	EEG_wt_PTZ['avEEG_WT']=EEG_wt_PTZ[['WT43','WT63','WT66','WT70','WT71']].mean(axis=1)


	name1=('EEG_WT.csv')
	path1=os.path.join(path,name1)
	EEG_wt.to_csv(path1)	#endregion




EEG_wt_1=pd.read_csv(fd.askopenfilename(title='SELECT EEG WT 1/5'))
EEG_wt_1['aEEG']=np.abs(hilbert(EEG_wt_1['EEG']))
EEG_wt_1=EEG_wt_1.add_suffix('_1')
EEG_wt_1=EEG_wt_1.rename({'Time Stamp_1':'HMS'},axis=1)

EEG_wt_2=pd.read_csv(fd.askopenfilename(title='SELECT EEG WT 2/5'))
EEG_wt_2=EEG_wt_2.iloc[:,1:]
EEG_wt_2['aEEG']=np.abs(hilbert(EEG_wt_2['EEG']))
EEG_wt_2=EEG_wt_2.add_suffix('_2')

EEG_wt_3=pd.read_csv(fd.askopenfilename(title='SELECT EEG WT 3/5'))
EEG_wt_3=EEG_wt_3.iloc[:,1:]
EEG_wt_3['aEEG']=np.abs(hilbert(EEG_wt_3['EEG']))
EEG_wt_3=EEG_wt_3.add_suffix('_3')

EEG_wt_4=pd.read_csv(fd.askopenfilename(title='SELECT EEG WT 4/5'))
EEG_wt_4=EEG_wt_4.iloc[:,1:]
EEG_wt_4['aEEG']=np.abs(hilbert(EEG_wt_4['EEG']))
EEG_wt_4=EEG_wt_4.add_suffix('_4')

EEG_wt_5=pd.read_csv(fd.askopenfilename(title='SELECT EEG WT 5/5'))
EEG_wt_5=EEG_wt_5.iloc[:,1:]
EEG_wt_5['aEEG']=np.abs(hilbert(EEG_wt_5['EEG']))
EEG_wt_5=EEG_wt_5.add_suffix('_5')

EEG_wt=pd.concat([EEG_wt_1,EEG_wt_2,EEG_wt_3,EEG_wt_4,EEG_wt_5],axis=1)
maxlen_wt=len(EEG_wt.index)
#EEG_wt=EEG_wt.drop(range(1338,maxlen_wt)) #1mn recording
EEG_wt=EEG_wt.drop(range(8032,maxlen_wt)) #10s recording
EEG_wt['avEEG_WT']=EEG_wt[["EEG_1","EEG_2","EEG_3","EEG_4","EEG_5"]].mean(axis=1)
EEG_wt['avhEEG_WT']=EEG_wt[["aEEG_1","aEEG_2","aEEG_3","aEEG_4","aEEG_5"]].mean(axis=1)
print("EEG WT loaded.")
#endregion
#region Create envelope + plot
windowsize=20
env_wt=pd.DataFrame()
env_wt['wt_upperEnv']=EEG_wt['avhEEG_WT'].rolling(window=windowsize).max().shift(int(-windowsize/2))
env_wt['wt_lowerEnv']=EEG_wt['avhEEG_WT'].rolling(window=windowsize).min().shift(int(-windowsize/2))
print("Envelope created.")
#region Plot envelope
# x=range(8031)
# y=diff_wt['EEG_mean']
# df=pd.DataFrame(data={"y":y},index=x)
# extract_col=diff_wt['EEG_1']
# df=df.join(extract_col)
# windowsize=20
# df['y_upperEnv']=df['y'].rolling(window=windowsize).max().shift(int(-windowsize/2))
# df['y_lowerEnv']=df['y'].rolling(window=windowsize).min().shift(int(-windowsize/2))
# cmap=ListedColormap(['none','red','grey','grey'])
# df.plot(cmap=cmap)
# plt.tight_layout()
# plt.show()
x=range(8031)
y=EEG_wt['avhEEG_WT']
df=pd.DataFrame(data={"y":y},index=x)
extract_col=EEG_wt['aEEG_1']
df=df.join(extract_col)
windowsize=20
df['y_upperEnv']=env_wt['wt_upperEnv']
df['y_lowerEnv']=env_wt['wt_lowerEnv']
cmap=ListedColormap(['none','red','grey'])
df.plot(cmap=cmap,linewidth=0.75)
plt.tight_layout()
name00="EEG_WT_3m_env.png"
path00=os.path.join(path,name00)
plt.savefig(path00,dpi=1200,bbox_inches='tight',transparent=True,pad_inches=0)
plt.show()
print("Plot done.")
#endregion
#endregion
#region Scan EEG signal outside of up&low envelope
env_wt['EEG_1_range']=(EEG_wt['EEG_1']<=env_wt['wt_upperEnv'])&(EEG_wt['EEG_1']>=env_wt['wt_lowerEnv'])
env_wt['EEG_2_range']=(EEG_wt['EEG_2']<=env_wt['wt_upperEnv'])&(EEG_wt['EEG_2']>=env_wt['wt_lowerEnv'])
env_wt['EEG_3_range']=(EEG_wt['EEG_3']<=env_wt['wt_upperEnv'])&(EEG_wt['EEG_3']>=env_wt['wt_lowerEnv'])
env_wt['EEG_4_range']=(EEG_wt['EEG_4']<=env_wt['wt_upperEnv'])&(EEG_wt['EEG_4']>=env_wt['wt_lowerEnv'])
env_wt['EEG_5_range']=(EEG_wt['EEG_5']<=env_wt['wt_upperEnv'])&(EEG_wt['EEG_5']>=env_wt['wt_lowerEnv'])
print("EEG WT scanned.")
#endregion
#region Scan EEG signal outside of Hilbert envelope
env_wt['EEG_1_range']=(EEG_wt['aEEG_1']<=env_wt['wt_upperEnv'])
env_wt['EEG_2_range']=(EEG_wt['aEEG_2']<=env_wt['wt_upperEnv'])
env_wt['EEG_3_range']=(EEG_wt['aEEG_3']<=env_wt['wt_upperEnv'])
env_wt['EEG_4_range']=(EEG_wt['aEEG_4']<=env_wt['wt_upperEnv'])
env_wt['EEG_5_range']=(EEG_wt['aEEG_5']<=env_wt['wt_upperEnv'])
print("EEG WT scanned.")
#endregion
#region Drop NaN
env_wt=env_wt.drop(range(0,9))
env_wt=env_wt.drop(range(8022,8032))
env_wt=env_wt.reset_index()
env_wt=env_wt.drop('index',axis=1)
#endregion
#region Values > threshold
count_wt1=(env_wt['EEG_1_range']==False).sum()
EEG_counts_wt1=env_wt.loc[env_wt['EEG_1_range']==False]
count_wt2=(env_wt['EEG_2_range']==False).sum()
EEG_counts_wt2=env_wt.loc[env_wt['EEG_2_range']==False]
count_wt3=(env_wt['EEG_3_range']==False).sum()
EEG_counts_wt3=env_wt.loc[env_wt['EEG_3_range']==False]
count_wt4=(env_wt['EEG_4_range']==False).sum()
EEG_counts_wt4=env_wt.loc[env_wt['EEG_4_range']==False]
count_wt5=(env_wt['EEG_5_range']==False).sum()
EEG_counts_wt5=env_wt.loc[env_wt['EEG_5_range']==False]
EEG_counts_wt=count_wt1,count_wt2,count_wt3,count_wt4,count_wt5
EEG_counts_wt=pd.DataFrame(EEG_counts_wt)
EEG_counts_wt.columns=['EEG']
EEG_counts_wt['Mean']=EEG_counts_wt['EEG'].mean()
print("Counting WT done.")
#endregion
print("WT done.")

#region EEG SOD upload
print('SELECT EEG SOD')
EEG_sod_1=pd.read_csv(fd.askopenfilename(title='SELECT EEG SOD 1/6'))
EEG_sod_1['aEEG']=np.abs(hilbert(EEG_sod_1['EEG']))
EEG_sod_1=EEG_sod_1.add_suffix('_1')
EEG_sod_1=EEG_sod_1.rename({'Time Stamp_1':'HMS'},axis=1)

EEG_sod_2=pd.read_csv(fd.askopenfilename(title='SELECT EEG SOD 2/6'))
EEG_sod_2=EEG_sod_2.iloc[:,1:]
EEG_sod_2['aEEG']=np.abs(hilbert(EEG_sod_2['EEG']))
EEG_sod_2=EEG_sod_2.add_suffix('_2')

EEG_sod_3=pd.read_csv(fd.askopenfilename(title='SELECT EEG SOD 3/6'))
EEG_sod_3=EEG_sod_3.iloc[:,1:]
EEG_sod_3['aEEG']=np.abs(hilbert(EEG_sod_3['EEG']))
EEG_sod_3=EEG_sod_3.add_suffix('_3')

EEG_sod_4=pd.read_csv(fd.askopenfilename(title='SELECT EEG SOD 4/6'))
EEG_sod_4=EEG_sod_4.iloc[:,1:]
EEG_sod_4['aEEG']=np.abs(hilbert(EEG_sod_4['EEG']))
EEG_sod_4=EEG_sod_4.add_suffix('_4')

EEG_sod_5=pd.read_csv(fd.askopenfilename(title='SELECT EEG SOD 5/6'))
EEG_sod_5=EEG_sod_5.iloc[:,1:]
EEG_sod_5['aEEG']=np.abs(hilbert(EEG_sod_5['EEG']))
EEG_sod_5=EEG_sod_5.add_suffix('_5')

EEG_sod=pd.concat([EEG_sod_1,EEG_sod_2,EEG_sod_3,EEG_sod_4,EEG_sod_5],axis=1)
EEG_sod=EEG_sod.rename({'Time Stamp_1':'HMS'},axis=1)
maxlen_sod=len(EEG_sod.index)
#EEG_sod=EEG_sod.drop(range(1338,maxlen_sod)) #1mn recording
EEG_sod=EEG_sod.drop(range(8032,maxlen_sod)) #10s recording
EEG_sod['avEEG_sod']=EEG_sod[["EEG_1","EEG_2","EEG_3","EEG_4","EEG_5"]].mean(axis=1)
print("EEG SOD loaded.")
#endregion
#region Plot envelope
x=range(8031)
y=EEG_wt['avEEG_WT']
df=pd.DataFrame(data={"y":y},index=x)
extract_col=EEG_sod['EEG_4']
df=df.join(extract_col)
windowsize=20
df['y_upperEnv']=env_wt['wt_upperEnv']
#df['y_lowerEnv']=env_wt['wt_lowerEnv']
cmap=ListedColormap(['none','red','grey','grey'])
df.plot(cmap=cmap,linewidth=0.75)
plt.tight_layout()
name00="EEG_sod_3m_env.png"
path00=os.path.join(path,name00)
plt.savefig(path00,dpi=1200,bbox_inches='tight',transparent=True,pad_inches=0)
plt.show()
print("Plot done.")
#endregion
#region Scan EEG signal outside of up&low envelope
env_sod=pd.DataFrame()
env_sod['wt_upperEnv']=EEG_wt['avEEG_WT'].rolling(window=windowsize).max().shift(int(-windowsize/2))
env_sod['wt_lowerEnv']=EEG_wt['avEEG_WT'].rolling(window=windowsize).min().shift(int(-windowsize/2))
env_sod['EEG_1_range']=(EEG_sod['EEG_1']<=env_sod['wt_upperEnv'])&(EEG_sod['EEG_1']>=env_sod['wt_lowerEnv'])
env_sod['EEG_2_range']=(EEG_sod['EEG_2']<=env_sod['wt_upperEnv'])&(EEG_sod['EEG_2']>=env_sod['wt_lowerEnv'])
env_sod['EEG_3_range']=(EEG_sod['EEG_3']<=env_sod['wt_upperEnv'])&(EEG_sod['EEG_3']>=env_sod['wt_lowerEnv'])
env_sod['EEG_4_range']=(EEG_sod['EEG_4']<=env_sod['wt_upperEnv'])&(EEG_sod['EEG_4']>=env_sod['wt_lowerEnv'])
env_sod['EEG_5_range']=(EEG_sod['EEG_5']<=env_sod['wt_upperEnv'])&(EEG_sod['EEG_5']>=env_sod['wt_lowerEnv'])
env_sod['EEG_6_range']=(EEG_sod['EEG_6']<=env_sod['wt_upperEnv'])&(EEG_sod['EEG_6']>=env_sod['wt_lowerEnv'])
print("EEG SOD scanned.")
#endregion
#region Scan EEG signal outside of Hilbert envelope
env_sod=pd.DataFrame()
env_sod['wt_upperEnv']=EEG_wt['avEEG_WT'].rolling(window=windowsize).max().shift(int(-windowsize/2))
env_sod['EEG_1_range']=(EEG_sod['EEG_1']<=env_sod['wt_upperEnv'])
env_sod['EEG_2_range']=(EEG_sod['EEG_2']<=env_sod['wt_upperEnv'])
env_sod['EEG_3_range']=(EEG_sod['EEG_3']<=env_sod['wt_upperEnv'])
env_sod['EEG_4_range']=(EEG_sod['EEG_4']<=env_sod['wt_upperEnv'])
env_sod['EEG_5_range']=(EEG_sod['EEG_5']<=env_sod['wt_upperEnv'])
env_sod['EEG_6_range']=(EEG_sod['EEG_6']<=env_sod['wt_upperEnv'])
print("EEG SOD scanned.")
#endregion
#region Drop NaN
env_sod=env_sod.drop(range(0,9))
env_sod=env_sod.drop(range(8022,8032))
env_sod=env_sod.reset_index()
env_sod=env_sod.drop('index',axis=1)
#endregion
#region Values > threshold
count_sod1=(env_sod['EEG_1_range']==False).sum()
EEG_counts_sod1=env_sod.loc[env_sod['EEG_1_range']==False]
count_sod2=(env_sod['EEG_2_range']==False).sum()
EEG_counts_sod2=env_sod.loc[env_sod['EEG_2_range']==False]
count_sod3=(env_sod['EEG_3_range']==False).sum()
EEG_counts_sod3=env_sod.loc[env_sod['EEG_3_range']==False]
count_sod4=(env_sod['EEG_4_range']==False).sum()
EEG_counts_sod4=env_sod.loc[env_sod['EEG_4_range']==False]
count_sod5=(env_sod['EEG_5_range']==False).sum()
EEG_counts_sod5=env_sod.loc[env_sod['EEG_5_range']==False]
count_sod6=(env_sod['EEG_6_range']==False).sum()
EEG_counts_sod6=env_sod.loc[env_sod['EEG_6_range']==False]
EEG_counts_sod=count_sod1,count_sod2,count_sod3,count_sod4,count_sod5,count_sod6
EEG_counts_sod=pd.DataFrame(EEG_counts_sod)
EEG_counts_sod.columns=['EEG']
EEG_counts_sod['Mean']=EEG_counts_sod['EEG'].mean()
print("Counting SOD done.")
#endregion
print("SOD done.")
#endregion

#region Envelope analysis FUS
#region EEG WT upload
print('SELECT EEG WT')
EEG_wt_1=pd.read_csv(fd.askopenfilename(title='SELECT EEG WT 1/5'))
EEG_wt_1['aEEG']=np.abs(hilbert(EEG_wt_1['EEG']))
EEG_wt_1=EEG_wt_1.add_suffix('_1')
EEG_wt_1=EEG_wt_1.rename({'Time Stamp_1':'HMS'},axis=1)

EEG_wt_2=pd.read_csv(fd.askopenfilename(title='SELECT EEG WT 2/5'))
EEG_wt_2=EEG_wt_2.iloc[:,1:]
EEG_wt_2['aEEG']=np.abs(hilbert(EEG_wt_2['EEG']))
EEG_wt_2=EEG_wt_2.add_suffix('_2')

EEG_wt_3=pd.read_csv(fd.askopenfilename(title='SELECT EEG WT 3/5'))
EEG_wt_3=EEG_wt_3.iloc[:,1:]
EEG_wt_3['aEEG']=np.abs(hilbert(EEG_wt_3['EEG']))
EEG_wt_3=EEG_wt_3.add_suffix('_3')

EEG_wt_4=pd.read_csv(fd.askopenfilename(title='SELECT EEG WT 4/5'))
EEG_wt_4=EEG_wt_4.iloc[:,1:]
EEG_wt_4['aEEG']=np.abs(hilbert(EEG_wt_4['EEG']))
EEG_wt_4=EEG_wt_4.add_suffix('_4')

EEG_wt=pd.concat([EEG_wt_1,EEG_wt_2,EEG_wt_3,EEG_wt_4],axis=1)
maxlen_wt=len(EEG_wt.index)
#EEG_wt=EEG_wt.drop(range(1338,maxlen_wt)) #1mn recording
EEG_wt=EEG_wt.drop(range(8032,maxlen_wt)) #10s recording
EEG_wt['avEEG_WT']=EEG_wt[["EEG_1","EEG_2","EEG_3","EEG_4"]].mean(axis=1)
EEG_wt['avhEEG_WT']=EEG_wt[["aEEG_1","aEEG_2","aEEG_3","aEEG_4"]].mean(axis=1)
print("EEG WT loaded.")
#endregion
#region Create envelope + plot
windowsize=20
env_wt=pd.DataFrame()
env_wt['wt_upperEnv']=EEG_wt['avhEEG_WT'].rolling(window=windowsize).max().shift(int(-windowsize/2))
env_wt['wt_lowerEnv']=EEG_wt['avhEEG_WT'].rolling(window=windowsize).min().shift(int(-windowsize/2))
print("Envelope created.")
#region Plot envelope
# x=range(8031)
# y=diff_wt['EEG_mean']
# df=pd.DataFrame(data={"y":y},index=x)
# extract_col=diff_wt['EEG_1']
# df=df.join(extract_col)
# windowsize=20
# df['y_upperEnv']=df['y'].rolling(window=windowsize).max().shift(int(-windowsize/2))
# df['y_lowerEnv']=df['y'].rolling(window=windowsize).min().shift(int(-windowsize/2))
# cmap=ListedColormap(['none','red','grey','grey'])
# df.plot(cmap=cmap)
# plt.tight_layout()
# plt.show()
x=range(8031)
y=EEG_wt['avhEEG_WT']
df=pd.DataFrame(data={"y":y},index=x)
extract_col=EEG_wt['aEEG_1']
df=df.join(extract_col)
windowsize=20
df['y_upperEnv']=env_wt['wt_upperEnv']
df['y_lowerEnv']=env_wt['wt_lowerEnv']
cmap=ListedColormap(['none','red','grey'])
df.plot(cmap=cmap,linewidth=0.75)
plt.tight_layout()
name00="EEG_WT_3m_env.png"
path00=os.path.join(path,name00)
plt.savefig(path00,dpi=1200,bbox_inches='tight',transparent=True,pad_inches=0)
plt.show()
print("Plot done.")
#endregion
#endregion
#region Scan EEG signal outside of up&low envelope
env_wt['EEG_1_range']=(EEG_wt['EEG_1']<=env_wt['wt_upperEnv'])&(EEG_wt['EEG_1']>=env_wt['wt_lowerEnv'])
env_wt['EEG_2_range']=(EEG_wt['EEG_2']<=env_wt['wt_upperEnv'])&(EEG_wt['EEG_2']>=env_wt['wt_lowerEnv'])
env_wt['EEG_3_range']=(EEG_wt['EEG_3']<=env_wt['wt_upperEnv'])&(EEG_wt['EEG_3']>=env_wt['wt_lowerEnv'])
env_wt['EEG_4_range']=(EEG_wt['EEG_4']<=env_wt['wt_upperEnv'])&(EEG_wt['EEG_4']>=env_wt['wt_lowerEnv'])
print("EEG WT scanned.")
#endregion
#region Scan EEG signal outside of Hilbert envelope
env_wt['EEG_1_range']=(EEG_wt['aEEG_1']<=env_wt['wt_upperEnv'])
env_wt['EEG_2_range']=(EEG_wt['aEEG_2']<=env_wt['wt_upperEnv'])
env_wt['EEG_3_range']=(EEG_wt['aEEG_3']<=env_wt['wt_upperEnv'])
env_wt['EEG_4_range']=(EEG_wt['aEEG_4']<=env_wt['wt_upperEnv'])
print("EEG WT scanned.")
#endregion
#region Drop NaN
env_wt=env_wt.drop(range(0,9))
env_wt=env_wt.drop(range(8022,8032))
env_wt=env_wt.reset_index()
env_wt=env_wt.drop('index',axis=1)
#endregion
#region Values > threshold
count_wt1=(env_wt['EEG_1_range']==False).sum()
EEG_counts_wt1=env_wt.loc[env_wt['EEG_1_range']==False]
count_wt2=(env_wt['EEG_2_range']==False).sum()
EEG_counts_wt2=env_wt.loc[env_wt['EEG_2_range']==False]
count_wt3=(env_wt['EEG_3_range']==False).sum()
EEG_counts_wt3=env_wt.loc[env_wt['EEG_3_range']==False]
count_wt4=(env_wt['EEG_4_range']==False).sum()
EEG_counts_wt4=env_wt.loc[env_wt['EEG_4_range']==False]
EEG_counts_wt=count_wt1,count_wt2,count_wt3,count_wt4
EEG_counts_wt=pd.DataFrame(EEG_counts_wt)
EEG_counts_wt.columns=['EEG']
EEG_counts_wt['Mean']=EEG_counts_wt['EEG'].mean()
print("Counting WT done.")
#endregion
print("WT done.")

#region EEG fus upload
print('SELECT EEG FUS')
EEG_fus_1=pd.read_csv(fd.askopenfilename(title='SELECT EEG fus 1/6'))
EEG_fus_1['aEEG']=np.abs(hilbert(EEG_fus_1['EEG']))
EEG_fus_1=EEG_fus_1.add_suffix('_1')
EEG_fus_1=EEG_fus_1.rename({'Time Stamp_1':'HMS'},axis=1)

EEG_fus_2=pd.read_csv(fd.askopenfilename(title='SELECT EEG fus 2/6'))
EEG_fus_2=EEG_fus_2.iloc[:,1:]
EEG_fus_2['aEEG']=np.abs(hilbert(EEG_fus_2['EEG']))
EEG_fus_2=EEG_fus_2.add_suffix('_2')

EEG_fus_3=pd.read_csv(fd.askopenfilename(title='SELECT EEG fus 3/6'))
EEG_fus_3=EEG_fus_3.iloc[:,1:]
EEG_fus_3['aEEG']=np.abs(hilbert(EEG_fus_3['EEG']))
EEG_fus_3=EEG_fus_3.add_suffix('_3')

EEG_fus_4=pd.read_csv(fd.askopenfilename(title='SELECT EEG fus 4/6'))
EEG_fus_4=EEG_fus_4.iloc[:,1:]
EEG_fus_4['aEEG']=np.abs(hilbert(EEG_fus_4['EEG']))
EEG_fus_4=EEG_fus_4.add_suffix('_4')

EEG_fus_5=pd.read_csv(fd.askopenfilename(title='SELECT EEG fus 5/6'))
EEG_fus_5=EEG_fus_5.iloc[:,1:]
EEG_fus_5['aEEG']=np.abs(hilbert(EEG_fus_5['EEG']))
EEG_fus_5=EEG_fus_5.add_suffix('_5')

EEG_fus=pd.concat([EEG_fus_1,EEG_fus_2,EEG_fus_3,EEG_fus_4,EEG_fus_5],axis=1)
EEG_fus=EEG_fus.rename({'Time Stamp_1':'HMS'},axis=1)
maxlen_fus=len(EEG_fus.index)
#EEG_fus=EEG_fus.drop(range(1338,maxlen_fus)) #1mn recording
EEG_fus=EEG_fus.drop(range(8032,maxlen_fus)) #10s recording
EEG_fus['avEEG_fus']=EEG_fus[["EEG_1","EEG_2","EEG_3","EEG_4","EEG_5"]].mean(axis=1)
print("EEG FUS loaded.")
#endregion
#region Plot envelope
x=range(8031)
y=EEG_wt['avEEG_WT']
df=pd.DataFrame(data={"y":y},index=x)
extract_col=EEG_fus['EEG_4']
df=df.join(extract_col)
windowsize=20
df['y_upperEnv']=env_wt['wt_upperEnv']
#df['y_lowerEnv']=env_wt['wt_lowerEnv']
cmap=ListedColormap(['none','red','grey','grey'])
df.plot(cmap=cmap,linewidth=0.75)
plt.tight_layout()
name00="EEG_fus_3m_env.png"
path00=os.path.join(path,name00)
plt.savefig(path00,dpi=1200,bbox_inches='tight',transparent=True,pad_inches=0)
plt.show()
print("Plot done.")
#endregion
#region Scan EEG signal outside of up&low envelope
env_fus=pd.DataFrame()
env_fus['wt_upperEnv']=EEG_wt['avEEG_WT'].rolling(window=windowsize).max().shift(int(-windowsize/2))
env_fus['wt_lowerEnv']=EEG_wt['avEEG_WT'].rolling(window=windowsize).min().shift(int(-windowsize/2))
env_fus['EEG_1_range']=(EEG_fus['EEG_1']<=env_fus['wt_upperEnv'])&(EEG_fus['EEG_1']>=env_fus['wt_lowerEnv'])
env_fus['EEG_2_range']=(EEG_fus['EEG_2']<=env_fus['wt_upperEnv'])&(EEG_fus['EEG_2']>=env_fus['wt_lowerEnv'])
env_fus['EEG_3_range']=(EEG_fus['EEG_3']<=env_fus['wt_upperEnv'])&(EEG_fus['EEG_3']>=env_fus['wt_lowerEnv'])
env_fus['EEG_4_range']=(EEG_fus['EEG_4']<=env_fus['wt_upperEnv'])&(EEG_fus['EEG_4']>=env_fus['wt_lowerEnv'])
env_fus['EEG_5_range']=(EEG_fus['EEG_5']<=env_fus['wt_upperEnv'])&(EEG_fus['EEG_5']>=env_fus['wt_lowerEnv'])
env_fus['EEG_6_range']=(EEG_fus['EEG_6']<=env_fus['wt_upperEnv'])&(EEG_fus['EEG_6']>=env_fus['wt_lowerEnv'])
print("EEG FUS scanned.")
#endregion
#region Scan EEG signal outside of Hilbert envelope
env_fus=pd.DataFrame()
env_fus['wt_upperEnv']=EEG_wt['avEEG_WT'].rolling(window=windowsize).max().shift(int(-windowsize/2))
env_fus['EEG_1_range']=(EEG_fus['EEG_1']<=env_fus['wt_upperEnv'])
env_fus['EEG_2_range']=(EEG_fus['EEG_2']<=env_fus['wt_upperEnv'])
env_fus['EEG_3_range']=(EEG_fus['EEG_3']<=env_fus['wt_upperEnv'])
env_fus['EEG_4_range']=(EEG_fus['EEG_4']<=env_fus['wt_upperEnv'])
env_fus['EEG_5_range']=(EEG_fus['EEG_5']<=env_fus['wt_upperEnv'])
env_fus['EEG_6_range']=(EEG_fus['EEG_6']<=env_fus['wt_upperEnv'])
print("EEG FUS scanned.")
#endregion
#region Drop NaN
env_fus=env_fus.drop(range(0,9))
env_fus=env_fus.drop(range(8022,8032))
env_fus=env_fus.reset_index()
env_fus=env_fus.drop('index',axis=1)
#endregion
#region Values > threshold
count_fus1=(env_fus['EEG_1_range']==False).sum()
EEG_counts_fus1=env_fus.loc[env_fus['EEG_1_range']==False]
count_fus2=(env_fus['EEG_2_range']==False).sum()
EEG_counts_fus2=env_fus.loc[env_fus['EEG_2_range']==False]
count_fus3=(env_fus['EEG_3_range']==False).sum()
EEG_counts_fus3=env_fus.loc[env_fus['EEG_3_range']==False]
count_fus4=(env_fus['EEG_4_range']==False).sum()
EEG_counts_fus4=env_fus.loc[env_fus['EEG_4_range']==False]
count_fus5=(env_fus['EEG_5_range']==False).sum()
EEG_counts_fus5=env_fus.loc[env_fus['EEG_5_range']==False]
count_fus6=(env_fus['EEG_6_range']==False).sum()
EEG_counts_fus6=env_fus.loc[env_fus['EEG_6_range']==False]
EEG_counts_fus=count_fus1,count_fus2,count_fus3,count_fus4,count_fus5,count_fus6
EEG_counts_fus=pd.DataFrame(EEG_counts_fus)
EEG_counts_fus.columns=['EEG']
EEG_counts_fus['Mean']=EEG_counts_fus['EEG'].mean()
print("Counting FUS done.")
#endregion
print("FUS done.")
#endregion
