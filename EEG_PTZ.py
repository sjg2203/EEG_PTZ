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
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import PySimpleGUI as sg
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
from tensorpac import Pac
from yasa import transition_matrix
from scipy.integrate import simpson
from tkinter import filedialog as fd
from scipy.signal import welch,hilbert
from sklearn.metrics import accuracy_score

import mne
import numpy as np
import adi

#Setting the MNE info for the ecog data
sampling_freq=1000
ch_names=['C1','C2','C3']
ch_types=['ecog']*3
info=mne.create_info(ch_names,ch_types=ch_types,sfreq=sampling_freq)
info['description']='My custom dataset'
info.set_montage('standard_1020')

#Converting the ecog data to NumPy array from .adicht file using SDK
input_fname=fd.askopenfilename(title='SELECT EDF FILE',filetypes=(("ADICHT files","*.adicht"),("all files","*.*")))
f=adi.read_file(input_fname)
ecog1=f.channels[0].get_data(1)
ecog2=f.channels[1].get_data(1)
ecog3=f.channels[2].get_data(1)

#Scaling the values appropriately to be interpreted by the CSP (SDK always returns incorrect units)
ecog1=ecog1*(10000)
ecog2=ecog2*(10000)
ecog3=ecog3*(10000)

#Creating an array of (number_of_channels, number_of_samples) which looks like (2, 286749)
EcogData = np.array([ecog1,ecog2,ecog3])
#Creating MNE data
raw=mne.io.RawArray(EcogData,info)

#Specifying the onset times and duration of each trial, and labelling them with strings and classification codes (left, right) (1, -1)
event_on=np.array([69,80,91,104,117,129,142,153,166,178,191,202,214,227,240,253,264,276,287,299,310,322,335,347,358,371,382,395,409,420,431,444,456,468,479,492,503,516,528,540])
duration=[6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],
