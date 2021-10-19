#!/usr/bin/env python3

# ./energy_spectrum.py ../bounded_long_runs/1D_bounded_Tb_01/1D_bounded_Tb_01_L/

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mp
import matplotlib
import sys
import os.path
from os.path import join as pjoin
from glob import glob
from io import StringIO
import re
from tqdm import tqdm
import argparse
from tasktimer import TaskTimer
from parsers import *
from scipy.fft import fft, fftfreq
from scipy.signal import blackman

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str,
                    help='Folder to simulation runs')
parser.add_argument('-f', type=int, nargs='?', default=0,
                    help='From time step')
parser.add_argument('-t', type=int, nargs='?', default=-1,
                    help='To time step')
parser.add_argument('-s', type=int, nargs='?', default=1,
                    help='Time step step')
args = parser.parse_args()

folder = sys.argv[1]

vars = parse_xoopic_input(pjoin(folder, 'input.inp'))
lx = vars['Grid'][0]['x1f']
mI = vars['Species'][1]['m']
spwtI = vars['BeamEmitter'][1]['np2c']

mE = vars['Species'][0]['m']
dt = vars['Control'][0]['dt']
spwtE = vars['BeamEmitter'][0]['np2c']


# if ("double" in folder):
#     tstart = dt
#     tstop  = 1000000*dt
# else:

files = glob(pjoin(folder, 'xv', '*'))

pat = re.compile('\d+')
files.sort(key=lambda x: int(pat.findall(x)[-1]))

files = files[args.f:args.t:args.s]

labels = ['Hydrogen','Electron']
nSpecies = len(labels)

timer = TaskTimer()

ymin = np.inf*np.ones(nSpecies)
ymax = -np.inf*np.ones(nSpecies)

if os.path.exists(pjoin(folder,'KE_data.npz')):
    KEdata = np.load(pjoin(folder,'KE_data.npz'))
    KE_I_total = KEdata['KE_I_total']
    KE_E_total = KEdata['KE_E_total']
    KE_I_avg = KEdata['KE_I_avg']
    KE_E_avg = KEdata['KE_E_avg']
else:
    raise Exception('-KE_data.npz- file not found. Run ke_species.py first.')

tstart = dt
tstop  = len(KE_I_total)*dt
time = np.linspace(tstart,tstop,len(KE_I_total))

############### FFT ################
N = len(KE_I_total)

w = blackman(N)
yfE = fft(KE_E_avg,norm='ortho')
yfI = fft(KE_I_avg,norm='ortho')
ywfE = fft(KE_E_avg*w,norm='ortho')
ywfI = fft(KE_I_avg*w,norm='ortho')

xf = fftfreq(N,dt*200)[:N//2]

yffE = 2.0 * np.abs(yfE[0:N//2])
yffI = 2.0 * np.abs(yfI[0:N//2])

ywffE = 2.0 * np.abs(ywfE[0:N//2])
ywffI = 2.0 * np.abs(ywfI[0:N//2])


##### FIG SIZE CALC ############
figsize = np.array([150,150/1.618]) #Figure size in mm
dpi = 300                         #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=True)
mp.rc('font', family='sans-serif', size=14, serif='Computer Modern Roman')
mp.rc('axes', titlesize=14)
mp.rc('axes', labelsize=14)
mp.rc('xtick', labelsize=14)
mp.rc('ytick', labelsize=14)
mp.rc('legend', fontsize=14)

fig,(ax1,ax2) = plt.subplots(2,1,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
# div = mp.make_axes_locatable(ax)
# cax = div.append_axes('right', '4%', '4%')
ax1.plot(xf,yffI, '-b', lw=1, label = "FFT")
# ax1.plot(xf,ywffI, '-r', lw=1,label = "FFT-blackman")

ax2.plot(xf,yffE, '-b', lw=1,label = "FFT")
# ax2.plot(xf,ywffE, '-r', lw=1,label = "FFT-blackman")

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("PSD (Ion)")
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("PSD (Electron)")
ax1.legend()
ax2.legend()
plt.savefig(pjoin(folder, 'PSD_sys.png'),dpi=dpi)


print(timer)
plt.show()
