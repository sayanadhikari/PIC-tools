#!/usr/bin/env python3

# ./ke_species.py ../bounded_long_runs/1D_bounded_Tb_01/1D_bounded_Tb_01_L/

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
from scipy.constants import value as constants
from scipy.constants import speed_of_light as c0

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

KE_I_avg = []
KE_E_avg = []
time = []

lenvar = ['_L_half','_L','_L_double']

for l in range(len(lenvar)):
    runName= os.path.basename(os.path.dirname(folder))+lenvar[l]
    vars = parse_xoopic_input(pjoin(folder,runName,'input.inp'))
    lx = vars['Grid'][0]['x1f']
    mI = vars['Species'][1]['m']
    if 'BeamEmitter' in vars:
        spwtI = vars['BeamEmitter'][1]['np2c']
    elif 'Load' in vars:
        spwtI = vars['Load'][1]['np2c']

    mE = vars['Species'][0]['m']
    dt = vars['Control'][0]['dt']
    if 'BeamEmitter' in vars:
        spwtE = vars['BeamEmitter'][0]['np2c']
    elif 'Load' in vars:
        spwtE = vars['Load'][0]['np2c']

    ne = 1E13 #vars['Load'][0]['density']
    ni = ne

    eps0 = constants('electric constant')
    kb = constants('Boltzmann constant')
    me = constants('electron mass')
    e = constants('elementary charge')
    # c0 = constants('speed_of_light')

    wpi = np.sqrt(e**2*ni/(eps0*mI))
    wpe = np.sqrt(e**2*ne/(eps0*mE))

    rest_mass_energy_e = mE*c0*c0
    rest_mass_energy_I = mI*c0*c0
    # print("Rest mass energy: %e"%rest_mass_energy_e)
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

    force_load = True

    # if os.path.exists(pjoin(folder,runName,'KE_data.npz')) and force_load:
    KEdata = np.load(pjoin(folder,runName,'KE_data.npz'))
    # KE_I_total = KEdata['KE_I_total']
    # KE_E_total = KEdata['KE_E_total']
    KE_I_avg.append(KEdata['KE_I_avg'])
    KE_E_avg.append(KEdata['KE_E_avg'])
    time.append(KEdata['time'])

KE_I_avg = np.array(KE_I_avg)
KE_E_avg = np.array(KE_E_avg)
time = np.array(time)

# print(KE_I_avg[0].shape,KE_I_avg[1].shape,KE_I_avg[2].shape)
##### FIG SIZE CALC ############
figsize = np.array([100,100/1.618]) #Figure size in mm
dpi = 300                         #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=True)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)

fig,(ax1,ax2) = plt.subplots(2,1,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)

# ax1.plot(time,np.log(KE_I_total),lw=0.5, label = "$KE_I$")
# ax2.plot(time,np.log(KE_E_total),lw=0.5, label = "$KE_e$")
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# ax1.set_xlabel("$t \omega_{pi}$")
# ax1.set_ylabel('$\ln \sqrt{\epsilon_i / m_ic^2}$')
# # ax1.set_ylabel('$\Epsilon\\rho(x)\,[\mathrm{m}]$')
# ax2.set_xlabel("$t \omega_{pi}$")
# ax2.set_ylabel('$\ln \sqrt{\epsilon_e  / m_ec^2}$')
# plt.savefig(pjoin(folder, 'KE_total.png'),dpi=dpi)
# ax1.clear()
# ax2.clear()
#
# print(KE_I_avg[0])
ax1.plot(time[0],(KE_I_avg[0]),lw=0.5, label='$\\tilde{x} = 512$')
ax1.plot(time[1],(KE_I_avg[1]),lw=0.5, label='$\\tilde{x} = 1024$')
ax1.plot(time[2],(KE_I_avg[2]),lw=0.5, label='$\\tilde{x} = 2048$')


ax2.plot(time[0],(KE_E_avg[0]),lw=0.5, label='$\\tilde{x} = 512$')
ax2.plot(time[1],(KE_E_avg[1]),lw=0.5, label='$\\tilde{x} = 1024$')
ax2.plot(time[2],(KE_E_avg[2]),lw=0.5, label='$\\tilde{x} = 2048$')

ax1.set_xlim([time[0][1000],time[0][-1]])
ax2.set_xlim([time[0][1000],time[0][-1]])

ax1.set_xlabel("$\\tau$")
ax1.set_ylabel('$\\varepsilon_i / \\varepsilon_{i}^{th}$')
leg = ax1.legend(loc='upper right')
# ax1.set_ylabel('$\Epsilon\\rho(x)\,[\mathrm{m}]$')
ax2.set_xlabel("$\\tau$")
ax2.set_ylabel('$\\varepsilon_e  / \\varepsilon_{e}^{th}$')
leg = ax2.legend(loc='upper right')
plt.savefig(pjoin(folder, 'KE_avg_all.png'),dpi=dpi)


print(timer)
plt.show()
