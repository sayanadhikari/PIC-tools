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

vars = parse_xoopic_input(pjoin(folder, 'input.inp'))
lx = vars['Grid'][0]['x1f']
mI = vars['Species'][1]['m']


ne = 1E13 #vars['Load'][0]['density']
ni = ne

eps0 = constants('electric constant')
kb = constants('Boltzmann constant')
# me = constants('electron mass')
e = constants('elementary charge')


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

if 'BeamEmitter' in vars:
    units_0 = vars['BeamEmitter'][0]['units']
    if units_0 == 'MKS':
        vthE = vars['BeamEmitter'][0]['temperature']
        tEeV   = 0.5*mE*(vthE*vthE)/e
        tEK    = tEeV*11604.525

    units_1 = vars['BeamEmitter'][1]['units']
    if units_1 == 'MKS':
        vthI = vars['BeamEmitter'][1]['temperature']
        tIeV   = 0.5*mI*(vthI*vthI)/e
        tIK    = tIeV*11604.525
        vb  = vars['BeamEmitter'][1]['v1drift']
        tIbeV   = 0.5*mI*(vb*vb)/e
        tIbK    = tIbeV*11604.525
if 'Load' in vars:
    units_0 = vars['Load'][0]['units']
    if units_0 == 'MKS':
        vthE = vars['Load'][0]['temperature']
        tEeV   = 0.5*mE*(vthE*vthE)/e
        tEK    = tEeV*11604.525

    units_1 = vars['Load'][1]['units']
    if units_1 == 'MKS':
        vthI = vars['Load'][1]['temperature']
        tIeV   = 0.5*mI*(vthI*vthI)/e
        tIK    = tIeV*11604.525
        vb  = vars['Load'][1]['v1drift']
        tIbeV   = 0.5*mI*(vb*vb)/e
        tIbK    = tIbeV*11604.525

# c0 = constants('speed_of_light')

wpi = np.sqrt(e**2*ni/(eps0*mI))
wpe = np.sqrt(e**2*ne/(eps0*mE))

rest_mass_energy_e = mE*c0*c0
rest_mass_energy_I = mI*c0*c0
print(vthE,vthI)
Eth_e = 0.5*mE*vthE*vthE
Eth_I = 0.5*mI*vthI*vthI
print('Thermal Energy: ',Eth_e, Eth_I)
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

if os.path.exists(pjoin(folder,'KE_data.npz')) and force_load:
    KEdata = np.load(pjoin(folder,'KE_data.npz'))
    KE_I_total = KEdata['KE_I_total']
    KE_E_total = KEdata['KE_E_total']
    KE_I_avg = KEdata['KE_I_avg']
    KE_E_avg = KEdata['KE_E_avg']
    time = KEdata['time']
else:
    KE_I_total = []
    KE_E_total = []
    KE_I_avg = []
    KE_E_avg = []
    time = []

    for file in timer.iterate(files):

        timer.task('Read file')

        regex = re.compile(r'\d+')
        timeStamp = [int(x) for x in regex.findall(file)][-1]
        time.append(timeStamp*dt*wpi)

        # This method reads files at half the time of np.loadtxt()
        with open(file, 'rb') as f:
            f.readline() # Skip first line
            data = np.array([line.strip().split() for line in f], float)
        # print(data)
        for i in range(nSpecies):

            timer.task('Calculate KE')

            ind, = np.where(data[:,0]==i)

            ymin[i] = min(ymin[i], np.percentile(data[ind,3], 1))
            ymax[i] = max(ymax[i], np.percentile(data[ind,3], 99))

            if i == 0:
                KE_I_total.append(np.sum((0.5*mI*data[ind,3]**2)/Eth_I))
                # KE_I_total.append(np.sqrt(np.sum((0.5*mI*data[ind,3]**2)/Eth_I)))
                KE_I_avg.append(np.average(0.5*mI*data[ind,3]**2)/Eth_I)
                # print(np.average(0.5*mI*data[ind,3]**2)/Eth_I,Eth_I)
            else:
                KE_E_total.append(np.sum((0.5*mE*data[ind,3]**2)/Eth_e))
                # KE_E_total.append(np.sqrt(np.sum((0.5*mE*data[ind,3]**2)/Eth_e)))
                KE_E_avg.append(np.average(0.5*mE*data[ind,3]**2)/Eth_e)
                # print(np.average(0.5*mE*data[ind,3]**2)/Eth_e,Eth_e)

            # axs[i].cla()
            # axs[i].scatter(data[ind,2],data[ind,3],s=1,marker='.',color='b',alpha=0.6)
            # axs[i].set_xlabel("$x [m]$")
            # axs[i].set_ylabel("$v [m/s]$")
            # axs[i].set_xlim([0,lx])
            # axs[i].set_ylim([0.9*ymin[i], 1.1*ymax[i]])
            # axs[i].set_title(labels[i])


    KE_I_total = np.array(KE_I_total)*spwtI
    KE_E_total = np.array(KE_E_total)*spwtE
    KE_I_avg = np.array(KE_I_avg)
    KE_E_avg = np.array(KE_E_avg)
    time = np.array(time)

    np.savez_compressed(pjoin(folder,'KE_data.npz'),time = time, KE_I_total=KE_I_total,KE_I_avg=KE_I_avg,KE_E_total=KE_E_total,KE_E_avg=KE_E_avg)

# tstart = dt
# tstop  = len(KE_I_total)*dt
# print(tstart, tstop, wpe)
# time = np.linspace(tstart,tstop,len(KE_I_total))/(1/wpe)

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
ax1.plot(time,(KE_I_total),lw=1.0, label = "$KE_I$")
ax2.plot(time,(KE_E_total),lw=1.0, label = "$KE_e$")
# ax1.set_yscale('log')
# ax2.set_yscale('log')
ax1.set_xlabel("$t \omega_{pi}$")
ax1.set_ylabel('$\ln \sqrt{\epsilon_i / m_ic^2}$')
# ax1.set_ylabel('$\Epsilon\\rho(x)\,[\mathrm{m}]$')
ax2.set_xlabel("$t \omega_{pi}$")
ax2.set_ylabel('$\ln \sqrt{\epsilon_e  / m_ec^2}$')

# ax1.set_xlim([time[1000],time[-1]])
ax1.set_xlim([time[0],time[-1]])
ax2.set_xlim([time[0],time[-1]])

plt.savefig(pjoin(folder, 'KE_total.png'),dpi=dpi)
ax1.clear()
ax2.clear()
#
ax1.plot(time,(KE_I_avg),'k',lw=1.0, label = "$KE_I$")
ax2.plot(time,(KE_E_avg),'k',lw=1.0, label = "$KE_e$")

ax1.set_xlim([time[0],time[-1]])
ax2.set_xlim([time[0],time[-1]])
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
ax1.set_xlabel("$\\tau$")
ax1.set_ylabel('$\\varepsilon_i / \\varepsilon_{i}^{th}$')
# ax1.set_ylabel('$\ln \sqrt{\epsilon_i / m_ic^2}$')
ax2.set_xlabel("$\\tau$")
ax2.set_ylabel('$\\varepsilon_e / \\varepsilon_{e}^{th}$')
# ax2.set_ylabel('$\ln \sqrt{\epsilon_e  / m_ec^2}$')
plt.savefig(pjoin(folder, 'KE_avg.png'),dpi=dpi)


print(timer)
plt.show()
