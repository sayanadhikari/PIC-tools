#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mp
import argparse

import sys
import os.path
from os.path import join as pjoin

from parsers import *
from scipy.constants import value as constants
from scipy.constants import speed_of_light as c0

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str, help='Folder to simulation runs')
parser.add_argument('-t', type=int, nargs='?', default=-1, help='time step')

args = parser.parse_args()

folder = sys.argv[1]
timestep = args.t

vars = parse_xoopic_input(pjoin(folder, 'input.inp'))
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

labels = ['TE','KE','Efield','Bfield']
nparams = len(labels)

fileName = "energy_%08d"%timestep+".dat"


s_index,_,timeAll, dataAll = np.loadtxt(pjoin(folder,'energy',fileName),unpack=True)

data = []
for i in range(nparams):
    ind, = np.where(s_index==i)
    time = timeAll[ind]*wpi
    data.append(dataAll[ind])
    # Bfield,Efield,KE,TE = np.array_split(dataAll,4)

data = np.array(data)
TE,KE,Efield,Bfield = data[0,:],data[1,:],data[2,:],data[3,:]

######################################
##### FIG SIZE CALC ############
figsize = np.array([150,150/1.618]) #Figure size in mm
dpi = 300                         #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution
#########################



mp.rc('text', usetex=True)
mp.rc('font', family='sans-serif', size=14, serif='Computer Modern Roman')
mp.rc('axes', titlesize=14)
mp.rc('axes', labelsize=14)
mp.rc('xtick', labelsize=14)
mp.rc('ytick', labelsize=14)
mp.rc('legend', fontsize=14)

fig,(ax1,ax2) = plt.subplots(2,1,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)

# ax1.plot(time[50:], TE[50:],lw=1.5, label='TE')
ax1.plot(time[50:], np.log(np.sqrt(KE[50:]/rest_mass_energy_e)),lw=1.5, label='KE')
ax2.plot(time[50:], np.log(np.sqrt(Efield[50:]/rest_mass_energy_e)),lw=1.5, label='Efield')
# ax.set_xscale('log')
# ax1.set_yscale('log')
# ax2.set_yscale('log')
# ax.semilogy(time, Bfield,lw=1.0, label='Bfield')
# ax.set_xlim([3.3633682027497936e-05,5.6056136712496555e-05])
# ax.set_ylim([0.6e-9,0.1e-8])
ax1.set_xlim([time[50],time[-1]])
ax2.set_xlim([time[50],time[-1]])

ax1.set_xlabel("$t \omega_{pi}$")
ax1.set_ylabel('$\ln \sqrt{\epsilon_{tot}/ m_ec^2}$')
# ax1.legend();
ax2.set_xlabel("$t \omega_{pi}$")
ax2.set_ylabel('$\ln \sqrt{\epsilon_{E}/ m_ec^2}$')
# ax2.legend();

# ax.set_ylabel('$\phi\,[\mathrm{V}]$')

plt.savefig(pjoin(folder,'energy_all_%d'%timestep+'.png'),dpi=dpi)
plt.show() #After savefig
