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

TE = []
KE = []
Efield = []
Bfield = []

lenvar = ['_L_half','_L','_L_double']

for l in range(len(lenvar)):
    runName= os.path.basename(os.path.dirname(folder))+lenvar[l]
    vars = parse_xoopic_input(pjoin(folder,runName,'input.inp'))
    lx = vars['Grid'][0]['x1f']
    mI = vars['Species'][1]['m']
    spwtI = vars['BeamEmitter'][1]['np2c']

    mE = vars['Species'][0]['m']
    dt = vars['Control'][0]['dt']
    spwtE = vars['BeamEmitter'][0]['np2c']

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

    s_index,_,timeAll, dataAll = np.loadtxt(pjoin(folder,runName,'energy',fileName),unpack=True)

    data = []
    for i in range(nparams):
        ind, = np.where(s_index==i)
        time = timeAll[ind]*wpi
        data.append(dataAll[ind])
        # Bfield,Efield,KE,TE = np.array_split(dataAll,4)

    data = np.array(data)
    TE.append(data[0,:])
    KE.append(data[1,:])
    Efield.append(data[2,:])
    Bfield.append(data[3,:])
    # TE,KE,Efield,Bfield = data[0,:],data[1,:],data[2,:],data[3,:]

TE = np.array(TE)
KE = np.array(KE)
Efield = np.array(Efield)
Bfield = np.array(Bfield)

# print(TE.shape)
# exit()
######################################
##### FIG SIZE CALC ############
figsize = np.array([100,100/1.618]) #Figure size in mm
dpi = 300                         #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution
#########################



mp.rc('text', usetex=True)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)

fig,axs = plt.subplots(2,1,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)

# ax1.plot(time[50:], TE[50:],lw=1.5, label='TE')
axs[0].plot(time[50:], np.log(np.sqrt(KE[0,50:]/rest_mass_energy_e)),lw=1.5, label='$\\tilde{x} = 512$')
axs[0].plot(time[50:], np.log(np.sqrt(KE[1,50:]/rest_mass_energy_e)),lw=1.5, label='$\\tilde{x} = 1024$')
axs[0].plot(time[50:], np.log(np.sqrt(KE[2,50:]/rest_mass_energy_e)),lw=1.5, label='$\\tilde{x} = 2048$')

axs[1].plot(time[50:], np.log(np.sqrt(Efield[0,50:]/rest_mass_energy_e)),lw=1.5, label='$\\tilde{x} = 512$')
axs[1].plot(time[50:], np.log(np.sqrt(Efield[1,50:]/rest_mass_energy_e)),lw=1.5, label='$\\tilde{x} = 1024$')
axs[1].plot(time[50:], np.log(np.sqrt(Efield[2,50:]/rest_mass_energy_e)),lw=1.5, label='$\\tilde{x} = 2048$')
# ax.set_xscale('log')
# ax1.set_yscale('log')
# ax2.set_yscale('log')
# ax.semilogy(time, Bfield,lw=1.0, label='Bfield')
axs[0].set_xlim([time[50],time[-1]])
axs[0].set_xlim([time[50],time[-1]])
axs[0].set_xlabel("$\\tau$")
axs[0].set_ylabel('$\ln \sqrt{\epsilon_{tot}/ m_ec^2}$')
axs[0].legend();
axs[1].set_xlabel("$\\tau$")
axs[1].set_ylabel('$\ln \sqrt{\epsilon_{E}/ m_ec^2}$')
axs[1].legend();
# box = axs[1].get_position()
# axs[1].set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.71])
# axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.53),fancybox=True, shadow=False, ncol=3)

# ax.set_ylabel('$\phi\,[\mathrm{V}]$')

plt.savefig(pjoin(folder,'energy_all_%d'%timestep+'.png'),dpi=dpi)
plt.show() #After savefig
