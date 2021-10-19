#!/usr/bin/env python3

# Usage: ./animate_xv.py path/to/run

import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np
# from pylab import *
import matplotlib
import sys
import os.path
from os.path import join as pjoin
from glob import glob
from io import StringIO
import re
from tqdm import tqdm
from os.path import join as pjoin
import argparse
from tasktimer import TaskTimer # pip install TaskTimer
from parsers import *
from scipy.constants import value as constants

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

files = glob(pjoin(folder, 'xv', '*'))

pat = re.compile('\d+')
files.sort(key=lambda x: int(pat.findall(x)[-1]))

files = files[args.f:args.t:args.s]


vars = parse_xoopic_input(pjoin(folder, 'input.inp'))
lx = vars['Grid'][0]['x1f']
dt = vars['Control'][0]['dt']
mI  = vars['Species'][1]['m']
mE = vars['Species'][0]['m']


######## Normalized Units ###############
ne = 1E13 #vars['Load'][0]['density']
ni = ne

eps0 = constants('electric constant')
kb = constants('Boltzmann constant')
e = constants('elementary charge')
gamma_e = 5./3

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

Te  = tEK #vars['tEK'] #1.6*11604
Ti  = tIK #vars['tIK'] #0.1*11604

dl	= np.sqrt(eps0*kb*Te/(ni*e*e))
dli	= np.sqrt(eps0*kb*Ti/(ni*e*e))

cia = np.sqrt(gamma_e*kb*Te/mI)

wpi = np.sqrt(e**2*ni/(eps0*mI))
wpe = np.sqrt(e**2*ne/(eps0*mE))

print("cia: %e"%cia)
print("vthe: %e"%vthE)

lx_norm = lx/dl

labels = ['Hydrogen','Electron']
nSpecies = len(labels)




timer = TaskTimer()

ymin = np.inf*np.ones(4)
ymax = -np.inf*np.ones(4)

force_load = True

if os.path.exists(pjoin(folder,'xv_data.npz')) and force_load:
    timer.task('Read file')
    xv_data = np.load(pjoin(folder,'xv_data.npz'), allow_pickle=True)
    # print(xv_data)
    xI = xv_data['xI']
    vI = xv_data['vI']
    xE = xv_data['xE']
    vE = xv_data['vE']
    time = xv_data['time']
    # print(xI[100].shape,vI.shape,time.shape)


else:
    xE = []
    vE = []
    xI = []
    vI = []
    time = []


    for file in timer.iterate(files):


        regex = re.compile(r'\d+')
        time.append([int(x) for x in regex.findall(file)][-1]*dt*wpi)
        # print(wpe,timeStamp)
        # exit()


        # This method reads files at half the time of np.loadtxt()
        with open(file, 'rb') as f:
            f.readline() # Skip first line
            data = np.array([line.strip().split() for line in f], float)
        # print(data)
        for i in range(nSpecies):

            timer.task('Process data')

            ind, = np.where(data[:,0]==i)

            ymin[i] = min(ymin[i], np.percentile(data[ind,3], 1))
            ymax[i] = max(ymax[i], np.percentile(data[ind,3], 99))

            # axs[i].cla()
            if i==0:
                xI.append(data[ind,2]/dl)
                # print((data[ind,2]/dl).shape)
                vI.append(data[ind,3]/cia)
                # axs[i].scatter(data[ind,2]/dl,data[ind,3]/cia,s=1,marker='.',color='b',alpha=0.6)
                # axs[i].set_ylabel("$v/C_s$")
                # axs[i].set_ylim([0.9*ymin[i]/cia, 1.1*ymax[i]/cia])
            else:
                xE.append(data[ind,2]/dl)
                vE.append(data[ind,3]/vthE)
                # axs[i].scatter(data[ind,2]/dl,data[ind,3]/vthE,s=1,marker='.',color='b',alpha=0.6)
                # axs[i].set_ylabel("$v/v_{th}$")
                # axs[i].set_ylim([0.9*ymin[i]/vthE, 1.1*ymax[i]/vthE])
            # axs[i].set_xlabel("$x/\lambda_D$")
            # axs[i].set_xlim([0,lx_norm])
            # axs[i].set_title(labels[i]+" (time = %f"%time+" $t\omega_{pi}$)")


    xI = np.array(xI)
    vI = np.array(vI)
    xE = np.array(xE)
    vE = np.array(vE)
    time = np.array(time)

    # print(xI.shape,vI.shape,time.shape)
    timer.task('Save data')
    np.savez_compressed(pjoin(folder,'xv_data.npz'),time = time, xI = xI, vI = vI, xE = xE, vE = vE)




##### FIG SIZE CALC ############
figsize = np.array([77,77*1.618]) #Figure size in mm
dpi = 300                         #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=True)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)


fig,axs = plt.subplots(4,1,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)

dt_norm = time[1]-time[0]
# timeslots = [0,7,12,46] #in t*omega_pi [for periodic]
timeslots = [100,200,300,400] #in t*omega_pi [for bounded]

# moviewriter = matplotlib.animation.FFMpegWriter(fps=10)
# with moviewriter.saving(fig, pjoin(folder, 'xv.mp4'), 100):

# for t in timeslots:
#     print(t)

for s in range(nSpecies):
    for i in timer.range(4):
        t = int(timeslots[i]/dt_norm)
        print(t)

        timer.task('Plot particles')

        axs[i].cla()
        if s == 0:
            axs[i].scatter(xI[t],vI[t],0.2,marker='.',color='b',alpha=1.0)
            axs[i].set_title(labels[0]+" (time = %d"%int(time[t])+" $t\omega_{pi}$)")
            axs[i].set_ylabel("$\\tilde{v_i}$")
            ymin[i] = min(ymin[i], np.percentile(vI[t], 1))
            ymax[i] = max(ymax[i], np.percentile(vI[t], 99))
            print(ymin[i],ymax[i])
            ymaxs = ymax[i]+abs(ymax[i]*0.5)
            ymins = ymin[i]-abs(ymaxs-ymax[i])
            print(ymins,ymaxs)
            axs[i].set_ylim([ymins,ymaxs])
        elif s==1:
            axs[i].scatter(xE[t],vE[t],0.2,marker='.',color='k',alpha=1.0)
            axs[i].set_ylabel("$\\tilde{v_e}$")
            ymin[i] = min(ymin[i], np.percentile(vE[t], 1))
            ymax[i] = max(ymax[i], np.percentile(vE[t], 99))
            # axs[i].set_ylim([ymin[i]*0.1,ymax[i]*1.5])
            axs[i].set_ylim([ymin[i]+ymin[i]*0.5,ymax[i]+ymax[i]*0.5])
        # axs[i].set_ylim([0.9*ymin[i], 1.1*ymax[i]])

        axs[i].set_xlabel("$\\tilde{x}$")
        axs[i].set_xlim([0,lx_norm])
        # print(ymin[i],ymax[i])

        # print(ymin[i]-ymin[i]*0.5,ymax[i]+ymin[i]*0.5)
        axs[i].set_title("$\\tau$ = %d"%int(time[t]))
        # axs[i].set_title(labels[s]+" (time = %d"%int(time[t])+" $t\omega_{pi}$)")


        timer.task('Save frame')

    plt.savefig(pjoin(folder, 'xv_multiplot_'+labels[s]+'.png'),dpi=dpi)






print(timer)
plt.show()
