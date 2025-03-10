#!/usr/bin/env python3

# Usage: ./animate_xv.py path/to/run

import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np
# from pylab import *
import matplotlib
import sys
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


##### FIG SIZE CALC ############
figsize = np.array([200,200/1.618]) #Figure size in mm
dpi = 300                         #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=14, serif='Computer Modern Roman')
mp.rc('axes', titlesize=14)
mp.rc('axes', labelsize=14)
mp.rc('xtick', labelsize=14)
mp.rc('ytick', labelsize=14)
mp.rc('legend', fontsize=14)


fig,axs = plt.subplots(nSpecies,1,figsize=figsize/25.4,dpi=ppi)

timer = TaskTimer()

ymin = np.inf*np.ones(nSpecies)
ymax = -np.inf*np.ones(nSpecies)

moviewriter = matplotlib.animation.FFMpegWriter(fps=10)
with moviewriter.saving(fig, pjoin(folder, 'xv.mp4'), 100):

    for file in timer.iterate(files):

        timer.task('Read file')

        regex = re.compile(r'\d+')
        timeStamp = [int(x) for x in regex.findall(file)][-1]*dt*wpi
        print(file)
        # exit()


        # This method reads files at half the time of np.loadtxt()
        with open(file, 'rb') as f:
            f.readline() # Skip first line
            data = np.array([line.strip().split() for line in f], float)
        # print(data)
        for i in range(nSpecies):

            timer.task('Plot particles')

            ind, = np.where(data[:,0]==i)

            ymin[i] = min(ymin[i], np.percentile(data[ind,3], 1))
            ymax[i] = max(ymax[i], np.percentile(data[ind,3], 99))

            axs[i].cla()
            if i==0:
                axs[i].scatter(data[ind,2]/dl,data[ind,3]/cia,s=1,marker='.',color='b',alpha=0.6)
                axs[i].set_ylabel("$v/C_s$")
                axs[i].set_ylim([0.9*ymin[i]/cia, 1.1*ymax[i]/cia])
            else:
                axs[i].scatter(data[ind,2]/dl,data[ind,3]/vthE,s=1,marker='.',color='b',alpha=0.6)
                axs[i].set_ylabel("$v/v_{th}$")
                axs[i].set_ylim([0.9*ymin[i]/vthE, 1.1*ymax[i]/vthE])
            axs[i].set_xlabel("$x/\lambda_D$")
            axs[i].set_xlim([0,lx_norm])
            axs[i].set_title(labels[i]+" (time = %f"%timeStamp+" $t\omega_{pi}$)")

        timer.task('Save frame')

        plt.tight_layout()
        moviewriter.grab_frame()

print(timer)
