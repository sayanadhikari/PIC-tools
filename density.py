#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np
from pylab import *
import cmath
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import sys
import os.path
from os.path import join as pjoin
from scipy.constants import value as constants
from glob import glob
import re
from tqdm import tqdm
import argparse
from tasktimer import TaskTimer
from parsers import *
from mpl_toolkits import mplot3d

parser = argparse.ArgumentParser(description='Plasma Dispersion Processor')
parser.add_argument('folder', type=str,help='Folder to simulation runs')
parser.add_argument('-t','--timeStep',default=10,type=int,help='timestep of the data, e.g. 1000')
parser.add_argument('-a','--avgLen',default=100,type=int,help='average over no of data, e.g. 100')
args        = parser.parse_args()
avgLen      = args.avgLen

folder = sys.argv[1]

vars = parse_xoopic_input(pjoin(folder, 'input.inp'))
nTStop = args.timeStep
Nx = vars['Grid'][0]['J']+1
Ny = vars['Grid'][0]['K']+1
Lx = vars['Grid'][0]['x1f']

# runName = "00200000"
# DIR="../data/phi/"
# numerical data file
def denAvg(param,nTStop,avgLen):
    nT = np.linspace((nTStop+1)-avgLen,nTStop,avgLen,dtype='int')
    dataDen = np.zeros(Nx*Ny)
    for i in nT:
        fileName=pjoin(folder,'density',param+'_%d'%i+'.txt')
        datax, datay, dataDenTemp = np.loadtxt(fileName, unpack=True)
        dataDen += dataDenTemp
    # dataDen = np.array(dataDen)
    dataDen /= avgLen
    X = np.reshape(datax, (Nx, Ny))
    Y = np.reshape(datay, (Nx, Ny))
    Den = np.reshape(dataDen,(Nx, Ny))
    return X,Y,Den


X,Y,denE = denAvg('denE',nTStop,avgLen)
X,Y,denH = denAvg('denH',nTStop,avgLen)

# Phi = E/np.max(E)
# print(Y[0,:])
###########################
###### 1D Density #########
m = int(Ny/2)+1
denE1D = denE[:,m]
denH1D = denH[:,m]

###################################################
#############Radial Basis Funtion###############
#### https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
################################################
xi = np.linspace(0,Lx,num=102400,endpoint=True)

rbfdenE1D = Rbf(X[:len(denE1D),0], denE1D)
fitdenE1D = rbfdenE1D(xi)

rbfdenH1D = Rbf(X[:len(denH1D),0], denH1D)
fitdenH1D = rbfdenH1D(xi)


######################################
##### FIG SIZE CALC ############
figsize = np.array([150,150/1.618]) #Figure size in mm
dpi = 300                         #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution
###################################

mp.rc('text', usetex=True)
mp.rc('font', family='sans-serif', size=14, serif='Computer Modern Roman')
mp.rc('axes', titlesize=14)
mp.rc('axes', labelsize=14)
mp.rc('xtick', labelsize=14)
mp.rc('ytick', labelsize=14)
mp.rc('legend', fontsize=14)

fig1,ax1 = plt.subplots(figsize=(figsize*1.5)/25.4,constrained_layout=True,dpi=ppi)

# ax1.plot(X[:len(denE1D),0], denE1D,label='$N_e$')
# ax1.plot(X[:len(denH1D),0], denH1D,label='$N_i$')
ax1.plot(xi, fitdenE1D,label='$N_e$')
ax1.plot(xi, fitdenH1D,label='$N_i$')


# ax1.set_xscale('log')
ax1.set_xlim([0.1, Lx])
ax1.set_xlabel('$x\,[\mathrm{m}]$')
ax1.set_ylabel('$N\,[\mathrm{m^{-3}}]$')
leg = ax1.legend();
# ax.set_ylabel('$\phi\,[\mathrm{V}]$')
plt.savefig(pjoin(folder,'density_line_%d'%nTStop+'.png'),dpi=dpi)




mp.rc('text', usetex=True)
mp.rc('font', family='sans-serif', size=14, serif='Computer Modern Roman')
mp.rc('axes', titlesize=14)
mp.rc('axes', labelsize=14)
mp.rc('xtick', labelsize=14)
mp.rc('ytick', labelsize=14)
mp.rc('legend', fontsize=14)

fig2,ax2 = plt.subplots(figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
cont = plt.contourf(X,Y,denE,100)
# ax2.plot(xi, fi, 'r', lw=0.5, label='$\phi_{RBF}$')
ax2.set_xlabel('$x\,[\mathrm{m}]$')
ax2.set_ylabel('$y\,[\mathrm{m}]$')
plt.colorbar(cont)
# leg = ax2.legend();
# ax.set_ylabel('$\phi\,[\mathrm{V}]$')
plt.savefig(pjoin(folder,'density_contour_%d'%nTStop+'.png'),dpi=dpi)


# fig3 = plt.figure(figsize=(figsize*2)/25.4,constrained_layout=True,dpi=ppi)
# ax3 = plt.axes(projection='3d')
#
# mp.rc('text', usetex=True)
# mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
# mp.rc('axes', titlesize=10)
# mp.rc('axes', labelsize=10)
# mp.rc('xtick', labelsize=10)
# mp.rc('ytick', labelsize=10)
# mp.rc('legend', fontsize=10)
#
# surf = ax3.plot_surface(X,Y,denE,cmap='viridis',edgecolor='none')
# # ax2.plot(xi, fi, 'r', lw=0.5, label='$\phi_{RBF}$')
# ax3.set_xlabel('$x\,[\mathrm{m}]$')
# ax3.set_ylabel('$y\,[\mathrm{m}]$')
# plt.colorbar(surf)
# # leg = ax2.legend();
# # ax.set_ylabel('$\phi\,[\mathrm{V}]$')
# plt.savefig(pjoin('figures','density_surface_%d'%nTStop+'.png'),dpi=dpi)



plt.show()
