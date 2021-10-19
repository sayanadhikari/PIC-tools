#!/usr/bin/env python3
# Usage: ./disprel.py path/to/folder

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mp
import numpy as np
import sys
import os.path
from os.path import join as pjoin
from scipy.constants import value as constants
from glob import glob
import re
from tqdm import tqdm
from parsers import *
import argparse



parser = argparse.ArgumentParser(description='Plasma Dispersion Processor')
parser.add_argument('-n','--norm', default='omega_pe', type=str, help='Normalizing frequency, Options: omega_pi, omega_pe')
parser.add_argument('-d','--dir', default='.', type=str, help='Save directory, e.g.: figures')

args        = parser.parse_args()
norm        = args.norm
dir         = args.dir

savedir = pjoin(dir)
# Analytical ion-acoustic dispresion relation
ne = 1E13 #vars['Load'][0]['density']
ni = ne

eps0 = constants('electric constant')
kb = constants('Boltzmann constant')
me = constants('electron mass')
e = constants('elementary charge')
c0 = constants('speed of light in vacuum')

mi  = 1*constants('atomic mass constant')
gamma_e = 5./3



tEeV  = 0.5                   #  [eV]
tEK   = tEeV*11604.525        #  [K]
tIeV  = 0.01                  #  [eV]
tIK   = tIeV*11604.525        #  [K]
tbeV = np.array([0.1, 0.5, 1.0, 7.5])       #  [eV]
# tbeV = np.array([0.1])       #  [eV]
tIbK = tbeV*11604.525        #  [K]
vb   = np.sqrt(2*(tbeV*e)/mi)  #[m/s]

Te  = tEK #vars['tEK'] #1.6*11604
Ti  = tIK #vars['tIK'] #0.1*11604


wpi = np.sqrt(e**2*ni/(eps0*mi))
wpe = np.sqrt(e**2*ne/(eps0*me))
dl	= np.sqrt(eps0*kb*Te/(ni*e*e))
dli	= np.sqrt(eps0*kb*Ti/(ni*e*e))
print("wpi={}".format(wpi))
print("dl={}".format(dl))
cia = np.sqrt(gamma_e*kb*Te/mi)

Nx = 1024
k     = 2*np.pi*np.arange(Nx)/(Nx*dl)

ka = np.linspace(0, np.max(k), Nx)
wac = np.sqrt((ka*cia)**2/(1+(ka*cia/wpi)**2))
wah = np.sqrt( (wpi**2) * (ka*ka * dli*dli * (Te/Ti))/(1+(ka*ka * dli*dli * (Te/Ti))) )
wl = np.sqrt( (wpe**2) * (1+me/mi) * (1+(3*ka*ka*dl*dl)/(1+me/mi)) )
# wb = ka*tIbK

kadl = ka*dl
kadl[0] = kadl[1]

print('vb/cia = ',vb/cia)

def periodic_dispersion(vb):
    # beam_density ratio
    alpha = 1
    # Coefficients
    coeff1 = ( 1 + ( 1 / (kadl*kadl) ) )
    coeff2 = -2*kadl*(vb/dl)*coeff1
    coeff3 = ( (kadl*kadl) * (1/(dl*dl)) * (vb*vb) * coeff1) - alpha*wpi*wpi
    roots = []
    for i in range(1,len(kadl)):
      # coeffs = [coeff1, coeff2[i], coeff3[i]]
      coeffs = [coeff1[i], coeff2[i], coeff3[i]]
      root = np.roots(coeffs)
      roots.append(root)
    roots = np.array(roots)/wpi
    return roots

def periodic_dispersion_norm(vb):
    vb = vb/cia
    # Coefficients
    coeff1 = ( 1 + ( 1 / (kadl*kadl) ) )
    coeff2 = -2 * kadl * vb * coeff1
    coeff3 = ((kadl*kadl) * (vb*vb) * coeff1) - 1
    roots = []
    for i in range(1,len(kadl)):
      # coeffs = [coeff1, coeff2[i], coeff3[i]]
      coeffs = [coeff1[i], coeff2[i], coeff3[i]]
      root = np.roots(coeffs)
      roots.append(root)
    roots = np.array(roots)
    return roots

def periodic_dispersion_ion(vb):
    # beam_density ratio
    alpha = 0.1
    # Coefficients
    coeff1 = ( 1 + ( 1 / (kadl*kadl) ) )
    coeff2 = -2*kadl*(vb/dl)*coeff1
    coeff3 = ( (kadl*kadl) * (1/(dl*dl)) * (vb*vb) * coeff1 ) - (1-alpha)*wpi*wpi
    coeff4 = 2*wpi*wpi*kadl*vb*(1/dl)
    coeff5 = - wpi*wpi * kadl*kadl * vb*vb * (1/(dl*dl))
    roots = []
    for i in range(1,len(kadl)):
      # coeffs = [coeff1, coeff2[i], coeff3[i]]
      coeffs = [coeff1[i], coeff2[i], coeff3[i], coeff4[i], coeff5[i]]
      root = np.roots(coeffs)
      roots.append(root)
    roots = np.array(roots)
    return roots

def periodic_dispersion_ion_norm(vb):
    vb = vb/cia
    # Coefficients
    coeff1 = (kadl*kadl) + 1
    coeff2 = - 2 * vb * coeff1
    coeff3 = (kadl*kadl*vb*vb) - (2*kadl*kadl) + (vb*vb)
    coeff4 = 2*kadl*kadl*vb
    coeff5 = - (kadl*kadl) * (vb*vb)
    roots = []
    for i in range(1,len(kadl)):
      # coeffs = [coeff1, coeff2[i], coeff3[i]]
      coeffs = [coeff1[i], coeff2[i], coeff3[i], coeff4[i], coeff5[i]]
      root = np.roots(coeffs)
      roots.append(root)
    roots = np.array(roots)
    return roots

def bounded_dispersion(vb,xi):
    # Coefficients
    coeff1 = ( 1 - (1/(9*dl*xi)) )
    coeff2 = -2*coeff1*kadl*vb/dl
    coeff3 = coeff1*(kadl*vb/dl)**2
    coeff4 = 2*wpi*wpi*kadl*vb/dl
    coeff5 = -(wpi*wpi)*(kadl*vb/dl)**2
    roots = []
    for i in range(1,len(kadl)):
      # coeffs = [coeff1, coeff2[i], coeff3[i]]
      coeffs = [coeff1, coeff2[i], coeff3[i], coeff4[i], coeff5[i]]
      root = np.roots(coeffs)
      roots.append(root)
    roots = np.array(roots)
    return roots
def bounded_dispersion_sp(vb,xi,n):
    # Coefficients
    c = kadl*(vb/cia)
    # 0.5*( c - np.sqrt(c**2)
    # root = 0.5*(c-np.sqrt(c^2-(2*(-4*kadl^2- np.sqrt(16*kadl^4+8*c^2*kadl^2*(2+2*kadl^2-xi-2*n*xi))))/(2+2*kadl^2-xi-2*n*xi)))
    coeff1 = -( xi*(2*n+1) + 2*(1+kadl**2) )
    coeff2 = 2*xi*c*(2*n+1) + 4*c*(1+kadl**2)
    coeff3 = -xi*c*c*(2*n+1) + 2*( ((c*kadl)*(c*kadl)) + (c*c) + (2*kadl*kadl) )
    coeff4 = -4*kadl*kadl*c
    coeff5 = -2*kadl*kadl*c*c
    roots = []
    for i in range(1,len(kadl)):
      # coeffs = [coeff1, coeff2[i], coeff3[i]]
      coeffs = [coeff1[i], coeff2[i], coeff3[i], coeff4[i], coeff5[i]]
      root = np.roots(coeffs)
      roots.append(root)
    roots = np.array(roots)
    return roots

wbf = []
wbs = []
wbn = []
xi = 0.01
n = 1
# vb = vb/cia
# print(vb)
for i in range(len(vb)):
    roots = periodic_dispersion_norm(vb[i])
    # roots = bounded_dispersion_sp(vb[i],xi,n)
    print(roots[:,0])
    print(roots[:,1])
    wbf.append(np.real(roots[:,0]))
    wbs.append(np.real(roots[:,1]))
    wbn.append(np.imag(roots[:,0]))

wbf = np.array(wbf)
wbs = np.array(wbs)
wbn = np.array(wbn)
# exit()

# print(wbf.shape)
# exit()
# print(roots[:,0])
# print(roots[:,1])




wac /= wpi


# ==== Figure =============

##### FIG SIZE CALC ############
figsize = np.array([77,77/1.618]) #Figure size in mm
dpi = 300                         #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=True)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)
linestyles =['-', '--', '-.', ':', '-x']
# fig,axs = plt.subplots(2,2,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
# idx = np.array([0,0,1,1])
# idy = np.array([0,1,0,1])
# for i in range(len(vb)):
#     plt.cla()
#     axs[idx[i],idy[i]].set_ylim([0, max(wbf[i,:])])
#     axs[idx[i],idy[i]].set_xlim([0, max(kadl)])
#     if norm == "omega_pi":
#         axs[idx[i],idy[i]].plot(kadl[1:], wbf[i,:], 'r', label='Fast Beam Mode')
#         axs[idx[i],idy[i]].plot(kadl[1:], wbs[i,:], 'k', label="Slow Beam Mode")
#         # plt.plot(kadl[1:], wbn[i,:], 'g', label="Normal Beam Mode")
#         axs[idx[i],idy[i]].plot(kadl, wac, 'b',label="Acoustic Mode")
#         # plt.plot(ka, wb, '--w',label="Beam driven waves")
#         leg = axs[idx[i],idy[i]].legend()
#         axs[idx[i],idy[i]].set_xlabel('$k \lambda_{D}$')
#         axs[idx[i],idy[i]].set_ylabel('$\omega/\omega_{pi}$')
#     else:
#         plt.plot(kadl, wl, 'r', label="langmuir wave")
#         plt.axhline(y=1.0, color='b', linestyle='-',label='$\omega_{pe}$')
#         leg = ax.legend()
#         ax.set_xlabel('$k~[1/m]$')
#         ax.set_ylabel('$\omega/\omega_{pe}$')
#     # plt.savefig(pjoin(savedir, norm+'_disprel_%d'%i+'.png'))

fig,ax = plt.subplots(figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
for i in range(len(vb)):
    ax.cla()
    ax.set_ylim([0, 2])
    ax.set_xlim([0, max(kadl)])
    if norm == "omega_pi":
        # ax.plot(kadl[1:], wbf[i,:], linestyles[i], lw = 2, label='$\omega_{fast}$')
        # ax.plot(kadl[1:], wbs[i,:], linestyles[i], lw = 2, label='$\omega_{slow}$')
        ax.plot(kadl[1:], wbf[i,:], 'r', linestyle='-.', lw = 1, label='$\\tilde{\omega_{f}}$')
        ax.plot(kadl[1:], wbs[i,:], 'k', linestyle='--', lw = 1, label='$\\tilde{\omega_{s}}$')
        # plt.plot(kadl[1:], wbn[i,:], 'g', label="Wave growth")
        # ax.plot(kadl, wac, 'b',label="Acoustic Mode")
        # plt.plot(ka, wb, '--w',label="Beam driven waves")

        # ax.set_xlabel('$k \lambda_{D}$')
        # ax.set_ylabel('$\omega/\omega_{pi}$')
    else:
        plt.plot(kadl, wl, 'r', label="langmuir wave")
        plt.axhline(y=1.0, color='b', linestyle='-',label='$\omega_{pe}$')
        leg = ax.legend()
        ax.set_xlabel('$k~[1/m]$')
        ax.set_ylabel('$\omega/\omega_{pe}$')
    ax.plot(kadl, wac, 'b', linestyle='-',lw = 1, label="$\\tilde{\omega_{a}}$")
    ax.set_xlabel('$\\tilde{k}$')
    ax.set_ylabel('$\\tilde{\omega}$')
    leg = ax.legend(loc='upper right')
    # leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(pjoin(savedir, norm+'_disprel_%d'%i+'.png'),dpi=dpi)
plt.show()
