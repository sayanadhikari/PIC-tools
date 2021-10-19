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
from collections import OrderedDict


parser = argparse.ArgumentParser(description='Plasma Dispersion Processor')
parser.add_argument('-n','--norm', default='omega_pi', type=str, help='Normalizing frequency, Options: omega_pi, omega_pe')
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
# tbeV = np.array([0.1, 0.5, 1.0, 7.5])       #  [eV]
tbeV = 0.1 #np.array([0.1])       #  [eV]
tIbK = tbeV*11604.525        #  [K]
vb   = np.sqrt(2*(tbeV*e)/mi)  #[m/s]
xi = 0.01

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
def bounded_dispersion_inhomo(vb,xi):
    # Coefficients
    c = kadl*(vb/cia)
    vbar = (vb/cia)
    # 0.5*( c - np.sqrt(c**2)
    # root = 0.5*(c-np.sqrt(c^2-(2*(-4*kadl^2- np.sqrt(16*kadl^4+8*c^2*kadl^2*(2+2*kadl^2-xi-2*n*xi))))/(2+2*kadl^2-xi-2*n*xi)))
    # coeff1 = - 4 * xi
    # coeff2 = 16 * c * xi
    # coeff3 = 4 * xi * (1 - 6*c*c)
    # coeff4 = -16 * c * xi * (1 - c*c)
    # coeff5 = ( 4*c*c*xi * (6 - c*c) + 4*kadl**4 )
    # coeff6 = - 8 * c * kadl*kadl * (kadl*kadl + 2*xi*vbar*vbar)
    # coeff7 = 4 * c*c * kadl*kadl * (xi*vbar*vbar + 2*kadl*kadl)
    # coeff8 = -4 * (c**3) * (kadl**4)
    # coeff9 = - (c**4) * (kadl**4)
    coeff1 = 4 * xi
    coeff2 = -16 * c * xi
    coeff3 = 24*xi*c*c - 4*xi
    coeff4 = 16*c*xi - 16*xi*c**3
    coeff5 = (4*xi*c**4) - (4*kadl**4) - (24*xi*c**2)
    coeff6 = (8*c*kadl**4) + (16*xi*c**3)
    coeff7 = -(8*c*c*kadl**4) - (4*xi*c**4)
    coeff8 = 4*(c**3)*(kadl**4)
    coeff9 = - (c**4) * (kadl**4)
    roots = []
    for i in range(1,len(kadl)):
      # coeffs = [coeff1, coeff2[i], coeff3[i]]
      coeffs = [coeff1, coeff2[i], coeff3[i], coeff4[i], coeff5[i], coeff6[i], coeff7[i], coeff8[i], coeff9[i]]
      root = np.roots(coeffs)
      roots.append(root)
    roots = np.array(roots)
    return roots

def bounded_dispersion_inhomo_woion(vb,xi,n):
    # Coefficients
    c = kadl*(vb/cia)
    # 0.5*( c - np.sqrt(c**2)
    # root = 0.5*(c-np.sqrt(c^2-(2*(-4*kadl^2- np.sqrt(16*kadl^4+8*c^2*kadl^2*(2+2*kadl^2-xi-2*n*xi))))/(2+2*kadl^2-xi-2*n*xi)))
    coeff1 = 2 - ((2*n+1)*xi) + (2*kadl*kadl)
    coeff2 = 2*c*xi*(2*n+1) - 4*c*(1+kadl**2)
    coeff3 = -xi*c*c*(2*n+1) + 2*c*c*(1+kadl**2) - (2*kadl*kadl)

    roots = []
    for i in range(1,len(kadl)):
      coeffs = [coeff1[i], coeff2[i], coeff3[i]]
      # coeffs = [coeff1[i], coeff2[i], coeff3[i], coeff4[i], coeff5[i]]
      root = np.roots(coeffs)
      roots.append(root)
    roots = np.array(roots)
    return roots

wbf = []
wbs = []
wbn = []

# xi_norm = 0.01
# n = np.linspace(0,100,3,dtype='int')
# vb = vb/cia
def verify_sol(omg,vb,xi):
    c = kadl[1:]*(vb/cia)
    sol = -2 * (np.sqrt(xi) / (kadl[1:]*kadl[1:])) * np.sqrt( 1 - (1/omg**2) ) + (1/omg**2) + ( 1 / (omg-c)**2 )
    return sol



print(vb)
# roots = periodic_dispersion(vb[i])
roots = bounded_dispersion_inhomo(vb,xi)
# sol = verify_sol(roots[:,5],vb,xi)
# print(sol)
# print(np.real(sol))
# sol1 = []

wbs = np.real(roots[:,3:])
wbs0 = []
wbs1 = []
wbs2 = []

for i in range(len(roots[:,0])):
    wbf.append(np.max([np.real(roots[i,0]),np.real(roots[i,1])]))
    wbs0.append(np.max([wbs[i,0],wbs[i,1],wbs[i,4]]))
    wbs1.append(np.max([wbs[i,1],wbs[i,4]]))
    wbs2.append(np.min([wbs[i,1],wbs[i,4]]))

wbs0 = np.array(wbs0)
wbs1 = np.array(wbs1)
wbs2 = np.array(wbs2)
wbs = np.array([wbs0,wbs1,wbs2])

wbf = np.array(wbf)



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
mp.rc('legend', fontsize=8)

linestyles =['-.', '--', '-', ':','-.', '--', '-', ':']

# print(linestyles[0])

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

fig,ax = plt.subplots(figsize=figsize/25.4,constrained_layout=True,dpi=dpi)
# ax.cla()
# ymax = max([max(wbf[i,:])+0.1*max(wbf[i,:]), max(wac)+0.1*max(wac)])
ax.set_ylim([0, 3.1])
ax.set_xlim([0, max(kadl)])
if norm == "omega_pi":
    ax.plot(kadl[1:], wbf, linestyles[2], lw = 1.0, label='$\\tilde{\omega_{f}}$')
    for i in range(3):
        ax.plot(kadl[1:], wbs[i,:], linestyles[i], lw = 1.0, label='$\\tilde{\omega_{s}}$(%d'%i+')')
    # plt.plot(kadl[1:], wbn[i,:], 'g', label="Normal Beam Mode")
    # plt.plot(kadl, wac, 'b',label="Acoustic Mode")
    # plt.plot(ka, wb, '--w',label="Beam driven waves")
    # leg = ax.legend()
else:
    plt.plot(kadl, wl, 'r', label="langmuir wave")
    plt.axhline(y=1.5, color='b', linestyle='-',label='$\omega_{pe}$')
    ax.set_xlabel('$k~[1/m]$')
    ax.set_ylabel('$\omega/\omega_{pe}$')

ax.plot(kadl, wac, 'b',lw = 1.0, label="$\\tilde{\omega_{a}}$")
ax.set_xlabel('$\\tilde{k}$')
ax.set_ylabel('$\\tilde{\omega}$')
leg = ax.legend(loc='upper right')
# leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper center', borderaxespad=0.)
# Shrink current axis's height by 10% on the bottom
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.71])
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.53),fancybox=True, shadow=False, ncol=3)

plt.savefig(pjoin(savedir, norm+'_disprel_%0.2f'%tbeV+'_xi_%0.2f'%xi+'.png'),dpi=dpi)
plt.show()
