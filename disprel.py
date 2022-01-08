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
parser.add_argument('-i','--input', default=None, type=str, help='Path to E-field data')
parser.add_argument('-per','--periodic', action='store_true', help='Add this if the system is periodic in Y')
parser.add_argument('-yLoc','--yLocation', default=1, type=int, help='In bounded (in Y) system Choose Y location, Options: e.g. 1 (Any number between 0-Ny)')
parser.add_argument('-pl','--plot', action='store_true', help='Add this if you want to plot the figure')
parser.add_argument('-n','--norm', default='omega_pe', type=str, help='Normalizing frequency, Options: omega_pi, omega_pe')
parser.add_argument('-ihg','--inhomogen', action='store_true', help='Add this if the system is inhomogeneous')
parser.add_argument('-l','--label', default=None, type=str, help='Add additioanl label to plot')

args        = parser.parse_args()
folder      = args.input
periodic    = args.periodic
yLoc        = args.yLocation	#8 # Choose y location
plot        = args.plot
norm        = args.norm
inhomo      = args.inhomogen
addlabel    = args.label

# Set processed data directory
folder_base= os.path.basename(os.path.dirname(folder))
savedir     = pjoin(folder, '..',folder_base+'_processed')

files = glob(pjoin(folder, 'Ex_*.dat'))
pat = re.compile('\d+')
files.sort(key=lambda x: int(pat.findall(x)[-1]))

# READ TEMPORAL GRID
vars = parse_xoopic_input(pjoin(folder, '..', 'input.inp'))
# print(vars['Control'][0]['dt'])
if vars['Region']['Diagnostic'][0]['VarName'] == 'Ex':
    dumpper = vars['Region']['Diagnostic'][0]['n_step']
else:
    dumpper = 1
dt = vars['Region']['Control'][0]['dt']*dumpper

n = np.array([int(pat.findall(a)[-1]) for a in files])
n += 1 # XOOPIC diagnostics is actually off by one
t = n*dt
Nt = len(t)
# print(dt,n,t,Nt)

if  os.path.exists(savedir) and os.path.exists(pjoin(savedir,'pro_data_file.npz')):
        print('processed data exists. Loading data ...')
        f = np.load(pjoin(savedir,'pro_data_file.npz'))['data']
        print('Shape of loaded data fp: ',f.shape)
        Nt = len(f[:,0])
        x = np.load(pjoin(savedir,"x.npz"))['x']
        print("Shape of loaded data x",x.shape)

else:
  if  os.path.exists(savedir)== False:
    os.mkdir(savedir)
  print('processed data does not exist. Wait, processing data ...')

  # READ SPATIAL GRID
  data = np.loadtxt(files[0])

  x, y = data[:,2:4].T
  Ny = len(np.where(x==x[0])[0])
  Nx = len(x)//Ny
  x = x.reshape(Nx, Ny)
  y = y.reshape(Nx, Ny)

  # READ FIELD
  f = np.zeros((Nt, Nx, Ny))

  for i, file in enumerate(tqdm(files)):
    data = np.loadtxt(file)
    f[i] = data[:,-1].reshape(Nx, Ny)

    # Remove y
  if periodic == False:
      f = f[:,:,yLoc-4:yLoc+4]
  f = np.average(f, axis=2)
  x = np.average(x, axis=1)
  np.savez_compressed(pjoin(savedir,'pro_data_file.npz'),data=f)
  np.savez_compressed(pjoin(savedir,"x.npz"),x=x)

if plot:
    # FFT axes
  # dt = vars['timeStep']*vars['save_step'] #t[1]-t[0]

  Mt = Nt
  Mx = vars['Region']['Grid'][0]['J'] #len(x)
  Lx = vars['Region']['Grid'][0]['x1f'] - vars['Region']['Grid'][0]['x1s']
  dx = Lx/Mx #x[1]-x[0]
  # dt = t[1]-t[0]
  # dx = x[1]-x[0]
  # Mt = len(t)
  # Mx = len(x)
  omega = 2*np.pi*np.arange(Mt)/(Mt*dt)
  k     = 2*np.pi*np.arange(Mx)/(Mx*dx)
  print('Length of k: ',len(k))
  print('Max of k: ',np.max(k))
  Omega, K = np.meshgrid(omega, k, indexing='ij')
  print('Shape of Omega: ',Omega.shape)
  F = np.fft.fftn(f, norm='ortho')

  halflen = np.array(F.shape, dtype=int)//2
  Omega = Omega[:halflen[0],:halflen[1]]
  K = K[:halflen[0],:halflen[1]]
  F = F[:halflen[0],:halflen[1]]

  # Analytical ion-acoustic dispresion relation
  ne = vars['Region']['Load'][0]['density']
  ni = ne
  eps0 = constants('electric constant')
  kb = constants('Boltzmann constant')
  me = constants('electron mass')
  e = constants('elementary charge')
  c0 = constants('speed of light in vacuum')

  mi  = vars['Region']['Species'][1]['m'] #40*constants('atomic mass constant')
  nK  = vars['Region']['Grid'][0]['J']
  gamma_e = 5./3
  # print(vars['Region']['Load'][0]['units'])

  if 'BeamEmitter' in vars['Region']:
      units_0 = vars['Region']['BeamEmitter'][0]['units']
      if units_0 == 'MKS':
          vthE = vars['Region']['BeamEmitter'][0]['temperature']
          tEeV   = 0.5*me*(vthE*vthE)/e
          tEK    = tEeV*11604.525

      units_1 = vars['Region']['BeamEmitter'][1]['units']
      if units_1 == 'MKS':
          vthI = vars['Region']['BeamEmitter'][1]['temperature']
          tIeV   = 0.5*mi*(vthI*vthI)/e
          tIK    = tIeV*11604.525
          vb  = vars['Region']['BeamEmitter'][1]['v1drift']
          tIbeV   = 0.5*mi*(vb*vb)/e
          tIbK    = tIbeV*11604.525
  if 'Load' in vars['Region']:
      units_0 = vars['Region']['Load'][0]['units']
      if units_0 == 'MKS':
          vthE = vars['Region']['Load'][0]['temperature']
          tEeV   = 0.5*me*(vthE*vthE)/e
          tEK    = tEeV*11604.525

      units_1 = vars['Region']['Load'][1]['units']
      if units_1 == 'MKS':
          vthI = vars['Region']['Load'][1]['temperature']
          tIeV   = 0.5*mi*(vthI*vthI)/e
          tIK    = tIeV*11604.525
          vb  = vars['Region']['Load'][1]['v1drift']
          tIbeV   = 0.5*mi*(vb*vb)/e
          tIbK    = tIbeV*11604.525


# For Plasma Source
  # vthE = 419368.69338080706
  # tEeV   = 0.5*me*(vthE*vthE)/e
  # tEK    = tEeV*11604.525
  #
  # vthI = 1384.122937841148
  # tIeV   = 0.5*mi*(vthI*vthI)/e
  # tIK    = tIeV*11604.525
  # vb  = 4376.981045261688
  # tIbeV   = 0.5*mi*(vb*vb)/e
  # tIbK    = tIbeV*11604.525

  Te  = tEK #vars['tEK'] #1.6*11604
  Ti  = tIK #vars['tIK'] #0.1*11604


  wpi = np.sqrt(e**2*ni/(eps0*mi))
  wpe = np.sqrt(e**2*ne/(eps0*me))
  dl	= np.sqrt(eps0*kb*Te/(ni*e*e))
  dli	= np.sqrt(eps0*kb*Ti/(ni*e*e))
  print("wpi={}".format(wpi))
  print("wpe={}".format(wpe))
  print("dl={}".format(dl))
  cia = np.sqrt(gamma_e*kb*Te/mi)
  ka = np.linspace(0, np.max(K), nK)
  wac = np.sqrt((ka*cia)**2/(1+(ka*cia/wpi)**2))
  wah = np.sqrt( (wpi**2) * (ka*ka * dli*dli * (Te/Ti))/(1+(ka*ka * dli*dli * (Te/Ti))) )
  wl = np.sqrt( (wpe**2) * (1+me/mi) * (1+(3*ka*ka*dl*dl)/(1+me/mi)) )
  # wea =
  wb = ka*tIbK

  kadl = ka*dl

  # # print(v)
  print('vb/cia = ',vb/cia)
  #
  # exit()

  # print(kadl)

  def periodic_dispersion_norm(vb):
     vb = vb/cia
     # Coefficients
     coeff1 = ( 1 + ( 1 / (kadl*kadl) ) )
     coeff2 = -2 * kadl * vb * coeff1
     coeff3 = ((kadl*kadl) * (vb*vb) * coeff1) - 1
     roots = []
     for i in range(1,len(kadl)):
         coeffs = [coeff1[i], coeff2[i], coeff3[i]]
         root = np.roots(coeffs)
         roots.append(root)
     roots = np.array(roots)
     return roots

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

  def bounded_dispersion_inhomo(vb,xi):
      # Coefficients
      c = kadl*(vb/cia)
      vbar = (vb/cia)
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

  if inhomo:
      xi = -1.0
      roots = bounded_dispersion_inhomo(vb,xi)
  else:
      roots = periodic_dispersion(vb)

  # print(roots[:,0])
  # print(roots[:,1])
  # wbf = np.real(roots[:,0])
  # wbs = np.real(roots[:,1])
  analytic = False
  if analytic:
      wbs = np.real(roots[:,3:])
      wbf = []
      wbs0 = []
      wbs1 = []
      wbs2 = []

      for i in range(len(roots[:,0])):
          wbf.append(np.max([np.real(roots[i,0]),np.real(roots[i,1])]))
          wbs0.append(np.max([wbs[i,0],wbs[i,1],wbs[i,4]]))
          # wbs1.append(np.max([wbs[i,1],wbs[i,4]]))
          wbs2.append(np.min([wbs[i,1],wbs[i,4]]))

      wbs0 = np.array(wbs0)
      # wbs1 = np.array(wbs1)
      wbs2 = np.array(wbs2)
      wbs = np.array([wbs0,wbs2])

      wbf = np.array(wbf)

  #omega_pe

  if norm == "omega_pi":
    omega /= wpi
    Omega /= wpi
  else:
    omega /=wpe
    Omega /=wpe

  wl /= wpe
  # print(wl)
  wac /= wpi
  wah /= wpi
  wb /= wpi
  K *= dl
  # print(wb)

  Z = np.log(np.abs(F))
  # Z = np.abs(F)


  # Z = np.imag(F)
  # print(np.max(Z))
  # Z /= np.max(Z)
  # Z /= 14492.03

  # [8.05648594639641, 8.321326392686734, 9.116097355123907] vb = 0.1
  # [8.256258928788315, 8.352575352771161, 9.581354281956306] vb = 0.5
  #Z = np.abs(F)

  # ==== Figure =============

  ##### FIG SIZE CALC ############
  figsize = np.array([80,80/1.618]) #Figure size in mm
  dpi = 300                         #Print resolution
  ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

  mp.rc('text', usetex=False)
  mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
  mp.rc('axes', titlesize=10)
  mp.rc('axes', labelsize=10)
  mp.rc('xtick', labelsize=10)
  mp.rc('ytick', labelsize=10)
  mp.rc('legend', fontsize=10)
  linestyles =['-.', '--', '-', ':']
  colors = ['y', 'c']

  fig,ax = plt.subplots(figsize=figsize/25.4,constrained_layout=True,dpi=dpi)
  oRange = len(K[:,0]) #for full omega len(K[:,0])
  # oRange = int(oRange/50)
  # print(K[:oRange,:].shape,Omega[:oRange,:].shape,Z[:oRange,:].shape)
  # print(oRange)
  if norm == "omega_pi":
    if inhomo:
        oRange = int(oRange/10) #for bounded system in x
    else:
        oRange = int(oRange/200) #for periodic system in x
    plt.pcolor(K[:oRange,:], Omega[:oRange,:], Z[:oRange,:],shading='auto',vmin=np.min(Z[:oRange,:]),vmax=np.max(Z[:oRange,:])) # 0.05*np.max(Z[:oRange,:])) #np.min(Z[:oRange,:])
    cbar = plt.colorbar()
    cbar.set_label('$\zeta$')
    ax.set_ylabel('$\omega/\omega_{pi}$')
  else:
    oRange = int(oRange/5)
    plt.pcolor(K[:oRange,:], Omega[:oRange,:], Z[:oRange,:],shading='auto',vmin=np.min(Z[:oRange,:]),vmax=np.max(Z[:oRange,:]))
    #plt.pcolor(K, Omega, Z,shading='auto',vmin=np.min(Z),vmax=np.max(Z))
    #plt.imshow(K, Omega, Z)
    ax.set_ylabel('$\omega/\omega_{pe}$')
    plt.colorbar()


  if analytic:
      if norm == "omega_pi":
        plt.plot(kadl[1:], wbf, color='w', linestyle='-.', lw = 1.5, label='$\\tilde{\omega_{f}}$')
        for i in range(2):
            plt.plot(kadl[1:], wbs[i,:], linestyles[i], color = colors[i],lw = 1.0, label='$\\tilde{\omega_{s}}$(%d'%i+')')
        plt.plot(kadl, wac, color='b', lw = 1.5,label="$\\tilde{\omega_{a}}$")
        # plt.plot(ka, wb, '--w',label="Beam driven waves")
        leg = ax.legend(loc='upper right',framealpha=0.5)
        ax.set_xlabel('$\\tilde{k}$')
        ax.set_ylabel('$\\tilde{\omega}$')
        # ax.set_xlabel('$k \lambda_{D}$')
        # ax.set_ylabel('$\omega/\omega_{pi}$')

      else:
        plt.plot(kadl, wl, '--w', label="langmuir wave")
        plt.axhline(y=1.0, color='w', linestyle='--',label='$\omega_{pe}$')
        leg = ax.legend(loc='upper right')
        ax.set_xlabel('$k~[1/m]$')
        ax.set_ylabel('$\omega/\omega_{pe}$')

  ax.set_ylim([0, 2])
  ax.set_xlabel('$k \lambda_{D}$')
  plt.savefig(pjoin(savedir, norm+'_'+addlabel+'_disprel.png'),dpi=dpi)
  plt.show()
