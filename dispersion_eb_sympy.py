import numpy as np 
from scipy.constants import value as constants
import matplotlib as mp
import matplotlib.pyplot as plt
from sympy import *
# -----------------------------------------
# Sympy Polynomial Coefficients Collection (Tutorial)
# -----------------------------------------
# x, y, z = symbols('x y z')
# expr = x*y+ x-3+2*x**2-z*x**2+x**3 
# expr1 = expr.coeff(x,3)
# print(expr1)
# -----------------------------------------
# For Polynomial of any sort
# -----------------------------------------
# x = symbols('x')
# expr = x*(x-2)
# #expr = (x-2)/x
# a = Poly(expr, x)
# print(a.all_coeffs())
# -----------------------------------------
# Dispersion relation for the electron beam
def symbolic_coeffs():
    w, k, alp, v0 = symbols('w, k, alp, v0')
    expr = (w**2)*(w-k*v0)**2 - (1-alp)*((w-k*v0)**2) - alp*w**2
    p = Poly(expr, w)
    coff = p.all_coeffs()    
    return coff

# ------------------------------------------
AMU = constants('atomic mass constant')
e = constants('elementary charge')
me = constants('electron mass')
mi = 40*AMU
alp = 0.1
k = np.linspace(0, 0.5, 100)
v0 = 5

coff = symbolic_coeffs()
#print(coff[1])
#print(type(coff[1]))

#c1 = eval(str(coff[1]))
#print(c1)

def EPW_dispersion():
     # Coefficients
     coeff1 = eval(str(coff[0]))
     coeff2 = eval(str(coff[1]))
     coeff3 = eval(str(coff[2]))
     coeff4 = eval(str(coff[3]))
     coeff5 = eval(str(coff[4]))
     roots = []
     for i in range(1,len(k)): 
         coeffs = [coeff1, coeff2[i], coeff3[i], coeff4[i], coeff5[i]]
         root = np.roots(coeffs)
         roots.append(root)
     roots = np.array(roots)
     return roots


roots_EPW = EPW_dispersion()
solved_analytic = True
if solved_analytic:    
    ep1 = np.real(roots_EPW[:,0])
    ep2 = np.real(roots_EPW[:,1])
    ep3 = np.real(roots_EPW[:,2])
    ep4 = np.real(roots_EPW[:,3])

    epim1 = np.imag(roots_EPW[:,0])
    epim2 = np.imag(roots_EPW[:,1])
    epim3 = np.imag(roots_EPW[:,2])
    epim4 = np.imag(roots_EPW[:,3])


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
figsize = np.array([80,80/1.618]) #Figure size in mm (FOR SINGLE FIGURE)
dpi = 1200                        #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if solved_analytic:
    fig, ax = plt.subplots(1, 2, figsize=figsize/10.4,constrained_layout=True,dpi=ppi)
    ax[0].plot(k[1:], ep1, color='r', linestyle='-', lw = 2.0, label='$EPW$')
    ax[0].plot(k[1:], ep2, color='g', linestyle='-', lw = 2.0, label='$EPW$') 
    ax[0].plot(k[1:], ep3, color='b', linestyle='-', lw = 2.0, label='$EPW$') 
    ax[0].plot(k[1:], ep4, color='k', linestyle='-', lw = 2.0, label='$EPW$')  
    
    ax[0].set_xlabel('$k \lambda_{D}$')
    ax[0].set_ylabel('$\omega/\omega_{pe}$')   
    ax[0].set_title('Real Roots')
    ax[0].grid(True)
    ax[0].set_xlim([0, np.max(k)])
    ax[0].set_ylim([0, np.max(ep2)])

    ax[1].plot(k[1:], epim1, color='r', linestyle='-', lw = 2.0, label='$EPW$')
    ax[1].plot(k[1:], epim2, color='g', linestyle='-', lw = 2.0, label='$EPW$')
    ax[1].plot(k[1:], epim3, color='b', linestyle='-', lw = 2.0, label='$EPW$')
    ax[1].plot(k[1:], epim4, color='k', linestyle='-', lw = 2.0, label='$EPW$')
    
    ax[1].set_xlabel('$k \lambda_{D}$')
    ax[1].set_ylabel('$\omega/\omega_{pe}$')  
    ax[1].set_title('Imaginary Roots') 
    ax[1].grid(True)

plt.show()
