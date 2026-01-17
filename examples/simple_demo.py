#!/usr/bin/env python3
import sys

from qsc.types import Results
from qsc import plot
from qsc.plot import plot_boundary
from qsc.calculate_r1 import solve_sigma_equation

import numpy as np
import jax
import jax.numpy as jnp

from qsc.qsc_method import Qsc_method
from qsc.calculate_r2 import calc_r2
from qsc import calculate_r1

import pickle

print("Running pyNACX...")
# stel = Qsc(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r2', B2c=-0.7, p2=-600000.)

# this runs the procedures for making a stellarator in a jax compatibile way    
ans = Qsc_method(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r1', B2c=-0.7, p2=-600000.)




print("pyNACX finished")
#print(stel.iota) # Rotational transform on-axis for this configuration
#print(stel.d2_volume_d_psi2) # Magnetic well V''(psi) 
#print(stel.DMerc_times_r2) # Mercier criterion parameter DMerc multiplied by r^2 
#print(stel.min_L_grad_B) # Scale length associated with the grad grad B tensor
#print(stel.grad_grad_B_inverse_scale_length) # Scale length associated with the grad grad B tensor Semo: grad_grad_B_inverse_scale_length is not calculted in r1
#print("plotting...")
plot_boundary(ans) # Plot the flux surface shape at the default radius r=1
#plot.plot(ans) # Plot relevant near axis parameters
