#!/usr/bin/env python3
import sys

from qsc.types import Results
from qsc import plot
from qsc.plot import plot_boundary


import numpy as np
import jax
import jax.numpy as jnp

from qsc.qsc_method import Qsc_method
from qsc.calculate_r2 import calc_r2
from qsc import calculate_r1
  
  

print("Running pyNACX...")
#stel = Qsc(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r2', B2c=-0.7, p2=-600000.)

# this runs the procedures for making a stellarator in a jax compatibile way    
ans = Qsc_method(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r1', B2c=-0.7, p2=-600000.)



res = calc_r2(ans.pre_calculation_results.init_axis_results.X1c, ans.complete_calculation_results.complete_r1_results.r1_results.Y1c, ans.complete_calculation_results.complete_r1_results.r1_results.Y1s, ans.inputs.B0 / jnp.abs(ans.pre_calculation_results.init_axis_results.G0), ans.pre_calculation_results.init_axis_results.d_d_varphi, ans.pre_calculation_results.solve_sigma_equation_results.iotaN, ans.pre_calculation_results.init_axis_results.torsion, ans.pre_calculation_results.init_axis_results.abs_G0_over_B0, ans.inputs.B2s, ans.inputs.B0, ans.pre_calculation_results.init_axis_results.curvature, ans.inputs.etabar, ans.inputs.B2c, ans.inputs.spsi, ans.inputs.sG, ans.inputs.p2, ans.pre_calculation_results.solve_sigma_equation_results.sigma, ans.inputs.I2/ans.inputs.B0, ans.inputs.nphi, ans.pre_calculation_results.init_axis_results.d_l_d_phi, ans.pre_calculation_results.init_axis_results.helicity, ans.inputs.nfp, ans.pre_calculation_results.init_axis_results.G0, ans.pre_calculation_results.solve_sigma_equation_results.iota, ans.inputs.I2, ans.pre_calculation_results.init_axis_results.varphi, ans.complete_calculation_results.complete_r1_results.r1_results.d_X1c_d_varphi, ans.complete_calculation_results.complete_r1_results.r1_results.d_Y1c_d_varphi, ans.complete_calculation_results.complete_r1_results.r1_results.d_Y1s_d_varphi, ans.pre_calculation_results.init_axis_results.d_phi, ans.pre_calculation_results.init_axis_results.axis_length)

#this holds all the results of the calculation ans will called by the plotting code, to minimize flattening and unflattening  


print("pyNACX finished")
#print(stel.iota) # Rotational transform on-axis for this configuration
#print(stel.d2_volume_d_psi2) # Magnetic well V''(psi) 
#print(stel.DMerc_times_r2) # Mercier criterion parameter DMerc multiplied by r^2 
#print(stel.min_L_grad_B) # Scale length associated with the grad grad B tensor
#print(stel.grad_grad_B_inverse_scale_length) # Scale length associated with the grad grad B tensor Semo: grad_grad_B_inverse_scale_length is not calculted in r1
#print("plotting...")
plot_boundary(ans) # Plot the flux surface shape at the default radius r=1
#plot.plot(ans) # Plot relevant near axis parameters
