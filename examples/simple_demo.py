#!/usr/bin/env python3
import sys

from qsc.types import Results
from qsc import plot
from qsc.plot import plot_boundary


import numpy as np
import jax
import jax.numpy as jnp

from qsc.qsc_method import Qsc_method
from qsc import calculate_r1
  
  

print("Running pyNACX...")
#stel = Qsc(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r2', B2c=-0.7, p2=-600000.)

# this runs the procedures for making a stellarator in a jax compatibile way    
ans = Qsc_method(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r2', B2c=-0.7, p2=-600000.)



#this holds all the results of the calculation ans will called by the plotting code, to minimize flattening and unflattening
res = Results(
  # init axis(pre calc)
  helicity = ans[0][0][0],
  etabar_squared_over_curvature_squared= ans[0][0][1],
  varphi= ans[0][0][2],
  d_d_phi= ans[0][0][3],
  d_varphi_d_phi= ans[0][0][4],
  d_d_varphi= ans[0][0][5],
  phi= ans[0][0][6],
  d_phi= ans[0][0][7],
  R0= ans[0][0][8],
  Z0= ans[0][0][9],
  R0p= ans[0][0][10],
  Z0p= ans[0][0][11],
  R0pp = ans[0][0][12],
  Z0pp = ans[0][0][13],
  R0ppp = ans[0][0][14],
  Z0ppp = ans[0][0][15],
  G0= ans[0][0][16],
  d_l_d_phi= ans[0][0][17],
  axis_length= ans[0][0][18],
  curvature= ans[0][0][19],
  torsion= ans[0][0][20],
  X1s= ans[0][0][21],
  X1c= ans[0][0][22],
  min_R0= ans[0][0][23],
  tangent_cylindrical= ans[0][0][24],
  normal_cylindrical= ans[0][0][25],
  binormal_cylindrical= ans[0][0][26],
  Bbar= ans[0][0][27],
  abs_G0_over_B0= ans[0][0][28],
  lasym= ans[0][0][29],
  R0_func= ans[0][0][30],
  Z0_func= ans[0][0][31],
  normal_R_spline= ans[0][0][32],
  normal_phi_spline= ans[0][0][33],
  normal_z_spline= ans[0][0][34],
  binormal_R_spline= ans[0][0][35],
  binormal_phi_spline= ans[0][0][36],
  binormal_z_spline= ans[0][0][37],
  tangent_R_spline= ans[0][0][38],
  tangent_phi_spline= ans[0][0][39],
  tangent_z_spline= ans[0][0][40],
  nu_spline= ans[0][0][41],
  # solve sigma equation (pre calc)
  sigma= ans[0][1][0],
  iota= ans[0][1][1],
  iotaN= ans[0][1][2],
  # r1_results (calc)
  # r1_results (r1)(calc) 
  Y1s= ans[1][0][0][0],
  Y1c= ans[1][0][0][1],
  X1s_untwisted= ans[1][0][0][2],
  X1c_untwisted= ans[1][0][0][3],
  Y1s_untwisted= ans[1][0][0][4],
  Y1c_untwisted= ans[1][0][0][5],
  elongation= ans[1][0][0][6],
  mean_elongation= ans[1][0][0][7],
  max_elongation= ans[1][0][0][8],
  d_X1c_d_varphi= ans[1][0][0][9],
  d_X1s_d_varphi= ans[1][0][0][10],
  d_Y1s_d_varphi= ans[1][0][0][11],
  d_Y1c_d_varphi= ans[1][0][0][12],
  #grad_b_tensor_results (r1)(calc)
  grad_B_tensor= ans[1][0][1][0],
  grad_B_tensor_cylindrical= ans[1][0][1][1],
  grad_B_colon_grad_B= ans[1][0][1][2],
  L_grad_B= ans[1][0][1][3],
  inv_L_grad_B= ans[1][0][1][4],
  min_L_grad_B= ans[1][0][1][5],
  # r2_results (calc)
  # mercier results (r2) (calc) 
  DGeod_times_r2= ans[1][1][0][0],
  d2_volume_d_psi2= ans[1][1][0][1],
  DWell_times_r2= ans[1][1][0][2],
  DMerc_times_r2= ans[1][1][0][3],
  #grad grad b results (r2) (calc) 
  grad_grad_B= ans[1][1][1][0],
  grad_grad_B_inverse_scale_length_vs_varphi= ans[1][1][1][1],
  L_grad_grad_B= ans[1][1][1][2],
  grad_grad_B_inverse_scale_length= ans[1][1][1][3],
  # r2 results (r2) (calc)
  N_helicity= ans[1][1][2][0],
  G2= ans[1][1][2][1],
  d_curvature_d_varphi= ans[1][1][2][2],
  d_torsion_d_varphi= ans[1][1][2][3],
  d_X20_d_varphi= ans[1][1][2][4],
  d_X2s_d_varphi= ans[1][1][2][5],
  d_X2c_d_varphi= ans[1][1][2][6],
  d_Y20_d_varphi= ans[1][1][2][7],
  d_Y2s_d_varphi= ans[1][1][2][8],
  d_Y2c_d_varphi= ans[1][1][2][9],
  d_Z20_d_varphi= ans[1][1][2][10],
  d_Z2s_d_varphi= ans[1][1][2][11],
  d_Z2c_d_varphi= ans[1][1][2][12],
  d2_X1c_d_varphi2= ans[1][1][2][13],
  d2_Y1c_d_varphi2= ans[1][1][2][14],
  d2_Y1s_d_varphi2= ans[1][1][2][15],
  V1= ans[1][1][2][16],
  V2= ans[1][1][2][17],
  V3= ans[1][1][2][18],
  X20= ans[1][1][2][19],
  X2s= ans[1][1][2][20],
  X2c= ans[1][1][2][21],
  Y20= ans[1][1][2][22],
  Y2s= ans[1][1][2][23],
  Y2c= ans[1][1][2][24],
  Z20= ans[1][1][2][25],
  Z2s= ans[1][1][2][26],
  Z2c= ans[1][1][2][27],
  beta_1s= ans[1][1][2][28],
  B20= ans[1][1][2][29],
  X20_untwisted= ans[1][1][2][30],
  X2s_untwisted= ans[1][1][2][31],
  X2c_untwisted= ans[1][1][2][32],
  Y20_untwisted= ans[1][1][2][33],
  Y2s_untwisted= ans[1][1][2][34],
  Y2c_untwisted= ans[1][1][2][35],
  Z20_untwisted= ans[1][1][2][36],
  Z2s_untwisted= ans[1][1][2][37],
  Z2c_untwisted= ans[1][1][2][38],
  # r singularity results (r2) (calc)
  r_singularity_vs_varphi= ans[1][1][3][0],
  inv_r_singularity_vs_varphi= ans[1][1][3][1],
  r_singularity_basic_vs_varphi= ans[1][1][3][2],
  r_singularity= ans[1][1][3][3],
  r_singularity_theta_vs_varphi= ans[1][1][3][4],
  r_singularity_residual_sqnorm= ans[1][1][3][5],
  # r3_results (r3)(calc)
  X3c1= ans[1][0][0],
  Y3c1= ans[1][0][1],
  Y3s1= ans[1][0][2],
  X3s1= ans[1][0][3],
  Z3c1= ans[1][0][4],
  Z3s1= ans[1][0][5],
  X3c3= ans[1][0][6],
  X3s3= ans[1][0][7],
  Y3c3= ans[1][0][8],
  Y3s3= ans[1][0][9],
  Z3c3= ans[1][0][10],
  Z3s3= ans[1][0][11],
  d_X3c1_d_varphi= ans[1][0][12],
  d_Y3c1_d_varphi= ans[1][0][13],
  d_Y3s1_d_varphi= ans[1][0][14],
  flux_constraint_coefficient= ans[1][0][15],
  B0_order_a_squared_to_cancel= ans[1][0][16],
  X3c1_untwisted= ans[1][0][17],
  Y3c1_untwisted= ans[1][0][18],
  Y3s1_untwisted= ans[1][0][19],
  X3s1_untwisted= ans[1][0][20],
  X3s3_untwisted= ans[1][0][21],
  X3c3_untwisted= ans[1][0][22],
  Y3c3_untwisted= ans[1][0][23],
  Y3s3_untwisted= ans[1][0][24],
  Z3s1_untwisted= ans[1][0][25],
  Z3s3_untwisted= ans[1][0][26],
  Z3c1_untwisted= ans[1][0][27],
  Z3c3_untwisted= ans[1][0][28],
  # inputs
  rc = ans[2][0], 
  zs = ans[2][1],
  rs = ans[2][2],
  zc = ans[2][3],
  nfp= ans[2][4],
  etabar= ans[2][5],
  sigma0= ans[2][6],
  B0= ans[2][7],
  I2= ans[2][8],
  sG= ans[2][9],
  spsi= ans[2][10],
  nphi= ans[2][11],
  B2s= ans[2][12],
  B2c= ans[2][13],
  p2= ans[2][14],
  order= ans[2][15],
)
  



print("pyNACX finished")
#print(stel.iota) # Rotational transform on-axis for this configuration
#print(stel.d2_volume_d_psi2) # Magnetic well V''(psi) 
#print(stel.DMerc_times_r2) # Mercier criterion parameter DMerc multiplied by r^2 
#print(stel.min_L_grad_B) # Scale length associated with the grad grad B tensor
#print(stel.grad_grad_B_inverse_scale_length) # Scale length associated with the grad grad B tensor Semo: grad_grad_B_inverse_scale_length is not calculted in r1
#print("plotting...")
plot_boundary(res) # Plot the flux surface shape at the default radius r=1
plot.plot(res) # Plot relevant near axis parameters
