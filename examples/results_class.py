from typing import NamedTuple
import jax.numpy as jnp


class Results(NamedTuple):
  """
  results class that stores the results of the calculation for use in plotting software
  """
  # init acess(pre calc)
  helicity: jnp.ndarray
  normal_cylindrical: jnp.ndarray
  etabar_squared_over_curvature_squared: jnp.ndarray
  varphi: jnp.ndarray
  d_d_phi: jnp.ndarray
  d_varphi_d_phi: jnp.ndarray
  d_d_varphi: jnp.ndarray
  phi: jnp.ndarray
  abs_G0_over_B0: jnp.ndarray
  d_phi: jnp.ndarray
  R0: jnp.ndarray
  Z0: jnp.ndarray
  R0p: jnp.ndarray
  Z0p: jnp.ndarray
  R0pp : jnp.ndarray
  Z0pp : jnp.ndarray
  R0ppp : jnp.ndarray
  Z0ppp : jnp.ndarray
  G0: jnp.ndarray
  d_l_d_phi: jnp.ndarray
  axis_length: jnp.ndarray
  curvature: jnp.ndarray
  torsion: jnp.ndarray
  X1s: jnp.ndarray
  X1c: jnp.ndarray
  min_R0: jnp.ndarray
  tangent_cylindrical: jnp.ndarray
  normal_cylindrical: jnp.ndarray
  binormal_cylindrical: jnp.ndarray
  Bbar: jnp.ndarray
  abs_G0_over_B0: jnp.ndarray
  lasym: jnp.ndarray
  R0_func: jnp.ndarray
  Z0_func: jnp.ndarray
  normal_R_spline: jnp.ndarray
  normal_phi_spline: jnp.ndarray
  normal_z_spline: jnp.ndarray
  binormal_R_spline: jnp.ndarray
  binormal_phi_spline: jnp.ndarray
  binormal_z_spline: jnp.ndarray
  tangent_R_spline: jnp.ndarray
  tangent_phi_spline: jnp.ndarray
  tangent_z_spline: jnp.ndarray
  nu_spline: jnp.ndarray 
  # solve sigma equation (pre calc)
  sigma: jnp.ndarray  
  iota: jnp.ndarray
  iotaN: jnp.ndarray
  # r1_results (calc)
  # r1_results (r1)(calc) 
  Y1s: jnp.ndarray 
  Y1c: jnp.ndarray 
  X1s_untwisted: jnp.ndarray 
  X1c_untwisted: jnp.ndarray 
  Y1s_untwisted: jnp.ndarray 
  Y1c_untwisted: jnp.ndarray 
  elongation: jnp.ndarray 
  mean_elongation: jnp.ndarray 
  max_elongation: jnp.ndarray 
  d_X1c_d_varphi: jnp.ndarray 
  d_X1s_d_varphi: jnp.ndarray 
  d_Y1s_d_varphi: jnp.ndarray 
  d_Y1c_d_varphi: jnp.ndarray 
  #grad_b_tensor_results (r1)(calc)
  grad_B_tensor: jnp.ndarray 
  grad_B_tensor_cylindrical: jnp.ndarray 
  grad_B_colon_grad_B: jnp.ndarray 
  L_grad_B: jnp.ndarray 
  inv_L_grad_B: jnp.ndarray 
  min_L_grad_B: jnp.ndarray 
  # r2_results (calc)
  # mercier results (r2) (calc) 
  DGeod_times_r2: jnp.ndarray 
  d2_volume_d_psi2: jnp.ndarray 
  DWell_times_r2: jnp.ndarray 
  DMerc_times_r2: jnp.ndarray
  #grad grad b results (r2) (calc) 
  grad_grad_B: jnp.ndarray  
  grad_grad_B_inverse_scale_length_vs_varphi: jnp.ndarray  
  L_grad_grad_B: jnp.ndarray 
  grad_grad_B_inverse_scale_length: jnp.ndarray 
  # r2 results (r2) (calc)
  N_helicity: jnp.ndarray 
  G2: jnp.ndarray 
  d_curvature_d_varphi: jnp.ndarray 
  d_torsion_d_varphi: jnp.ndarray 
  d_X20_d_varphi: jnp.ndarray 
  d_X2s_d_varphi: jnp.ndarray 
  d_X2c_d_varphi: jnp.ndarray 
  d_Y20_d_varphi: jnp.ndarray 
  d_Y2s_d_varphi: jnp.ndarray 
  d_Y2c_d_varphi: jnp.ndarray 
  d_Z20_d_varphi: jnp.ndarray 
  d_Z2s_d_varphi: jnp.ndarray 
  d_Z2c_d_varphi: jnp.ndarray 
  d2_X1c_d_varphi2: jnp.ndarray 
  d2_Y1c_d_varphi2: jnp.ndarray 
  d2_Y1s_d_varphi2: jnp.ndarray 
  V1: jnp.ndarray 
  V2: jnp.ndarray 
  V3: jnp.ndarray 
  X20: jnp.ndarray 
  X2s: jnp.ndarray 
  X2c: jnp.ndarray 
  Y20: jnp.ndarray 
  Y2s: jnp.ndarray 
  Y2c: jnp.ndarray 
  Z20: jnp.ndarray 
  Z2s: jnp.ndarray 
  Z2c: jnp.ndarray 
  beta_1s: jnp.ndarray 
  B20: jnp.ndarray 
  X20_untwisted: jnp.ndarray 
  X2s_untwisted: jnp.ndarray 
  X2c_untwisted: jnp.ndarray 
  Y20_untwisted: jnp.ndarray 
  Y2s_untwisted: jnp.ndarray 
  Y2c_untwisted: jnp.ndarray 
  Z20_untwisted: jnp.ndarray 
  Z2s_untwisted: jnp.ndarray 
  Z2c_untwisted: jnp.ndarray 
  # r singularity results (r2) (calc)
  r_singularity_vs_varphi: jnp.ndarray 
  inv_r_singularity_vs_varphi: jnp.ndarray 
  r_singularity_basic_vs_varphi: jnp.ndarray 
  r_singularity: jnp.ndarray 
  r_singularity_theta_vs_varphi: jnp.ndarray 
  r_singularity_residual_sqnorm: jnp.ndarray 
  # r3_results (r3)(calc)
  X3c1: jnp.ndarray
  Y3c1: jnp.ndarray
  Y3s1: jnp.ndarray
  X3s1: jnp.ndarray
  Z3c1: jnp.ndarray
  Z3s1: jnp.ndarray
  X3c3: jnp.ndarray
  X3s3: jnp.ndarray
  Y3c3: jnp.ndarray
  Y3s3: jnp.ndarray
  Z3c3: jnp.ndarray
  Z3s3: jnp.ndarray 
  d_X3c1_d_varphi: jnp.ndarray 
  d_Y3c1_d_varphi: jnp.ndarray 
  d_Y3s1_d_varphi: jnp.ndarray 
  flux_constraint_coefficient: jnp.ndarray 
  B0_order_a_squared_to_cancel: jnp.ndarray 
  X3c1_untwisted: jnp.ndarray 
  Y3c1_untwisted: jnp.ndarray 
  Y3s1_untwisted: jnp.ndarray 
  X3s1_untwisted: jnp.ndarray 
  X3s3_untwisted: jnp.ndarray 
  X3c3_untwisted: jnp.ndarray 
  Y3c3_untwisted: jnp.ndarray 
  Y3s3_untwisted: jnp.ndarray 
  Z3s1_untwisted: jnp.ndarray 
  Z3s3_untwisted: jnp.ndarray 
  Z3c1_untwisted: jnp.ndarray 
  Z3c3_untwisted: jnp.ndarray 