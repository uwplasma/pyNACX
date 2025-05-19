import numpy as np
import jax.numpy as jnp

#get inputs from file storing some inputs calculated in the original qsc
data = np.load('qsc/tests/values.npy', allow_pickle=True).item()
#data_jax = {k: jnp.asarray(v) for k, v in data.items()}

# run calculate with these inputs 
from qsc.qsc_method import calculate

r1_results, r2_results, r3_results = calculate(jnp.asarray(data['nfp']), jnp.asarray(data['etabar']), jnp.asarray(data['curvature']), jnp.asarray(data['sigma']), jnp.asarray(data['helicity']), jnp.asarray(data['varphi']), jnp.asarray(data['X1s']), jnp.asarray(data['X1c']), jnp.asarray(data['d_l_d_phi']), jnp.asarray(data['d_d_varphi']), jnp.asarray(data['sG']), jnp.asarray(data['spsi']), jnp.asarray(data['B0']), jnp.asarray(data['G0']), jnp.asarray(data['iotaN']), jnp.asarray(data['torsion']), jnp.asarray(data['abs_G0_over_B0']), jnp.asarray(data['B2s']), jnp.asarray(data['B2c']), jnp.asarray(data['p2']), jnp.asarray(data['I2']), data['nphi'], data['order'], jnp.asarray(data['iota']), jnp.asarray(data['d_l_d_varphi']), jnp.asarray(data['tangent_cylindrical']), jnp.asarray(data['normal_cylindrical']), jnp.asarray(data['binormal_cylindrical']), jnp.asarray(data['d_phi']), jnp.asarray(data['axis_length']))

#get correct results 
results = np.load('qsc/tests/results.npy', allow_pickle=True).item()


bool = jnp.allclose(r1_results[0][0], results['Y1s'])
bool = jnp.allclose(r1_results[0][1], results['Y1c'])
bool = jnp.allclose(r1_results[0][2], results['X1s_untwisted'])
bool = jnp.allclose(r1_results[0][3], results['X1c_untwisted'])
bool = jnp.allclose(r1_results[0][4], results['Y1s_untwisted'])
bool = jnp.allclose(r1_results[0][5], results['Y1c_untwisted'])
bool = jnp.allclose(r1_results[0][6], results['elongation'])
bool = jnp.allclose(r1_results[0][7], results['mean_elongation'])
bool = jnp.allclose(r1_results[0][8], results['max_elongation'])
bool = jnp.allclose(r1_results[0][9], results['d_X1c_d_varphi'])
bool = jnp.allclose(r1_results[0][10], results['d_X1s_d_varphi'])
bool = jnp.allclose(r1_results[0][11], results['d_Y1s_d_varphi'])
bool = jnp.allclose(r1_results[0][12], results['d_Y1c_d_varphi'])
#print(r1_results[1][0][0])
#print(results['grad_B_tensor'].items())
#bool = jnp.allclose(r1_results[1][0][0], results['grad_B_tensor']) // cant get data because encapsulated in struct class
bool = jnp.allclose(r1_results[1][0][1], results['grad_grad_B_inverse_scale_length_vs_varphi'])
bool = jnp.allclose(r1_results[1][0][2], results['L_grad_grad_B'])
bool = jnp.allclose(r1_results[1][0][3], results['grad_grad_B_inverse_scale_length'])
bool = jnp.allclose(r2_results[2][0], results['N_helicity'])
bool = jnp.allclose(r2_results[2][1], results['G2'])
bool = jnp.allclose(r2_results[2][2], results['d_curvature_d_varphi'])
bool = jnp.allclose(r2_results[2][3], results['d_torsion_d_varphi'])
bool = jnp.allclose(r2_results[2][4], results['d_X20_d_varphi'])
bool = jnp.allclose(r2_results[2][5], results['d_X2s_d_varphi'])
bool = jnp.allclose(r2_results[2][6], results['d_X2c_d_varphi'])
bool = jnp.allclose(r2_results[2][7], results['d_Y20_d_varphi'])
bool = jnp.allclose(r2_results[2][8], results['d_Y2s_d_varphi'])
bool = jnp.allclose(r2_results[2][9], results['d_Y2c_d_varphi'])
bool = jnp.allclose(r2_results[2][10], results['d_Z20_d_varphi'])
bool = jnp.allclose(r2_results[2][11], results['d_Z2s_d_varphi'])
bool = jnp.allclose(r2_results[2][12], results['d_Z2c_d_varphi'])
bool = jnp.allclose(r2_results[2][13], results['d2_X1c_d_varphi2'])
bool = jnp.allclose(r2_results[2][14], results['d2_Y1c_d_varphi2'])
bool = jnp.allclose(r2_results[2][15], results['d2_Y1s_d_varphi2'])
bool = jnp.allclose(r2_results[2][16], results['V1'])
bool = jnp.allclose(r2_results[2][17], results['V2'])
bool = jnp.allclose(r2_results[2][18], results['V3'])
bool = jnp.allclose(r2_results[2][19], results['X20'])
bool = jnp.allclose(r2_results[2][20], results['X2s'])
bool = jnp.allclose(r2_results[2][21], results['X2c'])
bool = jnp.allclose(r2_results[2][22], results['Y20'])
bool = jnp.allclose(r2_results[2][23], results['Y2s'])
bool = jnp.allclose(r2_results[2][24], results['Y2c'])
bool = jnp.allclose(r2_results[2][25], results['Z20'])
bool = jnp.allclose(r2_results[2][26], results['Z2s'])
bool = jnp.allclose(r2_results[2][27], results['Z2c'])
bool = jnp.allclose(r2_results[2][28], results['beta_1s'])
bool = jnp.allclose(r2_results[2][29], results['B20'])
bool = jnp.allclose(r2_results[2][30], results['X20_untwisted'])
bool = jnp.allclose(r2_results[2][31], results['X2s_untwisted'])
bool = jnp.allclose(r2_results[2][32], results['X2c_untwisted'])
bool = jnp.allclose(r2_results[2][33], results['Y20_untwisted'])
bool = jnp.allclose(r2_results[2][34], results['Y2s_untwisted'])
bool = jnp.allclose(r2_results[2][35], results['Y2c_untwisted'])
bool = jnp.allclose(r2_results[2][36], results['Z20_untwisted'])
bool = jnp.allclose(r2_results[2][37], results['Z2s_untwisted'])
bool = jnp.allclose(r2_results[2][38], results['Z2c_untwisted'])
bool = jnp.allclose(r2_results[0][0], results['DGeod_times_r2']) #mercier
bool = jnp.allclose(r2_results[0][1], results['d2_volume_d_psi2'])
bool = jnp.allclose(r2_results[0][2], results['DWell_times_r2'])
bool = jnp.allclose(r2_results[0][3], results['DMerc_times_r2'])
bool = jnp.allclose(r2_results[1][0], results['grad_grad_B']) # grad grad b
bool = jnp.allclose(r2_results[1][1], results['grad_grad_B_inverse_scale_length_vs_varphi'])
bool = jnp.allclose(r2_results[1][2], results['L_grad_grad_B'])
bool = jnp.allclose(r2_results[1][3], results['grad_grad_B_inverse_scale_length'])
bool = jnp.allclose(r2_results[3][0], results['r_singularity_vs_varphi']) # r singularity results
bool = jnp.allclose(r2_results[3][1], results['inv_r_singularity_vs_varphi']) 
bool = jnp.allclose(r2_results[3][2], results['r_singularity_basic_vs_varphi']) 
bool = jnp.allclose(r2_results[3][3], results['r_singularity']) 
bool = jnp.allclose(r2_results[3][4], results['r_singularity_theta_vs_varphi']) 
bool = jnp.allclose(r2_results[3][5], results['r_singularity_residual_sqnorm']) 
bool = jnp.allclose(r3_results[0][1], results['X3c1']) # r3
bool = jnp.allclose(r3_results[0][2], results['Y3c1'])
bool = jnp.allclose(r3_results[0][3], results['Y3s1'])
bool = jnp.allclose(r3_results[0][4], results['X3s1'])
bool = jnp.allclose(r3_results[0][5], results['Z3c1'])
bool = jnp.allclose(r3_results[0][6], results['Z3s1'])
bool = jnp.allclose(r3_results[0][7], results['X3c3'])
bool = jnp.allclose(r3_results[0][8], results['X3s3'])
bool = jnp.allclose(r3_results[0][9], results['Y3c3'])
bool = jnp.allclose(r3_results[0][10], results['Y3s3'])
bool = jnp.allclose(r3_results[0][11], results['Z3c3'])
bool = jnp.allclose(r3_results[0][12], results['Z3s3'])
bool = jnp.allclose(r3_results[0][13], results['d_X3c1_d_varphi'])
bool = jnp.allclose(r3_results[0][14], results['d_Y3c1_d_varphi'])
bool = jnp.allclose(r3_results[0][15], results['d_Y3s1_d_varphi'])
bool = jnp.allclose(r3_results[0][16], results['flux_constraint_coefficient'])
bool = jnp.allclose(r3_results[0][17], results['B0_order_a_squared_to_cancel'])
bool = jnp.allclose(r3_results[0][18], results['X3c1_untwisted'])
bool = jnp.allclose(r3_results[0][19], results['Y3c1_untwisted'])
bool = jnp.allclose(r3_results[0][20], results['Y3s1_untwisted'])
bool = jnp.allclose(r3_results[0][21], results['X3s1_untwisted'])
bool = jnp.allclose(r3_results[0][22], results['X3s3_untwisted'])
bool = jnp.allclose(r3_results[0][23], results['X3c3_untwisted'])
bool = jnp.allclose(r3_results[0][24], results['Y3c3_untwisted'])
bool = jnp.allclose(r3_results[0][25], results['Y3s3_untwisted'])
bool = jnp.allclose(r3_results[0][26], results['Z3s1_untwisted'])
bool = jnp.allclose(r3_results[0][27], results['Z3s3_untwisted'])
bool = jnp.allclose(r3_results[0][28], results['Z3c1_untwisted'])
bool = jnp.allclose(r3_results[0][29], results['Z3c3_untwisted'])



print(bool)  # all values within calculate are computed correctly as of may 19 2025


#tests during jit compilation 







