import numpy as np
import jax.numpy as jnp
import jax 
from qsc.init_axis import init_axis

# get inputs from file sotring some inputs calculated in the original qsc
data = np.load('qsc/tests/values.npy', allow_pickle=True).item()

# get results
results = np.load('qsc/tests/results.npy', allow_pickle=True).item()

# run method 
values = init_axis(nphi=data['nphi'], nfp=data['nfp'], rc=data['rc'], rs=data['rs'], zc=data['zc'], zs=data['zs'], nfourier=data['nfourier'], sG=data['sG'], B0=data['B0'], etabar=data['etabar'], spsi=data['spsi'], sigma0=data['sigma0'], order=data['order'], B2s=data['B2s'])

bool =  jnp.allclose(values[0], results['helicity']) & \
jnp.allclose(values[1], results['normal_cylindrical']) & \
jnp.allclose(values[2], results['etabar_squared_over_curvature_squared']) & \
jnp.allclose(values[3], results['varphi']) 
jnp.allclose(values[4], results['d_d_phi']) & \
jnp.allclose(values[5], results['d_varphi_d_phi']) & \
jnp.allclose(values[6], results['d_d_varphi']) & \
jnp.allclose(values[7], results['phi']) & \
jnp.allclose(values[8], results['abs_G0_over_B0']) & \
jnp.allclose(values[9], results['d_phi']) & \
jnp.allclose(values[10], results['R0']) & \
jnp.allclose(values[11], results['Z0']) & \
jnp.allclose(values[12], results['R0p']) & \
jnp.allclose(values[13], results['Z0p']) & \
jnp.allclose(values[14], results['R0pp']) & \
jnp.allclose(values[15], results['Z0pp']) & \
jnp.allclose(values[16], results['R0ppp']) & \
jnp.allclose(values[17], results['Z0ppp']) & \
jnp.allclose(values[18], results['G0']) & \
jnp.allclose(values[19], results['d_l_d_phi']) & \
jnp.allclose(values[20], results['axis_length']) & \
jnp.allclose(values[21], results['curvature']) & \
jnp.allclose(values[22], results['torsion']) & \
jnp.allclose(values[23], results['X1s']) & \
jnp.allclose(values[24], results['X1c']) & \
jnp.allclose(values[25], results['min_R0']) & \
jnp.allclose(values[26], results['tangent_cylindrical']) & \
jnp.allclose(values[27], results['normal_cylindrical']) & \
jnp.allclose(values[28], results['binormal_cylindrical']) & \
jnp.allclose(values[29], results['Bbar']) & \
jnp.allclose(values[30], results['abs_G0_over_B0']) & \
jnp.allclose(values[31], results['lasym']) & \
jnp.allclose(values[32], results['R0_func']) & \
jnp.allclose(values[33], results['Z0_func']) & \
jnp.allclose(values[34], results['normal_R_spline']) & \
jnp.allclose(values[35], results['normal_phi_spline']) & \
jnp.allclose(values[36], results['normal_z_spline']) & \
jnp.allclose(values[37], results['binormal_R_spline']) & \
jnp.allclose(values[38], results['binormal_phi_spline']) & \
jnp.allclose(values[39], results['binormal_z_spline']) & \
jnp.allclose(values[40], results['tangent_R_spline']) & \
jnp.allclose(values[41], results['tangent_phi_spline']) & \
jnp.allclose(values[42], results['tangent_z_spline']) & \
jnp.allclose(values[43], results['nu_spline']) 

print(f'un-jitted test: {bool}')

# jit the function 
jitted_function = jax.jit(init_axis)

# run method 
values = jitted_function(nphi=data['nphi'], nfp=data['nfp'], rc=data['rc'], rs=data['rs'], zc=data['zc'], zs=data['zs'], nfourier=data['nfourier'], sG=data['sG'], B0=data['B0'], etabar=data['etabar'], spsi=data['spsi'], sigma0=data['sigma0'], order=data['order'], B2s=data['B2s'])

bool =  jnp.allclose(values[0], results['helicity']) & \
jnp.allclose(values[1], results['normal_cylindrical']) & \
jnp.allclose(values[2], results['etabar_squared_over_curvature_squared']) & \
jnp.allclose(values[3], results['varphi']) 
jnp.allclose(values[4], results['d_d_phi']) & \
jnp.allclose(values[5], results['d_varphi_d_phi']) & \
jnp.allclose(values[6], results['d_d_varphi']) & \
jnp.allclose(values[7], results['phi']) & \
jnp.allclose(values[8], results['abs_G0_over_B0']) & \
jnp.allclose(values[9], results['d_phi']) & \
jnp.allclose(values[10], results['R0']) & \
jnp.allclose(values[11], results['Z0']) & \
jnp.allclose(values[12], results['R0p']) & \
jnp.allclose(values[13], results['Z0p']) & \
jnp.allclose(values[14], results['R0pp']) & \
jnp.allclose(values[15], results['Z0pp']) & \
jnp.allclose(values[16], results['R0ppp']) & \
jnp.allclose(values[17], results['Z0ppp']) & \
jnp.allclose(values[18], results['G0']) & \
jnp.allclose(values[19], results['d_l_d_phi']) & \
jnp.allclose(values[20], results['axis_length']) & \
jnp.allclose(values[21], results['curvature']) & \
jnp.allclose(values[22], results['torsion']) & \
jnp.allclose(values[23], results['X1s']) & \
jnp.allclose(values[24], results['X1c']) & \
jnp.allclose(values[25], results['min_R0']) & \
jnp.allclose(values[26], results['tangent_cylindrical']) & \
jnp.allclose(values[27], results['normal_cylindrical']) & \
jnp.allclose(values[28], results['binormal_cylindrical']) & \
jnp.allclose(values[29], results['Bbar']) & \
jnp.allclose(values[30], results['abs_G0_over_B0']) & \
jnp.allclose(values[31], results['lasym']) & \
jnp.allclose(values[32], results['R0_func']) & \
jnp.allclose(values[33], results['Z0_func']) & \
jnp.allclose(values[34], results['normal_R_spline']) & \
jnp.allclose(values[35], results['normal_phi_spline']) & \
jnp.allclose(values[36], results['normal_z_spline']) & \
jnp.allclose(values[37], results['binormal_R_spline']) & \
jnp.allclose(values[38], results['binormal_phi_spline']) & \
jnp.allclose(values[39], results['binormal_z_spline']) & \
jnp.allclose(values[40], results['tangent_R_spline']) & \
jnp.allclose(values[41], results['tangent_phi_spline']) & \
jnp.allclose(values[42], results['tangent_z_spline']) & \
jnp.allclose(values[43], results['nu_spline']) 

print(f'jitted test: {bool}')
