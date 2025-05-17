import numpy as np
import jax.numpy as jnp

#get inputs from file storing some inputs calculated in the original qsc
data = np.load('values.npy', allow_pickle=True).item()
#data_jax = {k: jnp.asarray(v) for k, v in data.items()}

# run calculate with these inputs 
from qsc.qsc_method import calculate
r1_results, r2_results, r3_result = calculate(jnp.asarray(data['nfp']), jnp.asarray(data['etabar']), jnp.asarray(data['curvature']), jnp.asarray(data['sigma']), jnp.asarray(data['helicity']), jnp.asarray(data['varphi']), jnp.asarray(data['X1s']), jnp.asarray(data['X1c']), jnp.asarray(data['d_l_d_phi']), jnp.asarray(data['d_d_varphi']), jnp.asarray(data['sG']), jnp.asarray(data['spsi']), jnp.asarray(data['B0']), jnp.asarray(data['G0']), jnp.asarray(data['iotaN']), jnp.asarray(data['torsion']), jnp.asarray(data['abs_G0_over_B0']), jnp.asarray(data['B2s']), jnp.asarray(data['B2c']), jnp.asarray(data['p2']), jnp.asarray(data['I2']), data['nphi'], data['order'], jnp.asarray(data['iota']), jnp.asarray(data['d_l_d_varphi']), jnp.asarray(data['tangent_cylindrical']), jnp.asarray(data['normal_cylindrical']), jnp.asarray(data['binormal_cylindrical']), jnp.asarray(data['d_phi']), jnp.asarray(data['axis_length']))


#get