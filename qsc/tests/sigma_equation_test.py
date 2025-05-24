import numpy as np
import jax.numpy as jnp
import jax 
from qsc.calculate_r1 import new_solve_sigma_equation

#get inputs from file storing some inputs calculated in the original qsc
data = np.load('qsc/tests/values.npy', allow_pickle=True).item()

sigma, iota, iotaN = new_solve_sigma_equation(nphi=data['nphi'], sigma0=data['sigma0'], helicity=data['helicity'], nfp=data['nfp'], d_d_varphi=data['d_d_varphi'], etabar_squared_over_curvature_squared=data['etabar_squared_over_curvature_squared'], spsi=data['spsi'], torsion=data['torsion'], I2=data['I2'], B0=data['B0'], G0=data['G0'])

# get results 
results = np.load('qsc/tests/results.npy', allow_pickle=True).item()



bool = jnp.allclose(results['sigma'], sigma) 
bool2 = jnp.allclose(results['iota'], iota) 
bool3 = jnp.allclose(results['iotaN'] , iotaN)

print(sigma)
print(iota)
print(iotaN)

print(bool)
print(bool2)
print(bool3)

"""
# testing function in jitted context 
jitted_function = jax.jit(new_solve_sigma_equation, static_argnames=['nphi' , 'sigma0'])

sigma, iota, iotaN = jitted_function(nphi=data['nphi'], sigma0=data['sigma0'], helicity=data['helicity'], nfp=data['nfp'], d_d_varphi=data['d_d_varphi'], etabar_squared_over_curvature_squared=data['etabar_squared_over_curvature_squared'], spsi=data['spsi'], torsion=data['torsion'], I2=data['I2'], B0=data['B0'], G0=data['G0'])

bool = jnp.allclose(results['sigma'], sigma) & \
jnp.allclose(results['iota'], iota) & \
jnp.allclose(results['iotaN'] , iotaN)

print(f'jitted test pass: {bool}')
"""