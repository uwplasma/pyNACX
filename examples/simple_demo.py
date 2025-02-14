#!/usr/bin/env python3
import sys
sys.path.append('/Users/z/Documents/GitHub/pyNACX')


import numpy as np
import jax.numpy as jnp
from qsc import Qsc
from qsc import derive_r3 , calculate_r1

"""
rc = []
zs = []
rs=[]
zc=[]
nfp=1
etabar=1.
sigma0=0.
B0=1.
I2=0. 
sG=1
spsi=1 
nphi=61
B2s=0. 
B2c=0. 
p2=0.
order="r1"

rc=[1, 0.09]
zs=[0, -0.09]
nfp=2
etabar=0.95
I2=0.9
order='r2'
B2c=-0.7 
p2=-600000.

find_max = jnp.array([len(rc), len(zs), len(rs), len(zc)])
nfourier = jnp.max(find_max)

rc = jnp.zeros(nfourier)
zs = jnp.zeros(nfourier)
rs = jnp.zeros(nfourier)
zc = jnp.zeros(nfourier)

rc = rc.at[:len(rc)].set(rc)
zs = zs.at[:len(zs)].set(zs)
rs = rs.at[:len(rs)].set(rs)
zc = zc.at[:len(zc)].set(zc)

_residual = calculate_r1._new_residual
_jacobian = calculate_r1._jacobian

print(f" {derive_r3.derive_X3s1(_residual, _jacobian, rc, zs, rs, zc, nfp, etabar, sigma0, I2, B0, sG, spsi, nphi, B2s, p2, B2c)}")
"""

print("Running pyNACX...")
stel = Qsc(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r1', B2c=-0.7, p2=-600000.)

print("pyNACX finished")
print(stel.iota) # Rotational transform on-axis for this configuration
#print(stel.d2_volume_d_psi2) # Magnetic well V''(psi) Semo: this is calculated in "mercier.py" in pyqsc which does not exist in pyNACX
#print(stel.DMerc_times_r2) # Mercier criterion parameter DMerc multiplied by r^2 Semo: this is calculated in "mercier.py" in pyqsc which does not exist in pyNACX
print(stel.min_L_grad_B) # Scale length associated with the grad grad B tensor
#print(stel.grad_grad_B_inverse_scale_length) # Scale length associated with the grad grad B tensor Semo: grad_grad_B_inverse_scale_length is not calculted in r1
print("plotting...")
stel.plot_boundary() # Plot the flux surface shape at the default radius r=1
stel.plot() # Plot relevant near axis parameters
