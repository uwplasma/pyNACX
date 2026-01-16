"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import logging
import numpy as np
from scipy.io import netcdf
import jax.numpy as jnp
import jax
from calculate_r3 import calc_r3
from qsc.init_axis import init_axis
from qsc.calculate_r1 import solve_sigma_equation, r1_diagnostics
from qsc.calculate_r2 import calc_r2
from qsc.grad_B_tensor import calculate_grad_B_tensor

#from numba import jit

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
        
def calculate(nfp, etabar, curvature, sigma, helicity, varphi, X1s, X1c, d_l_d_phi, d_d_varphi, sG, spsi, B0, G0, iotaN, torsion, abs_G0_over_B0, B2s, B2c, p2, I2, nphi, order, iota, d_l_d_varphi, tangent_cylindrical, normal_cylindrical, binormal_cylindrical, d_phi, axis_length):    
    """
    A jax compatible driver of main calculations.
    """
    print("\nCalculating R1...")
    r1_results = r1_diagnostics(nfp, etabar, sG, spsi, curvature, sigma, helicity, varphi, X1s, X1c, d_l_d_phi, d_d_varphi, B0, d_l_d_varphi, tangent_cylindrical, normal_cylindrical, binormal_cylindrical, iotaN, torsion)
    Y1s = r1_results[0][0]
    Y1c = r1_results[0][1]
    d_X1c_d_varphi = r1_results[0][9]
    d_Y1s_d_varphi = r1_results[0][11]
    d_Y1c_d_varphi = r1_results[0][12]
    
    #calc_r2_new(X1c, Y1c, Y1s, B0 / jnp.abs(G0), d_d_varphi, iotaN, torsion, abs_G0_over_B0, B2s, B0, curvature, etabar, B2c, spsi, sG, p2, sigma, I2/B0, nphi, d_l_d_phi, helicity, nfp, G0, iota, I2, varphi, d_X1c_d_varphi, d_Y1c_d_varphi, d_Y1s_d_varphi, d_phi, axis_length)
    
    r2_results = jax.lax.cond(order != 'r1',
                        lambda _:  calc_r2(X1c, Y1c, Y1s, B0 / jnp.abs(G0), d_d_varphi, iotaN, torsion, abs_G0_over_B0, B2s, B0, curvature, etabar, B2c, spsi, sG, p2, sigma, I2/B0, nphi, d_l_d_phi, helicity, nfp, G0, iota, I2, varphi, d_X1c_d_varphi, d_Y1c_d_varphi, d_Y1s_d_varphi, d_phi, axis_length),
                        lambda _: tuple(
                        tuple(jax.numpy.zeros_like(r) for r in inner_tuple)  # Apply zeros_like to each element of the inner tuple
                        for inner_tuple in calc_r2(
                            X1c, Y1c, Y1s, B0 / jnp.abs(G0), d_d_varphi, iotaN, torsion, abs_G0_over_B0, 
                            B2s, B0, curvature, etabar, B2c, spsi, sG, p2, sigma, I2 / B0, nphi, 
                            d_l_d_phi, helicity, nfp, G0, iota, I2, varphi, d_X1c_d_varphi, 
                            d_Y1c_d_varphi, d_Y1s_d_varphi, d_phi, axis_length)
                        ),
                        operand=None)
    

    print("\nCalculating R2...")

    X20 = r2_results[2][20]
    X2c = r2_results[2][22]
    X2s = r2_results[2][21]
    B20 = r2_results[2][30]
    Y20 = r2_results[2][23]
    Y2c = r2_results[2][25]
    Y2s = r2_results[2][24]
    Z20 = r2_results[2][26]
    Z2c = r2_results[2][28]
    Z2s = r2_results[2][27]
    d_Z20_d_varphi = r2_results[2][10]
    G2 = r2_results[2][1]
    N_helicity  = r2_results[2][0]
    
    print("\nCalculating R3...")

    r3_result = jax.lax.cond(order == 'r3',
                        lambda _: calc_r3(B0, G0, X20, Y1c, X2c, X2s, etabar*B0, X1c, X1s, Y1s, I2, iotaN, B20, Y20, Y2c, Y2s, Z20, abs_G0_over_B0, Z2c, Z2s, torsion, d_X1c_d_varphi, d_Y1c_d_varphi, d_d_varphi, spsi, p2, curvature, d_Z20_d_varphi, sG, G2, N_helicity, helicity, nfp, varphi),
                        lambda _: tuple(jnp.zeros_like(r) for r in calc_r3(B0, G0, X20, Y1c, X2c, X2s, etabar*B0, X1c, X1s, Y1s, I2, iotaN, B20, Y20, Y2c, Y2s, Z20, abs_G0_over_B0, Z2c, Z2s, torsion, d_X1c_d_varphi, d_Y1c_d_varphi, d_d_varphi, spsi, p2, curvature, d_Z20_d_varphi, sG, G2, N_helicity, helicity, nfp, varphi)),
                        operand=None)
    
    return r1_results, r2_results, r3_result

def get_dofs(self):
    """
    Return a 1D numpy vector of all possible optimizable
    degrees-of-freedom, for simsopt.
    """
    return jnp.concatenate((self.rc, self.zs, self.rs, self.zc,
                            jnp.array([self.etabar, self.sigma0, self.B2s, self.B2c, self.p2, self.I2, self.B0])))

def set_dofs(self, x):
    """
    For interaction with simsopt, set the optimizable degrees of
    freedom from a 1D numpy vector.
    """
    assert len(x) == self.nfourier * 4 + 7
    
    self.rc = jnp.array(x[self.nfourier * 0 : self.nfourier * 1])
    self.zs = jnp.array(x[self.nfourier * 1 : self.nfourier * 2])
    self.rs = jnp.array(x[self.nfourier * 2 : self.nfourier * 3])
    self.zc = jnp.array(x[self.nfourier * 3 : self.nfourier * 4])

    self.etabar = x[self.nfourier * 4 + 0]
    self.sigma0 = x[self.nfourier * 4 + 1]
    self.B2s = x[self.nfourier * 4 + 2]
    self.B2c = x[self.nfourier * 4 + 3]
    self.p2 = x[self.nfourier * 4 + 4]
    self.I2 = x[self.nfourier * 4 + 5]
    self.B0 = x[self.nfourier * 4 + 6]
    self.calculate()
    logger.info('set_dofs called with x={}. Now iota={}, elongation={}'.format(x, self.iota, self.max_elongation))
    
def _set_names(self):
    """
    For simsopt, sets the list of names for each degree of freedom.
    """
    names = []
    names += ['rc({})'.format(j) for j in range(self.nfourier)]
    names += ['zs({})'.format(j) for j in range(self.nfourier)]
    names += ['rs({})'.format(j) for j in range(self.nfourier)]
    names += ['zc({})'.format(j) for j in range(self.nfourier)]
    names += ['etabar', 'sigma0', 'B2s', 'B2c', 'p2', 'I2', 'B0']
    self.names = names

@classmethod
def from_cxx(cls, filename):
    """
    Load a configuration from a ``qsc_out.<extension>.nc`` output file
    that was generated by the C++ version of QSC. Almost all the
    data will be taken from the output file, over-writing any
    calculations done in python when the new Qsc object is
    created.
    """
    def to_string(nc_str):
        """ Convert a string from the netcdf binary format to a python string. """
        temp = [c.decode('UTF-8') for c in nc_str]
        return (''.join(temp)).strip()
    
    f = netcdf.netcdf_file(filename, mmap=False)
    nfp = f.variables['nfp'][()]
    nphi = f.variables['nphi'][()]
    rc = f.variables['R0c'][()]
    rs = f.variables['R0s'][()]
    zc = f.variables['Z0c'][()]
    zs = f.variables['Z0s'][()]
    I2 = f.variables['I2'][()]
    B0 = f.variables['B0'][()]
    spsi = f.variables['spsi'][()]
    sG = f.variables['sG'][()]
    etabar = f.variables['eta_bar'][()]
    sigma0 = f.variables['sigma0'][()]
    order_r_option = to_string(f.variables['order_r_option'][()])
    if order_r_option == 'r2.1':
        order_r_option = 'r3'
    if order_r_option == 'r1':
        p2 = 0.0
        B2c = 0.0
        B2s = 0.0
    else:
        p2 = f.variables['p2'][()]
        B2c = f.variables['B2c'][()]
        B2s = f.variables['B2s'][()]

    q = cls(nfp=nfp, nphi=nphi, rc=rc, rs=rs, zc=zc, zs=zs,
            B0=B0, sG=sG, spsi=spsi,
            etabar=etabar, sigma0=sigma0, I2=I2, p2=p2, B2c=B2c, B2s=B2s, order=order_r_option)
    
    def read(name, cxx_name=None):
        if cxx_name is None: cxx_name = name
        setattr(q, name, f.variables[cxx_name][()])

    [read(v) for v in ['R0', 'Z0', 'R0p', 'Z0p', 'R0pp', 'Z0pp', 'R0ppp', 'Z0ppp',
                        'sigma', 'curvature', 'torsion', 'X1c', 'Y1c', 'Y1s', 'elongation']]
    if order_r_option != 'r1':
        [read(v) for v in ['X20', 'X2c', 'X2s', 'Y20', 'Y2c', 'Y2s', 'Z20', 'Z2c', 'Z2s', 'B20']]
        if order_r_option != 'r2':
            [read(v) for v in ['X3c1', 'Y3c1', 'Y3s1']]
                
    f.close()
    return q
    
def min_R0_penalty(self):
    """
    This function can be used in optimization to penalize situations
    in which min(R0) < min_R0_constraint.
    """
    return jnp.max((0, self.min_R0_threshold - self.min_R0)) ** 2
    
