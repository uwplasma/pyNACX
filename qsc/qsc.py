"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import logging
import numpy as np
from scipy.io import netcdf
import jax.numpy as jnp
import jax
from qsc.calculate_r3 import calc_r3_new
from qsc.init_axis import init_axis
from qsc.calculate_r1 import new_solve_sigma_equation, r1_diagnostics
from qsc.calculate_r2 import calc_r2_new
from qsc.grad_B_tensor import calculate_grad_B_tensor

#from numba import jit

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Qsc():
    """
    This is the main class for representing the quasisymmetric
    stellarator construction.
    """
    
    # Import methods that are defined in separate files:
    from .init_axis import init_axis, convert_to_spline
    from .calculate_r1 import _residual, _jacobian, solve_sigma_equation, \
        _determine_helicity, r1_diagnostics
    from .grad_B_tensor import calculate_grad_B_tensor, calculate_grad_grad_B_tensor, \
        Bfield_cylindrical, Bfield_cartesian, grad_B_tensor_cartesian, \
        grad_grad_B_tensor_cylindrical, grad_grad_B_tensor_cartesian
    from .calculate_r2 import calculate_r2
    from .calculate_r3 import calculate_r3, calculate_shear
    # import mercier
    from .r_singularity import calculate_r_singularity
    from .plot import plot, plot_boundary, get_boundary, B_fieldline, B_contour, plot_axis, flux_tube
    from .Frenet_to_cylindrical import Frenet_to_cylindrical, to_RZ
    from .to_vmec import to_vmec
    from .util import B_mag
    from .configurations import from_paper, configurations
    
    def __init__(self, rc, zs, rs=[], zc=[], nfp=1, etabar=1., sigma0=0., B0=1.,
                 I2=0., sG=1, spsi=1, nphi=61, B2s=0., B2c=0., p2=0., order="r1"):
        """
        Create a quasisymmetric stellarator.
        """
        # First, force {rc, zs, rs, zc} to have the same length, for
        # simplicity.
        
        
        find_max = jnp.array([len(rc), len(zs), len(rs), len(zc)])
        nfourier = jnp.max(find_max)
        print(f"nfourier: {nfourier}")
        self.nfourier = nfourier
        nfourier = nfourier 
        self.rc = jnp.zeros(nfourier)
        self.zs = jnp.zeros(nfourier)
        self.rs = jnp.zeros(nfourier)
        self.zc = jnp.zeros(nfourier)

        self.rc = self.rc.at[:len(rc)].set(rc)
        self.zs = self.zs.at[:len(zs)].set(zs)
        self.rs = self.rs.at[:len(rs)].set(rs)
        self.zc = self.zc.at[:len(zc)].set(zc)
        
        rc = self.rc
        zs = self.zs
        rs = self.rs
        zc = self.zc
        
        # Force nphi to be odd:
        if jnp.mod(nphi, 2) == 0:
            nphi += 1

        if sG != 1 and sG != -1:
            raise ValueError('sG must be +1 or -1')
        
        if spsi != 1 and spsi != -1:
            raise ValueError('spsi must be +1 or -1')

        self.nfp = nfp
        self.etabar = etabar
        self.sigma0 = sigma0
        self.B0 = B0
        self.I2 = I2
        self.sG = sG
        self.spsi = spsi
        self.nphi = nphi
        self.B2s = B2s
        self.B2c = B2c
        self.p2 = p2
        self.order = order
        self.min_R0_threshold = 0.3
        #self._set_names()
      
        self.pre_calculations() #run initial calculations that rely on self 
        
        self.initialize()
        
    def change_nfourier(self, nfourier_new):
        """
        Resize the arrays of Fourier amplitudes. You can either increase
        or decrease nfourier.
        """
        rc_old = self.rc
        rs_old = self.rs
        zc_old = self.zc
        zs_old = self.zs
        index = jnp.min((self.nfourier, nfourier_new))
        self.rc = jnp.zeros(nfourier_new)
        self.rs = jnp.zeros(nfourier_new)
        self.zc = jnp.zeros(nfourier_new)
        self.zs = jnp.zeros(nfourier_new)
        
        self.rc = self.rc.at[:index].set(rc_old[:index])
        self.rs = self.rs.at[:index].set(rs_old[:index])
        self.zc = self.zc.at[:index].set(zc_old[:index])
        self.zs = self.zs.ad[:index].set(zs_old[:index])
       
        nfourier_old = self.nfourier
        self.nfourier = nfourier_new 
        self._set_names()
        # No need to recalculate if we increased the Fourier
        # resolution, only if we decreased it.
        if nfourier_new < nfourier_old:
            self.calculate()
            
    def pre_calculations(self):
        helicity,\
        normal_cylindrical, \
        etabar_squared_over_curvature_squared, \
        varphi, \
        d_d_phi, \
        d_varphi_d_phi, \
        d_d_varphi, \
        phi, \
        abs_G0_over_B0, \
        d_phi, \
        R0, \
        Z0, \
        R0p, \
        Z0p, \
        R0pp, \
        Z0pp, \
        R0ppp, \
        Z0ppp, \
        G0, \
        d_l_d_phi, \
        axis_length, \
        curvature, \
        torsion, \
        X1s, \
        X1c, \
        min_R0, \
        tangent_cylindrical, \
        normal_cylindrical, \
        binormal_cylindrical, \
        Bbar, \
        abs_G0_over_B0 = self.init_axis(self.nphi, self.nfp, self.rc, self.rs, self.zc, self.zs, self.nfourier, self.sG, self.B0, self.etabar, self.spsi, self.sigma0, self.order, self.B2s)
        
        self.helicity = helicity
        self.normal_cylindrical = normal_cylindrical
        self.etabar_squared_over_curvature_squared = etabar_squared_over_curvature_squared
        self.varphi = varphi
        self.d_d_phi = d_d_phi
        self.d_varphi_d_phi = d_varphi_d_phi
        self.d_d_varphi = d_d_varphi
        self.phi = phi
        self.abs_G0_over_B0 = abs_G0_over_B0
        self.d_phi = d_phi
        self.R0 = R0
        self.Z0 = Z0
        self.R0p = R0p
        self.Z0p = Z0p
        self.R0pp = R0pp    
        self.Z0pp = Z0pp
        self.R0ppp = R0ppp
        self.Z0ppp = Z0ppp
        self.G0 = G0
        self.d_l_d_phi = d_l_d_phi
        self.axis_length = axis_length
        self.curvature = curvature
        self.torsion = torsion
        self.X1s = X1s
        self.X1c = X1c
        self.min_R0 = min_R0
        self.tangent_cylindrical = tangent_cylindrical
        self.normal_cylindrical = normal_cylindrical
        self.binormal_cylindrical = binormal_cylindrical
        self.Bbar = Bbar
        self.d_l_d_varphi = abs_G0_over_B0
        
        
        print("\nInit axis completed...")
        
        sigma, iota, iotaN = new_solve_sigma_equation(self.nphi, self.sigma0, self.helicity, self.nfp, self.d_d_varphi, self.etabar_squared_over_curvature_squared, self.spsi, self.torsion, self.I2, self.B0, self.G0)
        self.sigma = sigma
        self.iota = iota
        self.iotaN = iotaN
        print("\nSigma equation solved...")
        
    def initialize(self): 
        
        print(f"nphi type: {type(self.nphi)}, nphi: {self.nphi}")
       
       
        print(type(self.nfp))
        print(type(self.etabar))
        print(type(self.curvature)) 
        print(type(self.sigma))
        print(type(self.helicity))
        print(type(self.varphi))
        print(type(self.X1s))
        print(type(self.X1c))
        print(type(self.d_l_d_phi))
        print(type(self.d_d_varphi))
        print(type(self.sG))
        print(type(self.spsi))
        print(type(self.B0))
        print(type(self.G0))
        print(type(self.iotaN))
        print(type(self.torsion))
        print(type(self.abs_G0_over_B0))
        print(type(self.B2s))
        print(type(self.B2c))
        print(type(self.p2))
        print(type(self.I2))
        print(type(self.nphi))
        print(type(self.order))
        print(type(self.iota))
        print(type(self.d_l_d_varphi))
        print(type(self.tangent_cylindrical))
        print(type(self.normal_cylindrical))
        print(type(self.binormal_cylindrical))
        print(type(self.d_phi))
        print(type(self.axis_length))

        
        jited_funtion = jax.jit(Qsc.calculate, static_argnames= ['order', 'nphi'])
        
        calculation_results = jited_funtion(nfp = self.nfp, etabar = self.etabar, curvature = self.curvature, sigma = self.sigma, helicity = self.helicity, varphi = self.varphi, X1s = self.X1s, X1c = self.X1c, d_l_d_phi = self.d_l_d_phi, d_d_varphi = self.d_d_varphi, sG = self.sG, spsi = self.spsi, B0 = self.B0, G0 = self.G0, iotaN = self.iotaN, torsion = self.torsion, abs_G0_over_B0 = self.abs_G0_over_B0, B2s = self.B2s, B2c = self.B2c, p2 = self.p2, I2 = self.I2, nphi = self.nphi, order = self.order, iota = self.iota, d_l_d_varphi = self.d_l_d_varphi, tangent_cylindrical = self.tangent_cylindrical, normal_cylindrical = self.normal_cylindrical, binormal_cylindrical = self.binormal_cylindrical, d_phi = self.d_phi, axis_length = self.axis_length)
        
        self.Y1s, self.Y1c, self.X1s_untwisted, self.X1c_untwisted, self.Y1s_untwisted, self.Y1c_untwisted, self.elongation, self.mean_elongation, self.max_elongation, self.d_X1c_d_varphi, self.d_X1s_d_varphi, self.d_Y1s_d_varphi, self.d_Y1c_d_varphi = calculation_results[0][0]
        
        self.grad_B_tensor, self.grad_B_tensor_cylindrical, self.grad_B_colon_grad_B, self.L_grad_B, self.inv_L_grad_B, self.min_L_grad_B = calculation_results[0][1]
        
        self.N_helicity, self.G2, self.d_curvature_d_varphi, self.d_torsion_d_varphi, self.d_X20_d_varphi, self.d_X2s_d_varphi, self.d_X2c_d_varphi, self.d_Y20_d_varphi, self.d_Y2s_d_varphi, self.d_Y2c_d_varphi, self.d_Z20_d_varphi, self.d_Z2s_d_varphi, self.d_Z2c_d_varphi, self.d2_X1c_d_varphi2, self.d2_Y1c_d_varphi2, self.d2_Y1s_d_varphi2, self.V1, self.V2, self.V3, self.X20, self.X2s, self.X2c, self.Y20, self.Y2s, self.Y2c, self.Z20, self.Z2s, self.Z2c, self.beta_1s, self.B20, self.X20_untwisted, self.X2s_untwisted, self.X2c_untwisted, self.Y20_untwisted, self.Y2s_untwisted, self.Y2c_untwisted, self.Z20_untwisted, self.Z2s_untwisted, self.Z2c_untwisted = calculation_results[1][2]
        
        self.DGeod_times_r2, self.d2_volume_d_psi2, self.DWell_times_r2, self.DMerc_times_r2 = calculation_results[1][0]
        
        self.grad_grad_B, self.grad_grad_B_inverse_scale_length_vs_varphi, self.L_grad_grad_B, self.grad_grad_B_inverse_scale_length = calculation_results[1][1]
        
        self.X3c1, self.Y3c1, self.Y3s1, self.X3s1, self.Z3c1, self.Z3s1, self.X3c3, self.X3s3, self.Y3c3, self.Y3s3, self.Z3c3, self.Z3s3, self.d_X3c1_d_varphi, self.d_Y3c1_d_varphi, self.d_Y3s1_d_varphi, self.flux_constraint_coefficient, self.B0_order_a_squared_to_cancel, self.X3c1_untwisted, self.Y3c1_untwisted, self.Y3s1_untwisted, self.X3s1_untwisted, self.X3s3_untwisted, self.X3c3_untwisted, self.Y3c3_untwisted, self.Y3s3_untwisted, self.Z3s1_untwisted, self.Z3s3_untwisted, self.Z3c1_untwisted, self.Z3c3_untwisted = calculation_results[2]
        
    def calculate(nfp, etabar, curvature, sigma, helicity, varphi, X1s, X1c, d_l_d_phi, d_d_varphi, sG, spsi, B0, G0, iotaN, torsion, abs_G0_over_B0, B2s, B2c, p2, I2, nphi, order, iota, d_l_d_varphi, tangent_cylindrical, normal_cylindrical, binormal_cylindrical, d_phi, axis_length):
        
        """
        A jax compatible driver of main calculations.
        """
        print("\nCalculating R1...")
        r1_results = r1_diagnostics(nfp, etabar, sG, spsi, curvature, sigma, helicity, varphi, X1s, X1c, d_l_d_phi, d_d_varphi, B0, d_l_d_varphi, tangent_cylindrical, normal_cylindrical, binormal_cylindrical, iotaN, torsion)


        Y1s = r1_results[0][0]
        Y1c = r1_results[0][1]
        d_X1c_d_varphi = r1_results[0][-4]
        d_Y1s_d_varphi = r1_results[0][-2]
        d_Y1c_d_varphi = r1_results[0][-1]
        r2_results = jax.lax.cond(order != 'r1',
                          lambda _:  calc_r2_new(X1c, Y1c, Y1s, B0 / jnp.abs(G0), d_d_varphi, iotaN, torsion, abs_G0_over_B0, B2s, B0, curvature, etabar, B2c, spsi, sG, p2, sigma, I2/B0, nphi, d_l_d_phi, helicity, nfp, G0, iota, I2, varphi, d_X1c_d_varphi, d_Y1c_d_varphi, d_Y1s_d_varphi, d_phi, axis_length),
                          lambda _: tuple(
                            tuple(jax.numpy.zeros_like(r) for r in inner_tuple)  # Apply zeros_like to each element of the inner tuple
                            for inner_tuple in calc_r2_new(
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
                         lambda _: calc_r3_new(B0, G0, X20, Y1c, X2c, X2s, etabar*B0, X1c, X1s, Y1s, I2, iotaN, B20, Y20, Y2c, Y2s, Z20, abs_G0_over_B0, Z2c, Z2s, torsion, d_X1c_d_varphi, d_Y1c_d_varphi, d_d_varphi, spsi, p2, curvature, d_Z20_d_varphi, sG, G2, N_helicity, helicity, nfp, varphi),
                         lambda _: tuple(jnp.zeros_like(r) for r in calc_r3_new(B0, G0, X20, Y1c, X2c, X2s, etabar*B0, X1c, X1s, Y1s, I2, iotaN, B20, Y20, Y2c, Y2s, Z20, abs_G0_over_B0, Z2c, Z2s, torsion, d_X1c_d_varphi, d_Y1c_d_varphi, d_d_varphi, spsi, p2, curvature, d_Z20_d_varphi, sG, G2, N_helicity, helicity, nfp, varphi)),
                         operand=None)
        
        return r1_results, r2_results, r3_result; 
    
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
        
