#!/usr/bin/env python3

import unittest
import os
from scipy.io import netcdf_file
import numpy as np
import logging
from qsc.qsc import Qsc
from qsc.util import to_Fourier

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fortran_plot_single(filename, ntheta=150, nphi = 4):
    """
    Function to extract boundary arrays from the fortran files
    """
    abs_filename = os.path.join(os.path.dirname(__file__), filename)
    f = netcdf_file(abs_filename,mode='r',mmap=False)
    r = f.variables['r'][()]
    nfp = f.variables['nfp'][()]
    nphi_axis = f.variables['N_phi'][()]
    mpol = f.variables['mpol'][()]
    ntor = f.variables['ntor'][()]
    RBC = f.variables['RBC'][()]
    RBS = f.variables['RBS'][()]
    ZBC = f.variables['ZBC'][()]
    ZBS = f.variables['ZBS'][()]
    R0c = f.variables['R0c'][()]
    R0s = f.variables['R0s'][()]
    Z0c = f.variables['Z0c'][()]
    Z0s = f.variables['Z0s'][()]

    theta1D = np.linspace(0,2*np.pi,ntheta)
    phi1D = np.linspace(0,2*np.pi,nphi)
    phi2D,theta2D = np.meshgrid(phi1D,theta1D)

    R = np.zeros((ntheta,nphi))
    z = np.zeros((ntheta,nphi))
    for m in range(mpol+1):
        for jn in range(ntor*2+1):
            n = jn-ntor
            angle = m * theta2D - nfp * n * phi2D
            sinangle = np.sin(angle)
            cosangle = np.cos(angle)
            R += RBC[m,jn] * cosangle + RBS[m,jn] * sinangle
            z += ZBC[m,jn] * cosangle + ZBS[m,jn] * sinangle

    R0 = np.zeros(nphi)
    z0 = np.zeros(nphi)
    for n in range(len(R0c)):
        angle = nfp * n * phi1D
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        R0 += R0c[n] * cosangle + R0s[n] * sinangle
        z0 += Z0c[n] * cosangle + Z0s[n] * sinangle

    return R, z, R0, z0, r, mpol, ntor, nphi_axis

def compare_to_fortran(name, filename):
    """
    Compare output from this python code to the fortran code, for one
    of the example configurations from the papers.
    """
    # Add the directory of this file to the specified filename:
    abs_filename = os.path.join(os.path.dirname(__file__), filename)
    f      = netcdf_file(abs_filename, 'r')
    nphi   = f.variables['N_phi'][()]
    mpol   = f.variables['mpol'][()]
    ntor   = f.variables['ntor'][()]
    r      = f.variables['r'][()]
    ntheta = 20

    py = Qsc.from_paper(name, nphi=nphi, order='r3')
    logger.info('Comparing to fortran file ' + abs_filename)

    def compare_field(fortran_name, py_field, rtol=1e-9, atol=1e-9):
        fortran_field = f.variables[fortran_name][()]
        logger.info('max difference in {}: {}'.format(fortran_name, np.max(np.abs(fortran_field - py_field))))
        np.testing.assert_allclose(fortran_field, py_field, rtol=rtol, atol=atol)

    compare_field('iota', py.iota)
    compare_field('curvature', py.curvature)
    compare_field('torsion', py.torsion)
    compare_field('sigma', py.sigma)
    compare_field('modBinv_sqrt_half_grad_B_colon_grad_B', 1 / py.L_grad_B)
    if hasattr(py, 'X20'):
        compare_field('X20', py.X20)
        compare_field('X2s', py.X2s)
        compare_field('X2c', py.X2c)
        compare_field('Y20', py.Y20)
        compare_field('Y2s', py.Y2s)
        compare_field('Y2c', py.Y2c)
        compare_field('Z20', py.Z20)
        compare_field('Z2s', py.Z2s)
        compare_field('Z2c', py.Z2c)
        compare_field('B20', py.B20)
        compare_field('d2_volume_d_psi2', py.d2_volume_d_psi2)
        compare_field('DWell_times_r2', py.DWell_times_r2)
        compare_field('DGeod_times_r2', py.DGeod_times_r2)
        compare_field('DMerc_times_r2', py.DMerc_times_r2)
        compare_field('grad_grad_B_inverse_scale_length_vs_zeta', py.grad_grad_B_inverse_scale_length_vs_varphi)
        compare_field('grad_grad_B_inverse_scale_length', py.grad_grad_B_inverse_scale_length)
        #compare_field('r_singularity', py.r_singularity) # Could be different if Newton refinement was on in 1 but not the other
        compare_field('r_singularity_basic_vs_zeta', py.r_singularity_basic_vs_varphi)
    if hasattr(py, 'X3c1'):
        compare_field('X3c1', py.X3c1)
        compare_field('X3s1', py.X3s1)
        compare_field('Y3c1', py.Y3c1)
        compare_field('Y3s1', py.Y3s1)
        compare_field('Z3c1', py.Z3c1)
        compare_field('Z3s1', py.Z3s1)
        compare_field('X3c3', py.X3c3)
        compare_field('X3s3', py.X3s3)
        compare_field('Y3c3', py.Y3c3)
        compare_field('Y3s3', py.Y3s3)
        compare_field('Z3c3', py.Z3c3)
        compare_field('Z3s3', py.Z3s3)
        compare_field('B0_order_a_squared_to_cancel', py.B0_order_a_squared_to_cancel)

    # logger.info('Creating RBC, RBS, ZBC and ZBS arrays')
    R_2D, Z_2D, _ = py.Frenet_to_cylindrical(r=r, ntheta=ntheta)
    RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, py.nfp, mpol, ntor, py.lasym)

    RBC = RBC.transpose()
    ZBS = ZBS.transpose()
    if py.lasym:
        RBS = RBS.transpose()
        ZBC = ZBC.transpose()

    # logger.info('Comparing RBC, RBS, ZBC and ZBS arrays')
    compare_field('RBC', RBC)
    compare_field('RBS', RBS)
    compare_field('ZBC', ZBC)
    compare_field('ZBS', ZBS)

    # logger.info('Test boundary and axis splines in cylindrical coordinates')
    R_fortran, Z_fortran, R0_fortran, Z0_fortran, r, mpol, ntor, _ = fortran_plot_single(filename=filename, ntheta=ntheta, nphi=nphi)
    _, _, Z_qsc, R_qsc = py.get_boundary(r=r, ntheta=ntheta, nphi=nphi, mpol=mpol, ntor=ntor, ntheta_fourier=2*mpol)
    phi_array = np.linspace(0, 2*np.pi, nphi)
    R0_qsc = py.R0_func(phi_array)
    Z0_qsc = py.Z0_func(phi_array)
    rtol = 1e-7
    atol = 1e-7
    np.testing.assert_allclose(R_fortran, R_qsc,   rtol=rtol, atol=atol)
    np.testing.assert_allclose(Z_fortran, Z_qsc,   rtol=rtol, atol=atol)
    np.testing.assert_allclose(R0_fortran, R0_qsc, rtol=rtol, atol=atol)
    np.testing.assert_allclose(Z0_fortran, Z0_qsc, rtol=rtol, atol=atol)

    f.close()
    
class QscTests(unittest.TestCase):

    def test_curvature_torsion(self):
        """
        Test that the curvature and torsion match an independent
        calculation using the fortran code.
        """
        
        # Stellarator-symmetric case:
        stel = Qsc(rc=[1.3, 0.3, 0.01, -0.001],
                   zs=[0, 0.4, -0.02, -0.003], nfp=5, nphi=15)
        
        curvature_fortran = [1.74354628565018, 1.61776632275718, 1.5167042487094, 
                             1.9179603622369, 2.95373444883134, 3.01448808361584, 1.7714523990583, 
                             1.02055493647363, 1.02055493647363, 1.77145239905828, 3.01448808361582, 
                             2.95373444883135, 1.91796036223691, 1.5167042487094, 1.61776632275717]
        
        torsion_fortran = [0.257226801231061, -0.131225053326418, -1.12989287766591, 
                           -1.72727988032403, -1.48973327005739, -1.34398161921833, 
                           -1.76040161697108, -2.96573007082039, -2.96573007082041, 
                           -1.7604016169711, -1.34398161921833, -1.48973327005739, 
                           -1.72727988032403, -1.12989287766593, -0.13122505332643]

        varphi_fortran = [0, 0.0909479184372571, 0.181828299105257, 
                          0.268782689120682, 0.347551637441381, 0.42101745128188, 
                          0.498195826255542, 0.583626271820683, 0.673010789615233, 
                          0.758441235180374, 0.835619610154036, 0.909085423994535, 
                          0.987854372315234, 1.07480876233066, 1.16568914299866]

        rtol = 1e-13
        atol = 1e-13
        np.testing.assert_allclose(stel.curvature, curvature_fortran, rtol=rtol, atol=atol)
        np.testing.assert_allclose(stel.torsion, torsion_fortran, rtol=rtol, atol=atol)
        np.testing.assert_allclose(stel.varphi, varphi_fortran, rtol=rtol, atol=atol)

        # Non-stellarator-symmetric case:
        stel = Qsc(rc=[1.3, 0.3, 0.01, -0.001],
                   zs=[0, 0.4, -0.02, -0.003],
                   rs=[0, -0.1, -0.03, 0.002],
                   zc=[0.3, 0.2, 0.04, 0.004], nfp=5, nphi=15)
        
        curvature_fortran = [2.10743037699653, 2.33190181686696, 1.83273654023051, 
                             1.81062232906827, 2.28640008392347, 1.76919841474321, 0.919988560478029, 
                             0.741327470169023, 1.37147330126897, 2.64680884158075, 3.39786486424852, 
                             2.47005615416209, 1.50865425515356, 1.18136509189105, 1.42042418970102]
        
        torsion_fortran = [-0.167822738386845, -0.0785778346620885, -1.02205137493593, 
                           -2.05213528002946, -0.964613202459108, -0.593496282035916, 
                           -2.15852857178204, -3.72911055219339, -1.9330792779459, 
                           -1.53882290974916, -1.42156496444929, -1.11381642382793, 
                           -0.92608309386204, -0.868339812017432, -0.57696266498748]

        varphi_fortran = [0, 0.084185130335249, 0.160931495903817, 
                          0.232881563535092, 0.300551168190665, 0.368933497012765, 
                          0.444686439112853, 0.528001290336008, 0.612254611059372, 
                          0.691096975269652, 0.765820243301147, 0.846373713025902, 
                          0.941973362938683, 1.05053459351092, 1.15941650366667]
        rtol = 1e-13
        atol = 1e-13
        np.testing.assert_allclose(stel.curvature, curvature_fortran, rtol=rtol, atol=atol)
        np.testing.assert_allclose(stel.torsion, torsion_fortran, rtol=rtol, atol=atol)
        np.testing.assert_allclose(stel.varphi, varphi_fortran, rtol=rtol, atol=atol)
            
    def test_compare_to_fortran(self):
        """
        Compare the output of this python code to the fortran code.
        """
        compare_to_fortran("r2 section 5.1", "quasisymmetry_out.LandremanSengupta2019_section5.1.nc")
        compare_to_fortran("r2 section 5.2", "quasisymmetry_out.LandremanSengupta2019_section5.2.nc")
        compare_to_fortran("r2 section 5.3", "quasisymmetry_out.LandremanSengupta2019_section5.3.nc")
        compare_to_fortran("r2 section 5.4", "quasisymmetry_out.LandremanSengupta2019_section5.4.nc")
        compare_to_fortran("r2 section 5.5", "quasisymmetry_out.LandremanSengupta2019_section5.5.nc")

    def test_change_nfourier(self):
        """
        Test the change_nfourier() method.
        """
        rtol = 1e-13
        atol = 1e-13
        s1 = Qsc.from_paper('r2 section 5.2')
        m = s1.nfourier
        for n in range(2, 7):
            s2 = Qsc.from_paper('r2 section 5.2')
            s2.change_nfourier(n)
            if n <= m:
                # We lowered nfourier
                np.testing.assert_allclose(s1.rc[:n], s2.rc, rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.rs[:n], s2.rs, rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.zc[:n], s2.zc, rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.zs[:n], s2.zs, rtol=rtol, atol=atol)
            else:
                # We increased nfourier
                np.testing.assert_allclose(s1.rc, s2.rc[:m], rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.rs, s2.rs[:m], rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.zc, s2.zc[:m], rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.zs, s2.zs[:m], rtol=rtol, atol=atol)
                z = np.zeros(n - s1.nfourier)
                np.testing.assert_allclose(z, s2.rc[m:], rtol=rtol, atol=atol)
                np.testing.assert_allclose(z, s2.rs[m:], rtol=rtol, atol=atol)
                np.testing.assert_allclose(z, s2.zc[m:], rtol=rtol, atol=atol)
                np.testing.assert_allclose(z, s2.zs[m:], rtol=rtol, atol=atol)
                
if __name__ == "__main__":
    unittest.main()
