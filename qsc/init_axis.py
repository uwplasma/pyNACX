"""
This module contains the routine to initialize quantities like
curvature and torsion from the magnetix axis shape.
"""

import logging
import numpy as np

from qsc.util import jax_fourier_minimum, convert_to_spline
from qsc.types import Init_Axis_Results
from .spectral_diff_matrix import jax_spectral_diff_matrix, spectral_diff_matrix
#from .util import jax_fourier_minimum

import jax
import jax.numpy as jnp


# Set default floating-point precision to 64-bit (double precision)
jax.config.update('jax_enable_x64', True)

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_helicity(nphi, normal_cylindrical, spsi, sG):
    """
    Determine the integer N associated with the type of quasisymmetry
    by counting the number of times the normal vector rotates
    poloidally as you follow the axis around toroidally.
    """

    def classify_quadrant(j):
        x, _, z = normal_cylindrical[j]

        def case_1(): return 1  # x >= 0, z >= 0
        def case_2(): return 4  # x >= 0, z < 0
        def case_3(): return 2  # x < 0, z >= 0
        def case_4(): return 3  # x < 0, z < 0

        return jax.lax.cond(x >= 0,
                        lambda: jax.lax.cond(z >= 0, case_1, case_2),
                        lambda: jax.lax.cond(z >= 0, case_3, case_4))

    quadrant = jnp.array([classify_quadrant(j) for j in range(nphi)])
    quadrant = jnp.append(quadrant, quadrant[0])

    def count_step(j, counter):
        counter = jax.lax.cond(
            (quadrant[j] == 4) & (quadrant[j+1] == 1),
            lambda: counter + 1,
            lambda: jax.lax.cond(
                (quadrant[j] == 1) & (quadrant[j+1] == 4),
                lambda: counter - 1,
                lambda: counter + (quadrant[j+1] - quadrant[j])
            )
        )
        return counter

    counter = jax.lax.fori_loop(0, nphi, count_step, 0)
    counter *= spsi * sG
    helicity = counter / 4
    return helicity

def init_axis(nphi, nfp, rc, rs, zc, zs, nfourier, sG, B0, etabar, spsi, sigma0, order, B2s)-> Init_Axis_Results:
    """
    Initialize the curvature, torsion, differentiation matrix, etc. waiting on interpax support for cubic spline
    """
    #from .util import jax_fourier_minimum
    
    # Generate phi
    
    phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
    d_phi = phi[1] - phi[0]
    

    # Compute n and the angles
    n = jnp.arange(0, nfourier) * nfp
    angles = jnp.outer(n, phi)
    sinangles = jnp.sin(angles)
    cosangles = jnp.cos(angles)
    

    # Compute R0, Z0, R0p, Z0p, R0pp, Z0pp, R0ppp, Z0ppp
    R0 = jnp.dot(rc, cosangles) + jnp.dot(rs, sinangles)
    
    Z0 = jnp.dot(zc, cosangles) + jnp.dot(zs, sinangles)
    R0p = jnp.dot(rc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(rs, n[:, jnp.newaxis] * cosangles)
    Z0p = jnp.dot(zc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(zs, n[:, jnp.newaxis] * cosangles)
    R0pp = jnp.dot(rc, -n[:, jnp.newaxis]**2 * cosangles) + jnp.dot(rs, -n[:, jnp.newaxis]**2 * sinangles)
    Z0pp = jnp.dot(zc, -n[:, jnp.newaxis]**2 * cosangles) + jnp.dot(zs, -n[:, jnp.newaxis]**2 * sinangles)
    R0ppp = jnp.dot(rc, n[:, jnp.newaxis]**3 * sinangles) + jnp.dot(rs, -n[:, jnp.newaxis]**3 * cosangles)
    Z0ppp = jnp.dot(zc, n[:, jnp.newaxis]**3 * sinangles) + jnp.dot(zs, -n[:, jnp.newaxis]**3 * cosangles)

    d_l_d_phi = jnp.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
    d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
    B0_over_abs_G0 = nphi / jnp.sum(d_l_d_phi)
    abs_G0_over_B0 = 1 / B0_over_abs_G0

    G0 = sG * abs_G0_over_B0 * B0

    # For these next arrays, the first dimension is phi, and the 2nd dimension is (R, phi, Z).
    d_r_d_phi_cylindrical = jnp.array([R0p, R0, Z0p])  #.transpose()
    d_r_d_phi_cylindrical = jnp.transpose(d_r_d_phi_cylindrical)
    
    d2_r_d_phi2_cylindrical = jnp.array([R0pp - R0, 2 * R0p, Z0pp])  #.transpose()
    d2_r_d_phi2_cylindrical = jnp.transpose(d2_r_d_phi2_cylindrical)
    
    d3_r_d_phi3_cylindrical = jnp.array([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]) #.transpose()
    d3_r_d_phi3_cylindrical = jnp.transpose(d3_r_d_phi3_cylindrical)

    print(f"d_l_d_phi -- Any NaNs in array? {jnp.isnan(d_l_d_phi).any()}")
    print(f"Any Infs in array? {jnp.isinf(d_l_d_phi).any()}")
    print(f"Max value: {jnp.max(d_l_d_phi)}, Min value: {jnp.min(d_l_d_phi)}")

    # Calculate tangent_cylindrical and d_tangent_d_l_cylindrical
    tangent_cylindrical = d_r_d_phi_cylindrical / d_l_d_phi[:, jnp.newaxis]
    d_tangent_d_l_cylindrical = ((-d_r_d_phi_cylindrical * d2_l_d_phi2[:, jnp.newaxis] / d_l_d_phi[:, jnp.newaxis]) \
                                + d2_r_d_phi2_cylindrical) / (d_l_d_phi[:, jnp.newaxis] * d_l_d_phi[:, jnp.newaxis])

    print(f"d_tangent_d_l_cylindrical -- Any NaNs in array? {jnp.isnan(d_tangent_d_l_cylindrical).any()}")
    print(f"Any Infs in array? {jnp.isinf(d_tangent_d_l_cylindrical).any()}")
    print(f"Max value: {jnp.max(d_tangent_d_l_cylindrical)}, Min value: {jnp.min(d_tangent_d_l_cylindrical)}")

    curvature = jnp.sqrt(d_tangent_d_l_cylindrical[:,0] * d_tangent_d_l_cylindrical[:,0] + \
                        d_tangent_d_l_cylindrical[:,1] * d_tangent_d_l_cylindrical[:,1] + \
                        d_tangent_d_l_cylindrical[:,2] * d_tangent_d_l_cylindrical[:,2])

    axis_length = jnp.sum(d_l_d_phi) * d_phi * nfp
    #rms_curvature = jnp.sqrt((jnp.sum(curvature * curvature * d_l_d_phi) * d_phi * nfp) / axis_length)
    #mean_of_R = jnp.sum(R0 * d_l_d_phi) * d_phi * nfp / axis_length
    #mean_of_Z = jnp.sum(Z0 * d_l_d_phi) * d_phi * nfp / axis_length
    #standard_deviation_of_R = jnp.sqrt(jnp.sum((R0 - mean_of_R) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)
    #tandard_deviation_of_Z = jnp.sqrt(jnp.sum((Z0 - mean_of_Z) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)

    
    
    


    # Calculate normal_cylindrical
    normal_cylindrical = d_tangent_d_l_cylindrical / curvature[:, jnp.newaxis]
    
    
    

    print('before helicity')
    helicity = calculate_helicity(nphi, normal_cylindrical, spsi, sG)
    print('after helicity')
    # b = t x n
    binormal_cylindrical = jnp.zeros((nphi, 3))
    binormal_cylindrical = binormal_cylindrical.at[:,0].set(tangent_cylindrical[:,1] * normal_cylindrical[:,2] - tangent_cylindrical[:,2] * normal_cylindrical[:,1])
    binormal_cylindrical = binormal_cylindrical.at[:,1].set(tangent_cylindrical[:,2] * normal_cylindrical[:,0] - tangent_cylindrical[:,0] * normal_cylindrical[:,2])
    binormal_cylindrical = binormal_cylindrical.at[:,2].set(tangent_cylindrical[:,0] * normal_cylindrical[:,1] - tangent_cylindrical[:,1] * normal_cylindrical[:,0])
    
    # TODO
    # We use the same sign convention for torsion as the
    # Landreman-Sengupta-Plunk paper, wikipedia, and
    # mathworld.wolfram.com/Torsion.html.  This sign convention is
    # opposite to Garren & Boozer's sign convention!
    torsion_numerator = (
        d_r_d_phi_cylindrical[:,0] * (d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,2] - d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,1])
        + d_r_d_phi_cylindrical[:,1] * (d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,0] - d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,2])
        + d_r_d_phi_cylindrical[:,2] * (d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,1] - d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,0])
    )

    torsion_denominator = (
        (d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,2] - d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,1]) ** 2
        + (d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,0] - d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,2]) ** 2
        + (d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,1] - d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,0]) ** 2
    )

    torsion = torsion_numerator / torsion_denominator
    etabar_squared_over_curvature_squared = etabar ** 2 / curvature ** 2
    #print('spectral diff matrix')
   
    d_d_phi = jax_spectral_diff_matrix(nphi, xmax=2 * jnp.pi / nfp)
    #print('after spectral diff matrix')
    d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi

    # Calculate d_d_varphi
    d_d_varphi = d_d_phi / d_varphi_d_phi[:, jnp.newaxis]
    
    # Compute the Boozer toroidal angle:
    varphi = jnp.zeros(nphi)
    for j in range(1, nphi):
        # To get toroidal angle on the full mesh, we need d_l_d_phi on the half mesh.
        varphi = varphi.at[j].set(varphi[j-1] + (d_l_d_phi[j-1] + d_l_d_phi[j]))
    varphi = varphi * (0.5 * d_phi * 2 * np.pi / axis_length)


    # Add all results to self:
    X1s = jnp.zeros(nphi)
    X1c = etabar / curvature
    print('before fourier min')
    min_R0 = jax_fourier_minimum(R0)
    print('after fourier min')
    Bbar = spsi * B0

    # The output is not stellarator-symmetric if (1) R0s is nonzero,
    # (2) Z0c is nonzero, (3) sigma_initial is nonzero, or (B2s is
    # nonzero and order != 'r1')
    lasym = jnp.max(jnp.abs(rs)) > 0 or jnp.max(jnp.abs(zc)) > 0 \
        or sigma0 != 0 or (order != 'r1' and B2s != 0)
    
    print('before spline')

    # Functions that converts a toroidal angle phi0 on the axis to the axis radial and vertical coordinates
    R0_func = convert_to_spline(sum([rc[i]*jnp.cos(i*nfp*phi) +\
                                               rs[i]*jnp.sin(i*nfp*phi) \
                                              for i in range(len(rc))]), phi, nfp)
    Z0_func = convert_to_spline(sum([zc[i]*jnp.cos(i*nfp*phi) +\
                                               zs[i]*jnp.sin(i*nfp*phi) \
                                              for i in range(len(zs))]), phi, nfp)
    #self.lasym = lasym
    #self.R0_func = R0_func
    #self.Z0_func = Z0_func

    # Spline interpolants for the cylindrical components of the Frenet-Serret frame:
    #got rid of self statments

    normal_R_spline     = convert_to_spline(normal_cylindrical[:,0], phi, nfp)
    normal_phi_spline   = convert_to_spline(normal_cylindrical[:,1], phi, nfp)
    normal_z_spline     = convert_to_spline(normal_cylindrical[:,2], phi, nfp)
    binormal_R_spline   = convert_to_spline(binormal_cylindrical[:,0], phi, nfp)
    binormal_phi_spline = convert_to_spline(binormal_cylindrical[:,1], phi, nfp)
    binormal_z_spline   = convert_to_spline(binormal_cylindrical[:,2], phi, nfp)
    tangent_R_spline    = convert_to_spline(tangent_cylindrical[:,0], phi, nfp)
    tangent_phi_spline  = convert_to_spline(tangent_cylindrical[:,1], phi, nfp)
    tangent_z_spline    = convert_to_spline(tangent_cylindrical[:,2], phi, nfp)
    
    normal_R_spline     = convert_to_spline(normal_cylindrical[:,0], phi, nfp)
    normal_phi_spline   = convert_to_spline(normal_cylindrical[:,1], phi, nfp)
    normal_z_spline     = convert_to_spline(normal_cylindrical[:,2], phi, nfp)
    binormal_R_spline   = convert_to_spline(binormal_cylindrical[:,0], phi, nfp)
    binormal_phi_spline = convert_to_spline(binormal_cylindrical[:,1], phi, nfp)
    binormal_z_spline   = convert_to_spline(binormal_cylindrical[:,2], phi, nfp)
    tangent_R_spline    = convert_to_spline(tangent_cylindrical[:,0], phi, nfp)
    tangent_phi_spline  = convert_to_spline(tangent_cylindrical[:,1], phi, nfp)
    tangent_z_spline    = convert_to_spline(tangent_cylindrical[:,2], phi, nfp)
    print('after spline')
    # Spline interpolant for nu = varphi - phi, used for plotting
    #self.nu_spline = self.convert_to_spline(varphi - phi, phi, nfp)
    nu_spline = convert_to_spline(varphi - phi, phi, nfp)

    return Init_Axis_Results( 
        helicity, 
        normal_cylindrical, 
        etabar_squared_over_curvature_squared, 
        varphi, 
        d_d_phi, 
        d_varphi_d_phi, 
        d_d_varphi, 
        phi, 
        abs_G0_over_B0, 
        d_phi, 
        R0, 
        Z0, 
        R0p, 
        Z0p, 
        R0pp, 
        Z0pp, 
        R0ppp, 
        Z0ppp, 
        G0, 
        d_l_d_phi, 
        axis_length, 
        curvature, 
        torsion, 
        X1s, 
        X1c, 
        min_R0, 
        tangent_cylindrical, 
        binormal_cylindrical, 
        Bbar, 
        lasym, 
        R0_func, 
        Z0_func, 
        normal_R_spline, 
        normal_phi_spline, 
        normal_z_spline, 
        binormal_R_spline, 
        binormal_phi_spline, 
        binormal_z_spline, 
        tangent_R_spline, 
        tangent_phi_spline, 
        tangent_z_spline, 
        nu_spline  
    )

    
    
    
    
