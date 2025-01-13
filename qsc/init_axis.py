"""
This module contains the routine to initialize quantities like
curvature and torsion from the magnetix axis shape.
"""

import logging
import numpy as np
from scipy.interpolate import CubicSpline as spline
from .spectral_diff_matrix import spectral_diff_matrix
from .util import fourier_minimum

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
    quadrant = jnp.zeros(nphi + 1)
    for j in range(nphi):
        if normal_cylindrical[j,0] >= 0:
            if normal_cylindrical[j,2] >= 0:
                quadrant[j] = 1
            else:
                quadrant[j] = 4
        else:
            if normal_cylindrical[j,2] >= 0:
                quadrant[j] = 2
            else:
                quadrant[j] = 3
    quadrant[nphi] = quadrant[0]

    counter = 0
    for j in range(nphi):
        if quadrant[j] == 4 and quadrant[j+1] == 1:
            counter += 1
        elif quadrant[j] == 1 and quadrant[j+1] == 4:
            counter -= 1
        else:
            counter += quadrant[j+1] - quadrant[j]

    # It is necessary to flip the sign of axis_helicity in order
    # to maintain "iota_N = iota + axis_helicity" under the parity
    # transformations.
    counter *= spsi * sG
    helicity = counter / 4
    return helicity

# Define periodic spline interpolant conversion used in several scripts and plotting
def convert_to_spline(self,array, phi, nfp):
    sp=spline(jnp.append(phi,2*np.pi/nfp), jnp.append(array,array[0]), bc_type='periodic')
    return sp

def init_axis(self, nphi, nfp, rc, rs, zc, zs, nfourier, sG, B0, etabar, spsi, sigma0, order, B2s):
    """
    Initialize the curvature, torsion, differentiation matrix, etc.
    """

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
    d_r_d_phi_cylindrical = jnp.array([R0p, R0, Z0p]).transpose()
    d2_r_d_phi2_cylindrical = jnp.array([R0pp - R0, 2 * R0p, Z0pp]).transpose()
    d3_r_d_phi3_cylindrical = jnp.array([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).transpose()

    # Calculate tangent_cylindrical and d_tangent_d_l_cylindrical
    tangent_cylindrical = d_r_d_phi_cylindrical / d_phi[:, jnp.newaxis]
    d_tangent_d_l_cylindrical = ((-d_r_d_phi_cylindrical * d2_l_d_phi2[:, jnp.newaxis] / d_l_d_phi[:, jnp.newaxis]) \
                                + d2_r_d_phi2_cylindrical) / (d_l_d_phi[:, jnp.newaxis] * d_l_d_phi[:, jnp.newaxis])

    curvature = jnp.sqrt(d_tangent_d_l_cylindrical[:,0] * d_tangent_d_l_cylindrical[:,0] + \
                        d_tangent_d_l_cylindrical[:,1] * d_tangent_d_l_cylindrical[:,1] + \
                        d_tangent_d_l_cylindrical[:,2] * d_tangent_d_l_cylindrical[:,2])

    axis_length = jnp.sum(d_l_d_phi) * d_phi * nfp
    rms_curvature = jnp.sqrt((jnp.sum(curvature * curvature * d_l_d_phi) * d_phi * nfp) / axis_length)
    mean_of_R = jnp.sum(R0 * d_l_d_phi) * d_phi * nfp / axis_length
    mean_of_Z = jnp.sum(Z0 * d_l_d_phi) * d_phi * nfp / axis_length
    standard_deviation_of_R = jnp.sqrt(jnp.sum((R0 - mean_of_R) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)
    standard_deviation_of_Z = jnp.sqrt(jnp.sum((Z0 - mean_of_Z) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)

    # Calculate normal_cylindrical
    normal_cylindrical = d_tangent_d_l_cylindrical / curvature[:, jnp.newaxis]

    helicity = calculate_helicity(nphi, normal_cylindrical, spsi, sG)

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

    d_d_phi = spectral_diff_matrix(nphi, xmax=2 * np.pi / nfp)
    d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi

    # Calculate d_d_varphi
    d_d_varphi = d_d_phi / d_varphi_d_phi[:, jnp.newaxis]
    
    # Compute the Boozer toroidal angle:
    varphi = jnp.zeros(nphi)
    for j in range(1, nphi):
        # To get toroidal angle on the full mesh, we need d_l_d_phi on the half mesh.
        varphi[j] = varphi[j-1] + (d_l_d_phi[j-1] + d_l_d_phi[j])
    varphi = varphi * (0.5 * d_phi * 2 * np.pi / axis_length)


    # Add all results to self:
    X1s = jnp.zeros(nphi)
    X1c = etabar / curvature
    min_R0 = fourier_minimum(R0)
    Bbar = spsi * B0

    # The output is not stellarator-symmetric if (1) R0s is nonzero,
    # (2) Z0c is nonzero, (3) sigma_initial is nonzero, or (B2s is
    # nonzero and order != 'r1')
    lasym = jnp.max(jnp.abs(rs)) > 0 or jnp.max(jnp.abs(zc)) > 0 \
        or sigma0 != 0 or (order != 'r1' and B2s != 0)

    # Functions that converts a toroidal angle phi0 on the axis to the axis radial and vertical coordinates
    R0_func = self.convert_to_spline(sum([rc[i]*jnp.cos(i*nfp*phi) +\
                                               rs[i]*jnp.sin(i*nfp*phi) \
                                              for i in range(len(rc))]), phi, nfp)
    Z0_func = self.convert_to_spline(sum([zc[i]*jnp.cos(i*nfp*phi) +\
                                               zs[i]*jnp.sin(i*nfp*phi) \
                                              for i in range(len(zs))]), phi, nfp)
    self.lasym = lasym
    self.R0_func = R0_func
    self.Z0_func = Z0_func

    # Spline interpolants for the cylindrical components of the Frenet-Serret frame:
    self.normal_R_spline     = self.convert_to_spline(normal_cylindrical[:,0], phi, nfp)
    self.normal_phi_spline   = self.convert_to_spline(normal_cylindrical[:,1], phi, nfp)
    self.normal_z_spline     = self.convert_to_spline(normal_cylindrical[:,2], phi, nfp)
    self.binormal_R_spline   = self.convert_to_spline(binormal_cylindrical[:,0], phi, nfp)
    self.binormal_phi_spline = self.convert_to_spline(binormal_cylindrical[:,1], phi, nfp)
    self.binormal_z_spline   = self.convert_to_spline(binormal_cylindrical[:,2], phi, nfp)
    self.tangent_R_spline    = self.convert_to_spline(tangent_cylindrical[:,0], phi, nfp)
    self.tangent_phi_spline  = self.convert_to_spline(tangent_cylindrical[:,1], phi, nfp)
    self.tangent_z_spline    = self.convert_to_spline(tangent_cylindrical[:,2], phi, nfp)

    # Spline interpolant for nu = varphi - phi, used for plotting
    self.nu_spline = self.convert_to_spline(varphi - phi, phi, nfp)

    return helicity,\
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
    abs_G0_over_B0
