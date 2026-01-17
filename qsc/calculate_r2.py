"""
This module contains the calculation for the O(r^2) solution
"""

import logging
import numpy as np
from jax import config
from qsc.grad_B_tensor import calculate_grad_grad_B_tensor
from qsc.r_singularity import calculate_r_singularity

from .mercier import mercier
from .util import mu0
import jax.numpy as jnp
from .calculate_r1_helpers import *
from .calculate_r2_helpers import * 
from qsc.types import R2_results, Complete_R2_Results

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@jax.jit(static_argnums=(18,21))
def calc_r2(X1c, Y1c, Y1s, B0_over_abs_G0, d_d_varphi, iota_N, torsion, abs_G0_over_B0, B2s, B0, curvature, etabar, B2c, spsi, sG, p2, sigma, I2_over_B0, nphi, d_l_d_phi, helicity, nfp, G0, iota, I2, varphi, d_X1c_d_varphi, d_Y1c_d_varphi, d_Y1s_d_varphi, d_phi, axis_length) -> Complete_R2_Results:
    V1 = X1c * X1c + Y1c * Y1c + Y1s * Y1s
    V2 = 2 * Y1s * Y1c
    V3 = X1c * X1c + Y1c * Y1c - Y1s * Y1s

    factor = - B0_over_abs_G0 / 8
    Z20 = factor*jnp.matmul(d_d_varphi,V1)
    Z2s = factor*(jnp.matmul(d_d_varphi,V2) - 2 * iota_N * V3)
    Z2c = factor*(jnp.matmul(d_d_varphi,V3) + 2 * iota_N * V2)

    qs = -iota_N * X1c - Y1s * torsion * abs_G0_over_B0
    qc = jnp.matmul(d_d_varphi,X1c) - Y1c * torsion * abs_G0_over_B0
    rs = jnp.matmul(d_d_varphi,Y1s) - iota_N * Y1c
    rc = jnp.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * torsion * abs_G0_over_B0

    X2s = B0_over_abs_G0 * (jnp.matmul(d_d_varphi,Z2s) - 2*iota_N*Z2c + B0_over_abs_G0 * ( abs_G0_over_B0*abs_G0_over_B0*B2s/B0 + (qc * qs + rc * rs)/2)) / curvature

    X2c = B0_over_abs_G0 * (jnp.matmul(d_d_varphi,Z2c) + 2*iota_N*Z2s - B0_over_abs_G0 * (-abs_G0_over_B0*abs_G0_over_B0*B2c/B0 \
           + abs_G0_over_B0*abs_G0_over_B0*etabar*etabar/2 - (qc * qc - qs * qs + rc * rc - rs * rs)/4)) / curvature

    beta_1s = -4 * spsi * sG * mu0 * p2 * etabar * abs_G0_over_B0 / (iota_N * B0 * B0)

    Y2s_from_X20 = -sG * spsi * curvature * curvature / (etabar * etabar)
    Y2s_inhomogeneous = sG * spsi * (-curvature/2 + curvature*curvature/(etabar*etabar)*(-X2c + X2s * sigma))

    Y2c_from_X20 = -sG * spsi * curvature * curvature * sigma / (etabar * etabar)
    Y2c_inhomogeneous = sG * spsi * curvature * curvature / (etabar * etabar) * (X2s + X2c * sigma)
    
    # Note: in the fX* and fY* quantities below, I've omitted the
    # contributions from X20 and Y20 to the d/dzeta terms. These
    # contributions are handled later when we assemble the large
    # matrix.

    fX0_from_X20 = -4 * sG * spsi * abs_G0_over_B0 * (Y2c_from_X20 * Z2s - Y2s_from_X20 * Z2c)
    fX0_from_Y20 = -torsion * abs_G0_over_B0 - 4 * sG * spsi * abs_G0_over_B0 * (Z2s) \
        - spsi * I2_over_B0 * (-2) * abs_G0_over_B0
    fX0_inhomogeneous = curvature * abs_G0_over_B0 * Z20 - 4 * sG * spsi * abs_G0_over_B0 * (Y2c_inhomogeneous * Z2s - Y2s_inhomogeneous * Z2c) \
        - spsi * I2_over_B0 * (0.5 * curvature * sG * spsi) * abs_G0_over_B0 + beta_1s * abs_G0_over_B0 / 2 * Y1c

    fXs_from_X20 = -torsion * abs_G0_over_B0 * Y2s_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (Y2c_from_X20 * Z20) \
        - spsi * I2_over_B0 * (- 2 * Y2s_from_X20) * abs_G0_over_B0
    fXs_from_Y20 = - 4 * spsi * sG * abs_G0_over_B0 * (-Z2c + Z20)
    fXs_inhomogeneous = jnp.matmul(d_d_varphi,X2s) - 2 * iota_N * X2c - torsion * abs_G0_over_B0 * Y2s_inhomogeneous + curvature * abs_G0_over_B0 * Z2s \
        - 4 * spsi * sG * abs_G0_over_B0 * (Y2c_inhomogeneous * Z20) \
        - spsi * I2_over_B0 * (0.5 * curvature * spsi * sG - 2 * Y2s_inhomogeneous) * abs_G0_over_B0 \
        - (0.5) * abs_G0_over_B0 * beta_1s * Y1s

    fXc_from_X20 = - torsion * abs_G0_over_B0 * Y2c_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (-Y2s_from_X20 * Z20) \
        - spsi * I2_over_B0 * (- 2 * Y2c_from_X20) * abs_G0_over_B0
    fXc_from_Y20 = - torsion * abs_G0_over_B0 - 4 * spsi * sG * abs_G0_over_B0 * (Z2s) \
        - spsi * I2_over_B0 * (-2) * abs_G0_over_B0
    fXc_inhomogeneous = jnp.matmul(d_d_varphi,X2c) + 2 * iota_N * X2s - torsion * abs_G0_over_B0 * Y2c_inhomogeneous + curvature * abs_G0_over_B0 * Z2c \
        - 4 * spsi * sG * abs_G0_over_B0 * (-Y2s_inhomogeneous * Z20) \
        - spsi * I2_over_B0 * (0.5 * curvature * sG * spsi - 2 * Y2c_inhomogeneous) * abs_G0_over_B0 \
        - (0.5) * abs_G0_over_B0 * beta_1s * Y1c

    fY0_from_X20 = torsion * abs_G0_over_B0 - spsi * I2_over_B0 * (2) * abs_G0_over_B0

    fY0_from_Y20 = jnp.zeros(nphi)
    fY0_inhomogeneous = -4 * spsi * sG * abs_G0_over_B0 * (X2s * Z2c - X2c * Z2s) \
        - spsi * I2_over_B0 * (-0.5 * curvature * X1c * X1c) * abs_G0_over_B0 - (0.5) * abs_G0_over_B0 * beta_1s * X1c

    fYs_from_X20 = -2 * iota_N * Y2c_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (Z2c)
    fYs_from_Y20 = jnp.full(nphi, -2 * iota_N)
    fYs_inhomogeneous = jnp.matmul(d_d_varphi,Y2s_inhomogeneous) - 2 * iota_N * Y2c_inhomogeneous + torsion * abs_G0_over_B0 * X2s \
        - 4 * spsi * sG * abs_G0_over_B0 * (-X2c * Z20) - 2 * spsi * I2_over_B0 * X2s * abs_G0_over_B0

    fYc_from_X20 = 2 * iota_N * Y2s_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (-Z2s)
    fYc_from_Y20 = jnp.zeros(nphi)
    fYc_inhomogeneous = jnp.matmul(d_d_varphi,Y2c_inhomogeneous) + 2 * iota_N * Y2s_inhomogeneous + torsion * abs_G0_over_B0 * X2c \
        - 4 * spsi * sG * abs_G0_over_B0 * (X2s * Z20) \
        - spsi * I2_over_B0 * (-0.5 * curvature * X1c * X1c + 2 * X2c) * abs_G0_over_B0 + 0.5 * abs_G0_over_B0 * beta_1s * X1c

    matrix = jnp.zeros((2 * nphi, 2 * nphi))
    right_hand_side = jnp.zeros(2 * nphi)


    def matrix_body(j, carry): 
        """
        calculations body needed for jax compatible for loops
        """
        matrix, Y1c, Y1s, X1c, d_d_varphi, Y2s_from_X20, Y2c_from_X20, fYs_from_X20, fY0_from_X20, fYc_from_X20, fXs_from_X20, fYs_from_Y20, fY0_from_Y20, fYc_from_Y20, fXs_from_Y20 = carry
    

        # Handle the terms involving d X_0 / d zeta and d Y_0 / d zeta:
        # ----------------------------------------------------------------

        # Equation 1, terms involving X0:
        # Contributions arise from Y1c * fYs - Y1s * fYc.
        matrix = matrix.at[j, 0:nphi].set(Y1c.at[j].get() * d_d_varphi.at[j, :].get() * Y2s_from_X20 - Y1s.at[j].get() * d_d_varphi.at[j, :].get() * Y2c_from_X20)
        # Equation 1, terms involving Y0:
        # Contributions arise from -Y1s * fY0 - Y1s * fYc, and they happen to be equal.
        matrix = matrix.at[j, nphi:(2*nphi)].set(-2 * Y1s.at[j].get() * d_d_varphi.at[j, :].get())
    
        # Equation 2, terms involving X0:
        # Contributions arise from -X1c * fX0 + Y1s * fYs + Y1c * fYc
        matrix = matrix.at[j+nphi, 0:nphi].set( -X1c.at[j].get() * d_d_varphi.at[j, :].get() + Y1s.at[j].get() * d_d_varphi.at[j, :].get() * Y2s_from_X20 + Y1c.at[j].get() * d_d_varphi.at[j, :].get() * Y2c_from_X20)
        # Equation 2, terms involving Y0:
        # Contributions arise from -Y1c * fY0 + Y1c * fYc, but they happen to cancel.

        # Now handle the terms involving X_0 and Y_0 without d/dzeta derivatives:
        # ----------------------------------------------------------------

        matrix = matrix.at[j, j       ].set(matrix.at[j, j       ].get() + X1c.at[j].get() * fXs_from_X20.at[j].get() - Y1s.at[j].get() * fY0_from_X20.at[j].get() + Y1c.at[j].get() * fYs_from_X20.at[j].get() - Y1s.at[j].get() * fYc_from_X20.at[j].get())
    
        matrix = matrix.at[j, j + nphi].set( matrix.at[j, j + nphi].get() + X1c.at[j].get() * fXs_from_Y20.at[j].get() - Y1s.at[j].get() * fY0_from_Y20.at[j].get() + Y1c.at[j].get() * fYs_from_Y20.at[j].get() - Y1s.at[j].get() * fYc_from_Y20.at[j].get())
        matrix = matrix.at[j + nphi, j       ].set(matrix.at[j + nphi, j       ].get() - X1c.at[j].get() * fX0_from_X20.at[j].get() + X1c.at[j].get() * fXc_from_X20.at[j].get() - Y1c.at[j].get() * fY0_from_X20.at[j].get() + Y1s.at[j].get() * fYs_from_X20.at[j].get() + Y1c.at[j].get() * fYc_from_X20.at[j].get())
        matrix = matrix.at[j + nphi, j + nphi].set(matrix.at[j + nphi, j + nphi].get() - X1c.at[j].get() * fX0_from_Y20.at[j].get() + X1c.at[j].get() * fXc_from_Y20.at[j].get() - Y1c.at[j].get() * fY0_from_Y20.at[j].get() + Y1s.at[j].get() * fYs_from_Y20.at[j].get() + Y1c.at[j].get() * fYc_from_Y20.at[j].get())
        return matrix, Y1c, Y1s, X1c, d_d_varphi, Y2s_from_X20, Y2c_from_X20, fYs_from_X20, fY0_from_X20, fYc_from_X20, fXs_from_X20, fYs_from_Y20, fY0_from_Y20, fYc_from_Y20, fXs_from_Y20

    carry = (matrix, Y1c, Y1s, X1c, d_d_varphi, Y2s_from_X20, Y2c_from_X20, fYs_from_X20, fY0_from_X20, fYc_from_X20, fXs_from_X20, fYs_from_Y20, fY0_from_Y20, fYc_from_Y20, fXs_from_Y20)
    matrix = jax.lax.fori_loop(0, nphi, matrix_body, carry)[0]

    right_hand_side = right_hand_side.at[0:nphi].set(-(X1c * fXs_inhomogeneous - Y1s * fY0_inhomogeneous + Y1c * fYs_inhomogeneous - Y1s * fYc_inhomogeneous))
    right_hand_side = right_hand_side.at[nphi:2 * nphi].set(-(- X1c * fX0_inhomogeneous + X1c * fXc_inhomogeneous - Y1c * fY0_inhomogeneous + Y1s * fYs_inhomogeneous + Y1c * fYc_inhomogeneous))

    solution = jnp.linalg.solve(matrix, right_hand_side)
    
    X20 = solution[0:nphi]
    Y20 = solution[nphi:2 * nphi] # solutions are the same but somehow Y20 is different 

    # Now that we have X20 and Y20 explicitly, we can reconstruct Y2s, Y2c, and B20:
    Y2s = Y2s_inhomogeneous + Y2s_from_X20 * X20
    Y2c = Y2c_inhomogeneous + Y2c_from_X20 * X20 + Y20

    B20 = B0 * (curvature * X20 - B0_over_abs_G0 * jnp.matmul(d_d_varphi,Z20) + (0.5) * etabar * etabar - mu0 * p2 / (B0 * B0) \
                - 0.25 * B0_over_abs_G0 * B0_over_abs_G0 * (qc * qc + qs * qs + rc * rc + rs * rs))

    normalizer = 1 / jnp.sum(d_l_d_phi)
    
    B20_mean = jnp.sum(B20 * d_l_d_phi) * normalizer
    B20_anomaly = B20 - B20_mean
    B20_residual = jnp.sqrt(jnp.sum((B20 - B20_mean) * (B20 - B20_mean) * d_l_d_phi) * normalizer) / B0
    B20_variation = jnp.max(B20) - jnp.min(B20)

    N_helicity = - helicity * nfp
    G2 = -mu0 * p2 * G0 / (B0 * B0) - iota * I2

    d_curvature_d_varphi = jnp.matmul(d_d_varphi, curvature)
    d_torsion_d_varphi = jnp.matmul(d_d_varphi, torsion)
    d_X20_d_varphi = jnp.matmul(d_d_varphi, X20)
    d_X2s_d_varphi = jnp.matmul(d_d_varphi, X2s)
    d_X2c_d_varphi = jnp.matmul(d_d_varphi, X2c)
    d_Y20_d_varphi = jnp.matmul(d_d_varphi, Y20)
    d_Y2s_d_varphi = jnp.matmul(d_d_varphi, Y2s)
    d_Y2c_d_varphi = jnp.matmul(d_d_varphi, Y2c)
    d_Z20_d_varphi = jnp.matmul(d_d_varphi, Z20)
    d_Z2s_d_varphi = jnp.matmul(d_d_varphi, Z2s)
    d_Z2c_d_varphi = jnp.matmul(d_d_varphi, Z2c)
    d2_X1c_d_varphi2 = jnp.matmul(d_d_varphi, d_X1c_d_varphi)
    d2_Y1c_d_varphi2 = jnp.matmul(d_d_varphi, d_Y1c_d_varphi)
    d2_Y1s_d_varphi2 = jnp.matmul(d_d_varphi, d_Y1s_d_varphi)

    # O(r^2) diagnostics:
    mercier_results = mercier(d_l_d_phi, B0, G0, p2, etabar, curvature, sigma, iota_N, iota, d_phi, nfp, axis_length, B20_mean, G2, I2)
    grad_grad_B_results = calculate_grad_grad_B_tensor(X1c, Y1s, Y1c, X20, X2s, X2c, Y20, Y2s, Y2c, Z20, Z2s, Z2c, iota_N, iota, curvature, torsion, G0, B0, sG, spsi, I2, G2, p2, B20, B2s, B2c, d_X1c_d_varphi, d_Y1s_d_varphi, d_Y1c_d_varphi, d_X20_d_varphi, d_X2s_d_varphi, d_X2c_d_varphi, d_Y20_d_varphi, d_Y2s_d_varphi, d_Y2c_d_varphi, d_Z20_d_varphi, d_Z2s_d_varphi, d_Z2c_d_varphi, d2_X1c_d_varphi2, d2_Y1s_d_varphi2, d2_Y1c_d_varphi2, d_curvature_d_varphi, d_torsion_d_varphi, nphi)
    r_singularity_results =  calculate_r_singularity(X1c, Y1s, Y1c, X20, X2s, X2c, Y20, Y2s, Y2c, Z20, Z2s, Z2c, iota_N, iota, G0, B0, curvature, torsion, nphi, sG, spsi, I2, G2, p2, B20, B2s, B2c, d_X1c_d_varphi, d_Y1s_d_varphi, d_Y1c_d_varphi, d_X20_d_varphi, d_X2s_d_varphi, d_X2c_d_varphi, d_Y20_d_varphi, d_Y2s_d_varphi, d_Y2c_d_varphi, d_Z20_d_varphi, d_Z2s_d_varphi, d_Z2c_d_varphi, d2_X1c_d_varphi2, d2_Y1s_d_varphi2, d2_Y1c_d_varphi2, d_curvature_d_varphi, d_torsion_d_varphi)

   #if helicity == 0:
    # X2c_untwisted = X2c
    # X2s_untwisted = X2s
    # X20_untwisted = X20
    # Y20_untwisted = Y20
    # Y2s_untwisted = Y2s
    # Y2c_untwisted = Y2c
    # Z20_untwisted = Z20
    # Z2s_untwisted = Z2s
    # Z2c_untwisted = Z2c
    #else:
    angle = -helicity * nfp * varphi
    sinangle = jnp.sin(angle)
    cosangle = jnp.cos(angle)
    X20_untwisted = X20
    Y20_untwisted = Y20
    Z20_untwisted = Z20
    sinangle = jnp.sin(2*angle)
    cosangle = jnp.cos(2*angle)

    X2s_untwisted = X2s *   cosangle  + X2c * sinangle

    X2c_untwisted = X2s * (-sinangle) + X2c * cosangle
    Y2s_untwisted = Y2s *   cosangle  + Y2c * sinangle
    Y2c_untwisted = Y2s * (-sinangle) + Y2c * cosangle
    Z2s_untwisted = Z2s *   cosangle  + Z2c * sinangle
    Z2c_untwisted = Z2s * (-sinangle) + Z2c * cosangle
    
    r2_results = R2_results(N_helicity, G2, d_curvature_d_varphi, d_torsion_d_varphi, d_X20_d_varphi, d_X2s_d_varphi, d_X2c_d_varphi, d_Y20_d_varphi, d_Y2s_d_varphi, d_Y2c_d_varphi, d_Z20_d_varphi, d_Z2s_d_varphi, d_Z2c_d_varphi, d2_X1c_d_varphi2, d2_Y1c_d_varphi2, d2_Y1s_d_varphi2, V1, V2, V3, X20, X2s, X2c, Y20, Y2s, Y2c, Z20, Z2s, Z2c, beta_1s, B20, X20_untwisted, X2s_untwisted, X2c_untwisted, Y20_untwisted, Y2s_untwisted, Y2c_untwisted, Z20_untwisted, Z2s_untwisted, Z2c_untwisted)
    
    return Complete_R2_Results(mercier_results, grad_grad_B_results, r2_results, r_singularity_results)