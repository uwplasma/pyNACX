#!/usr/bin/env python3

"""
Functions for computing the maximum r at which the flux surfaces
become singular.
"""

import logging
import warnings
import numpy as np
import jax
import jax.numpy as jnp

#from .util import Struct, fourier_minimum

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_r_singularity(X1c, Y1s, Y1c, X20, X2s, X2c, Y20, Y2s, Y2c, Z20, Z2s, Z2c, iotaN, iota, G0, B0, curvature, torsion, nphi, sG, spsi, I2, G2, p2, B20, B2s, B2c, d_X1c_d_varphi, d_Y1s_d_varphi, d_Y1c_d_varphi, d_X20_d_varphi, d_X2s_d_varphi, d_X2c_d_varphi, d_Y20_d_varphi, d_Y2s_d_varphi, d_Y2c_d_varphi, d_Z20_d_varphi, d_Z2s_d_varphi, d_Z2c_d_varphi, d2_X1c_d_varphi2, d2_Y1s_d_varphi2, d2_Y1c_d_varphi2, d_curvature_d_varphi, d_torsion_d_varphi):
    """
    """

    iota_N0 = iotaN
    iota = iota
    lp = jnp.abs(G0) / B0

    nphi = nphi
    sign_G = sG
    sign_psi = spsi

    r_singularity_basic_vs_varphi = jnp.zeros(nphi)
    r_singularity_vs_varphi = jnp.zeros(nphi)
    r_singularity_residual_sqnorm = jnp.zeros(nphi)
    r_singularity_theta_vs_varphi = jnp.zeros(nphi)
    
    # Write sqrt(g) = r * [g0 + r*g1c*cos(theta) + (r^2)*(g20 + g2s*sin(2*theta) + g2c*cos(2*theta) + ...]
    # The coefficients are evaluated in "20200322-02 Max r for Garren Boozer.nb", in the section "Order r^2 construction, quasisymmetry"

    g0 = lp * X1c * Y1s

    #g1s = -2*X20*Y1c + 2*X2c*Y1c + 2*X2s*Y1s + 2*X1c*Y20 - 2*X1c*Y2c
    # g1s vanishes for quasisymmetry.

    g1c = lp*(-2*X2s*Y1c + 2*X20*Y1s + 2*X2c*Y1s + 2*X1c*Y2s - X1c*X1c*Y1s*curvature)

    g20 = -4*lp*X2s*Y2c + 4*lp*X2c*Y2s + lp*X1c*X2s*Y1c*curvature - \
        2*lp*X1c*X20*Y1s*curvature - lp*X1c*X2c*Y1s*curvature - \
        lp*X1c*X1c*Y2s*curvature + 2*lp*Y1c*Y1s*Z2c*torsion - \
        lp*X1c*X1c*Z2s*torsion - lp*Y1c*Y1c*Z2s*torsion + lp*Y1s*Y1s*Z2s*torsion - \
        Y1s*Z20*d_X1c_d_varphi - Y1s*Z2c*d_X1c_d_varphi + \
        Y1c*Z2s*d_X1c_d_varphi - X1c*Z2s*d_Y1c_d_varphi - \
        X1c*Z20*d_Y1s_d_varphi + X1c*Z2c*d_Y1s_d_varphi + \
        X1c*Y1s*d_Z20_d_varphi
    
    g2c = -4*lp*X2s*Y20 + 4*lp*X20*Y2s + \
        lp*X1c*X2s*Y1c*curvature - lp*X1c*X20*Y1s*curvature - \
        2*lp*X1c*X2c*Y1s*curvature - lp*X1c*X1c*Y2s*curvature + \
        2*lp*Y1c*Y1s*Z20*torsion - lp*X1c*X1c*Z2s*torsion - \
        lp*Y1c*Y1c*Z2s*torsion - lp*Y1s*Y1s*Z2s*torsion - \
        Y1s*Z20*d_X1c_d_varphi - Y1s*Z2c*d_X1c_d_varphi + \
        Y1c*Z2s*d_X1c_d_varphi - X1c*Z2s*d_Y1c_d_varphi + \
        X1c*Z20*d_Y1s_d_varphi - X1c*Z2c*d_Y1s_d_varphi + \
        X1c*Y1s*d_Z2c_d_varphi
    
    g2s = 4*lp*X2c*Y20 - 4*lp*X20*Y2c + \
        lp*X1c*X20*Y1c*curvature - lp*X1c*X2c*Y1c*curvature - \
        2*lp*X1c*X2s*Y1s*curvature - lp*X1c*X1c*Y20*curvature + \
        lp*X1c*X1c*Y2c*curvature - lp*X1c*X1c*Z20*torsion - \
        lp*Y1c*Y1c*Z20*torsion + lp*Y1s*Y1s*Z20*torsion + \
        lp*X1c*X1c*Z2c*torsion + lp*Y1c*Y1c*Z2c*torsion + \
        lp*Y1s*Y1s*Z2c*torsion + Y1c*Z20*d_X1c_d_varphi - \
        Y1c*Z2c*d_X1c_d_varphi - Y1s*Z2s*d_X1c_d_varphi - \
        X1c*Z20*d_Y1c_d_varphi + X1c*Z2c*d_Y1c_d_varphi - \
        X1c*Z2s*d_Y1s_d_varphi + X1c*Y1s*d_Z2s_d_varphi
    # highorder is default False and not called with true 
    # We consider the system sqrt(g) = 0 and
    # d (sqrtg) / d theta = 0.
    # We algebraically eliminate r in "20200322-02 Max r for Garren Boozer.nb", in the section
    # "Keeping first 3 orders in the Jacobian".
    # We end up with the form in "20200322-01 Max r for GarrenBoozer.docx":
    # K0 + K2s*sin(2*theta) + K2c*cos(2*theta) + K4s*sin(4*theta) + K4c*cos(4*theta) = 0.

    K0 = 2*g1c*g1c*g20 - 3*g1c*g1c*g2c + 8*g0*g2c*g2c + 8*g0*g2s*g2s

    K2s = 2*g1c*g1c*g2s

    K2c = -2*g1c*g1c*g20 + 2*g1c*g1c*g2c

    K4s = g1c*g1c*g2s - 16*g0*g2c*g2s

    K4c = g1c*g1c*g2c - 8*g0*g2c*g2c + 8*g0*g2s*g2s

    coefficients =jnp.zeros((nphi,5))
    
    coefficients = coefficients.at[:, 4].set(4*(K4c*K4c + K4s*K4s))

    coefficients = coefficients.at[:, 3].set(4*(K4s*K2c - K2s*K4c))

    coefficients = coefficients.at[: ,2].set(K2s*K2s + K2c*K2c - 4*K0*K4c - 4*K4c*K4c - 4*K4s*K4s)

    coefficients = coefficients.at[: ,1].set(2*K0*K2s + 2*K4c*K2s - 4*K4s*K2c)

    coefficients = coefficients.at[: ,0].set((K0 + K4c)*(K0 + K4c) - K2c*K2c)
    
    """"""
    
    def body(j, carry):
        
        jphi = j
        
        roots = jax.numpy.roots(coefficients[jphi, :], strip_zeros = False) # Do I need to reverse the order of the coefficients?
        
        real_parts = jnp.real(roots)
        
        # This huge number indicates a true solution has not yet been found.
        rc = 1e+100
        
        # jr is at index 0
        jr = 0
        sin2theta = real_parts[jr]
        
        # Determine varpi by checking which choice gives the smaller residual in the K equation
        abs_cos2theta = jnp.sqrt(1 - sin2theta * sin2theta)
        residual_if_varpi_plus  = jnp.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] *   abs_cos2theta \
                                             + K4s[jphi] * 2 * sin2theta *   abs_cos2theta  + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
        residual_if_varpi_minus = jnp.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] * (-abs_cos2theta) \
                                             + K4s[jphi] * 2 * sin2theta * (-abs_cos2theta) + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
        
        
        varpi = jax.lax.cond(residual_if_varpi_plus > residual_if_varpi_minus, 
                             lambda _: -1, 
                             lambda _: 1, 
                             None)
        
        cos2theta = varpi * abs_cos2theta
        
        # The next few lines give an older method for computing varpi, which has problems in edge cases
        # where w (the root of the quartic polynomial) is very close to +1 or -1, giving varpi
        # not very close to +1 or -1 due to bad loss of precision.
        #
        #varpi_denominator = ((K4s*2*sin2theta + K2c) * sqrt(1 - sin2theta*sin2theta))
        #if (abs(varpi_denominator) < 1e-8) print *,"WARNING!!! varpi_denominator=",varpi_denominator
        #varpi = -(K0 + K2s * sin2theta + K4c*(1 - 2*sin2theta*sin2theta)) / varpi_denominator
        #if (abs(varpi*varpi-1) > 1e-3) print *,"WARNING!!! abs(varpi*varpi-1) =",abs(varpi*varpi-1)
        #varpi = nint(varpi) ! Ensure varpi is exactly either +1 or -1.
        #cos2theta = varpi * sqrt(1 - sin2theta*sin2theta)

        # To get (sin theta, cos theta) from (sin 2 theta, cos 2 theta), we consider two cases to
        # avoid precision loss when cos2theta is added to or subtracted from 1:
        get_cos_from_cos2 = cos2theta > 0
            
        abs_costheta = jax.lax.cond(get_cos_from_cos2, 
                                    lambda _: jnp.sqrt(0.5*(1 + cos2theta)), 
                                    lambda _: float(0), 
                                    None)
        
        abs_sintheta = jax.lax.cond(get_cos_from_cos2, 
                                    lambda _: float(0), 
                                    lambda _:jnp.sqrt(0.5 * (1 - cos2theta)), 
                                    None)
        
        varsigma = -1 
        
        def costheta_sintheta_decleration_get_cos_from_cos2_true_case():
          costheta = varsigma * abs_costheta
          sintheta = sin2theta / (2 * costheta)
          return costheta, sintheta
        
        def costheta_sintheta_decleration_get_cos_from_cos2_false_case(): 
          sintheta = varsigma * abs_sintheta
          costheta = sin2theta / (2 * sintheta)
          return costheta, sintheta
        
        costheta, sintheta = jax.lax.cond(get_cos_from_cos2, 
                                          lambda _ : costheta_sintheta_decleration_get_cos_from_cos2_true_case(), 
                                          lambda _ : costheta_sintheta_decleration_get_cos_from_cos2_false_case(), 
                                          None)
        
        linear_solutions = jnp.empty((0,), dtype=jnp.float32)

        denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
        
        def large_denominator_case(): 
          rr = g1c[jphi] * sintheta / denominator
          residual = g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta)
          return rr, residual
        
        
        rr, residual = jax.lax.cond(jnp.abs(denominator) > 1e-8,
                                    lambda _: large_denominator_case(),
                                    lambda _: (float(0), float(0)), 
                                    None)
        
        linear_solutions = jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                                         lambda _: jnp.concatenate([linear_solutions, jnp.atleast_1d(rr)]), 
                                         lambda _: linear_solutions, 
                                         None)
        
        quadratic_solutions = jnp.array()
        quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
        quadratic_B = costheta * g1c[jphi]
        quadratic_C = g0[jphi]
        
        radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
        
        def large_quadratic_case():
            rr = -quadratic_C / quadratic_B
            residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
            
            append_cond = jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5)
            jax.lax.cond(append_cond, 
                         lambda _: linear_solutions.append(rr), 
                         None, 
                         None)
            return rr, residual
        
        def small_quadratic_case():
            rr, residual = jax.lax.cond(radicand >= 0,
                         lambda _ : large_radicand_case(), 
                         lambda _ : rr, radicand, 
                         None)
            return rr, residual
        
        def large_radicand_case(): 
          radical = jnp.sqrt(radicand)
          sign_quadratic = -1 
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
          
          sign_quadratic = 1
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
            
        rr, residual = jax.lax.cond(jnp.abs(quadratic_A) < 1e-13,
                                    lambda _: large_quadratic_case(),
                                    lambda _: small_quadratic_case(),
                                    None)
        
        quadratic_solutions = jax.lax.cond(len(linear_solutions) > 1,
                     lambda _: [jnp.min(quadratic_solutions)], 
                     lambda _: None, 
                     None)
        
        rr = -1
        rr = jax.lax.cond(len(quadratic_solutions) > 0,
                          lambda _: quadratic_solutions[0],
                          lambda _: jax.lax.cond(len(linear_solutions) > 0,
                                                lambda _: linear_solutions[0],
                                                lambda _: 0, 
                                                None),
                          None)
        
        rc, sintheta_at_rc, costheta_at_rc = jax.lax.cond(jnp.logical_and(rr > 0, rr<rc), 
                                                          lambda _: (rr, sintheta, costheta), 
                                                          lambda _: None, 
                                                          None)
        
        varsigma = 1
        
        costheta, sintheta = jax.lax.cond(get_cos_from_cos2, 
                                          lambda _ : (varsigma * abs_costheta, sin2theta / (2 * costheta)), 
                                          lambda _ : (sin2theta / (2 * sintheta), varsigma * abs_sintheta), 
                                          None)
        
        linear_solutions = jnp.array()
        denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
        
        rr, residual = jax.lax.cond(jnp.abs(denominator) > 1e-8,
                                    lambda _: g1c[jphi] * sintheta / denominator, g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta),
                                    lambda _: 0, 
                                    None)
        
        quadratic_solutions = jnp.array()
        quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
        quadratic_B = costheta * g1c[jphi]
        quadratic_C = g0[jphi]
        
        radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
        
        def large_quadratic_case():
            rr = -quadratic_C / quadratic_B
            residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
            
            append_cond = jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5)
            jax.lax.cond(append_cond, 
                         lambda _: linear_solutions.append(rr), 
                         None, 
                         None)
            return rr, residual
        
        def small_quadratic_case():
            rr, residual = jax.lax.cond(radicand >= 0,
                         lambda _ : large_radicand_case(), 
                         lambda _ : rr, radicand, 
                         None)
            return rr, residual
        
        def large_radicand_case(): 
          radical = jnp.sqrt(radicand)
          sign_quadratic = -1 
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
          
          sign_quadratic = 1
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
            
        rr, residual = jax.lax.cond(jnp.abs(quadratic_A) < 1e-13,
                                    lambda _: large_quadratic_case(),
                                    lambda _: small_quadratic_case(),
                                    None)
        
        quadratic_solutions = jax.lax.cond(len(linear_solutions) > 1,
                     lambda _: [jnp.min(quadratic_solutions)], 
                     lambda _: None, 
                     None)
        
        rr = -1
        rr = jax.lax.cond(len(quadratic_solutions) > 0,
                          lambda _: quadratic_solutions[0],
                          lambda _: jax.lax.cond(len(linear_solutions) > 0,
                                                lambda _: linear_solutions[0],
                                                lambda _: 0, 
                                                None),
                          None)
        
        rc, sintheta_at_rc, costheta_at_rc = jax.lax.cond(jnp.logical_and(rr > 0, rr<rc), 
                                                          lambda _: (rr, sintheta, costheta), 
                                                          lambda _: None, 
                                                          None)
        
        
                    
        r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi.at[jphi].set(rc)
    
        r_singularity_vs_varphi = r_singularity_vs_varphi.at[jphi].set(rc)
        r_singularity_residual_sqnorm = r_singularity_residual_sqnorm.at[jphi].set(0) # Newton_residual_sqnorm
        r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi.at[jphi].set(0) # theta FIX ME!!
        # jr is at index 1
        jr = 1
        sin2theta = real_parts[jr]
        
        # Determine varpi by checking which choice gives the smaller residual in the K equation
        abs_cos2theta = jnp.sqrt(1 - sin2theta * sin2theta)
        residual_if_varpi_plus  = jnp.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] *   abs_cos2theta \
                                             + K4s[jphi] * 2 * sin2theta *   abs_cos2theta  + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
        residual_if_varpi_minus = jnp.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] * (-abs_cos2theta) \
                                             + K4s[jphi] * 2 * sin2theta * (-abs_cos2theta) + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
        
        
        varpi = jax.lax.cond(residual_if_varpi_plus > residual_if_varpi_minus, 
                             lambda _: -1, 
                             lambda _: 1, 
                             None)
        
        cos2theta = varpi * abs_cos2theta
        
        # The next few lines give an older method for computing varpi, which has problems in edge cases
        # where w (the root of the quartic polynomial) is very close to +1 or -1, giving varpi
        # not very close to +1 or -1 due to bad loss of precision.
        #
        #varpi_denominator = ((K4s*2*sin2theta + K2c) * sqrt(1 - sin2theta*sin2theta))
        #if (abs(varpi_denominator) < 1e-8) print *,"WARNING!!! varpi_denominator=",varpi_denominator
        #varpi = -(K0 + K2s * sin2theta + K4c*(1 - 2*sin2theta*sin2theta)) / varpi_denominator
        #if (abs(varpi*varpi-1) > 1e-3) print *,"WARNING!!! abs(varpi*varpi-1) =",abs(varpi*varpi-1)
        #varpi = nint(varpi) ! Ensure varpi is exactly either +1 or -1.
        #cos2theta = varpi * sqrt(1 - sin2theta*sin2theta)

        # To get (sin theta, cos theta) from (sin 2 theta, cos 2 theta), we consider two cases to
        # avoid precision loss when cos2theta is added to or subtracted from 1:
        get_cos_from_cos2 = cos2theta > 0
            
        abs_costheta = jax.lax.cond(get_cos_from_cos2, 
                                    lambda _: np.sqrt(0.5*(1 + cos2theta)), 
                                    lambda _: 0, 
                                    None)
        
        abs_sintheta = jax.lax.cond(get_cos_from_cos2, 
                                    lambda _: 0, 
                                    lambda _: np.sqrt(0.5 * (1 - cos2theta)), 
                                    None)
        
        varsigma = -1 
        
        costheta, sintheta = jax.lax.cond(get_cos_from_cos2, 
                                          lambda _ : (varsigma * abs_costheta, sin2theta / (2 * costheta)), 
                                          lambda _ : (sin2theta / (2 * sintheta), varsigma * abs_sintheta), 
                                          None)
        
        linear_solutions = jnp.array()
        denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
        
        rr, residual = jax.lax.cond(jnp.abs(denominator) > 1e-8,
                                    lambda _: g1c[jphi] * sintheta / denominator, g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta),
                                    lambda _: 0, 
                                    None)
        
        quadratic_solutions = jnp.array()
        quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
        quadratic_B = costheta * g1c[jphi]
        quadratic_C = g0[jphi]
        
        radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
        
        def large_quadratic_case():
            rr = -quadratic_C / quadratic_B
            residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
            
            append_cond = jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5)
            jax.lax.cond(append_cond, 
                         lambda _: linear_solutions.append(rr), 
                         None, 
                         None)
            return rr, residual
        
        def small_quadratic_case():
            rr, residual = jax.lax.cond(radicand >= 0,
                         lambda _ : large_radicand_case(), 
                         lambda _ : rr, radicand, 
                         None)
            return rr, residual
        
        def large_radicand_case(): 
          radical = jnp.sqrt(radicand)
          sign_quadratic = -1 
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
          
          sign_quadratic = 1
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
            
        rr, residual = jax.lax.cond(jnp.abs(quadratic_A) < 1e-13,
                                    lambda _: large_quadratic_case(),
                                    lambda _: small_quadratic_case(),
                                    None)
        
        quadratic_solutions = jax.lax.cond(len(linear_solutions) > 1,
                     lambda _: [jnp.min(quadratic_solutions)], 
                     lambda _: None, 
                     None)
        
        rr = -1
        rr = jax.lax.cond(len(quadratic_solutions) > 0,
                          lambda _: quadratic_solutions[0],
                          lambda _: jax.lax.cond(len(linear_solutions) > 0,
                                                lambda _: linear_solutions[0],
                                                lambda _: 0, 
                                                None),
                          None)
        
        rc, sintheta_at_rc, costheta_at_rc = jax.lax.cond(jnp.logical_and(rr > 0, rr<rc), 
                                                          lambda _: (rr, sintheta, costheta), 
                                                          lambda _: None, 
                                                          None)
        
        varsigma = 1
        
        costheta, sintheta = jax.lax.cond(get_cos_from_cos2, 
                                          lambda _ : (varsigma * abs_costheta, sin2theta / (2 * costheta)), 
                                          lambda _ : (sin2theta / (2 * sintheta), varsigma * abs_sintheta), 
                                          None)
        
        linear_solutions = jnp.array()
        denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
        
        rr, residual = jax.lax.cond(jnp.abs(denominator) > 1e-8,
                                    lambda _: g1c[jphi] * sintheta / denominator, g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta),
                                    lambda _: 0, 
                                    None)
        
        quadratic_solutions = jnp.array()
        quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
        quadratic_B = costheta * g1c[jphi]
        quadratic_C = g0[jphi]
        
        radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
        
        def large_quadratic_case():
            rr = -quadratic_C / quadratic_B
            residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
            
            append_cond = jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5)
            jax.lax.cond(append_cond, 
                         lambda _: linear_solutions.append(rr), 
                         None, 
                         None)
            return rr, residual
        
        def small_quadratic_case():
            rr, residual = jax.lax.cond(radicand >= 0,
                         lambda _ : large_radicand_case(), 
                         lambda _ : rr, radicand, 
                         None)
            return rr, residual
        
        def large_radicand_case(): 
          radical = jnp.sqrt(radicand)
          sign_quadratic = -1 
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
          
          sign_quadratic = 1
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
            
        rr, residual = jax.lax.cond(jnp.abs(quadratic_A) < 1e-13,
                                    lambda _: large_quadratic_case(),
                                    lambda _: small_quadratic_case(),
                                    None)
        
        quadratic_solutions = jax.lax.cond(len(linear_solutions) > 1,
                     lambda _: [jnp.min(quadratic_solutions)], 
                     lambda _: None, 
                     None)
        
        rr = -1
        rr = jax.lax.cond(len(quadratic_solutions) > 0,
                          lambda _: quadratic_solutions[0],
                          lambda _: jax.lax.cond(len(linear_solutions) > 0,
                                                lambda _: linear_solutions[0],
                                                lambda _: 0, 
                                                None),
                          None)
        
        rc, sintheta_at_rc, costheta_at_rc = jax.lax.cond(jnp.logical_and(rr > 0, rr<rc), 
                                                          lambda _: (rr, sintheta, costheta), 
                                                          lambda _: None, 
                                                          None)
        
        
                    
        r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi.at[jphi].set(rc)
    
        r_singularity_vs_varphi = r_singularity_vs_varphi.at[jphi].set(rc)
        r_singularity_residual_sqnorm = r_singularity_residual_sqnorm.at[jphi].set(0) # Newton_residual_sqnorm
        r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi.at[jphi].set(0) # theta FIX ME!!
        
        # jr is at index 2
        jr = 2
        sin2theta = real_parts[jr]
        
        # Determine varpi by checking which choice gives the smaller residual in the K equation
        abs_cos2theta = jnp.sqrt(1 - sin2theta * sin2theta)
        residual_if_varpi_plus  = jnp.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] *   abs_cos2theta \
                                             + K4s[jphi] * 2 * sin2theta *   abs_cos2theta  + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
        residual_if_varpi_minus = jnp.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] * (-abs_cos2theta) \
                                             + K4s[jphi] * 2 * sin2theta * (-abs_cos2theta) + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
        
        
        varpi = jax.lax.cond(residual_if_varpi_plus > residual_if_varpi_minus, 
                             lambda _: -1, 
                             lambda _: 1, 
                             None)
        
        cos2theta = varpi * abs_cos2theta
        
        # The next few lines give an older method for computing varpi, which has problems in edge cases
        # where w (the root of the quartic polynomial) is very close to +1 or -1, giving varpi
        # not very close to +1 or -1 due to bad loss of precision.
        #
        #varpi_denominator = ((K4s*2*sin2theta + K2c) * sqrt(1 - sin2theta*sin2theta))
        #if (abs(varpi_denominator) < 1e-8) print *,"WARNING!!! varpi_denominator=",varpi_denominator
        #varpi = -(K0 + K2s * sin2theta + K4c*(1 - 2*sin2theta*sin2theta)) / varpi_denominator
        #if (abs(varpi*varpi-1) > 1e-3) print *,"WARNING!!! abs(varpi*varpi-1) =",abs(varpi*varpi-1)
        #varpi = nint(varpi) ! Ensure varpi is exactly either +1 or -1.
        #cos2theta = varpi * sqrt(1 - sin2theta*sin2theta)

        # To get (sin theta, cos theta) from (sin 2 theta, cos 2 theta), we consider two cases to
        # avoid precision loss when cos2theta is added to or subtracted from 1:
        get_cos_from_cos2 = cos2theta > 0
            
        abs_costheta = jax.lax.cond(get_cos_from_cos2, 
                                    lambda _: np.sqrt(0.5*(1 + cos2theta)), 
                                    lambda _: 0, 
                                    None)
        
        abs_sintheta = jax.lax.cond(get_cos_from_cos2, 
                                    lambda _: 0, 
                                    lambda _: np.sqrt(0.5 * (1 - cos2theta)), 
                                    None)
        
        varsigma = -1 
        
        costheta, sintheta = jax.lax.cond(get_cos_from_cos2, 
                                          lambda _ : (varsigma * abs_costheta, sin2theta / (2 * costheta)), 
                                          lambda _ : (sin2theta / (2 * sintheta), varsigma * abs_sintheta), 
                                          None)
        
        linear_solutions = jnp.array()
        denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
        
        rr, residual = jax.lax.cond(jnp.abs(denominator) > 1e-8,
                                    lambda _: g1c[jphi] * sintheta / denominator, g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta),
                                    lambda _: 0, 
                                    None)
        
        quadratic_solutions = jnp.array()
        quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
        quadratic_B = costheta * g1c[jphi]
        quadratic_C = g0[jphi]
        
        radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
        
        def large_quadratic_case():
            rr = -quadratic_C / quadratic_B
            residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
            
            append_cond = jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5)
            jax.lax.cond(append_cond, 
                         lambda _: linear_solutions.append(rr), 
                         None, 
                         None)
            return rr, residual
        
        def small_quadratic_case():
            rr, residual = jax.lax.cond(radicand >= 0,
                         lambda _ : large_radicand_case(), 
                         lambda _ : rr, radicand, 
                         None)
            return rr, residual
        
        def large_radicand_case(): 
          radical = jnp.sqrt(radicand)
          sign_quadratic = -1 
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
          
          sign_quadratic = 1
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
            
        rr, residual = jax.lax.cond(jnp.abs(quadratic_A) < 1e-13,
                                    lambda _: large_quadratic_case(),
                                    lambda _: small_quadratic_case(),
                                    None)
        
        quadratic_solutions = jax.lax.cond(len(linear_solutions) > 1,
                     lambda _: [jnp.min(quadratic_solutions)], 
                     lambda _: None, 
                     None)
        
        rr = -1
        rr = jax.lax.cond(len(quadratic_solutions) > 0,
                          lambda _: quadratic_solutions[0],
                          lambda _: jax.lax.cond(len(linear_solutions) > 0,
                                                lambda _: linear_solutions[0],
                                                lambda _: 0, 
                                                None),
                          None)
        
        rc, sintheta_at_rc, costheta_at_rc = jax.lax.cond(jnp.logical_and(rr > 0, rr<rc), 
                                                          lambda _: (rr, sintheta, costheta), 
                                                          lambda _: None, 
                                                          None)
        
        varsigma = 1
        
        costheta, sintheta = jax.lax.cond(get_cos_from_cos2, 
                                          lambda _ : (varsigma * abs_costheta, sin2theta / (2 * costheta)), 
                                          lambda _ : (sin2theta / (2 * sintheta), varsigma * abs_sintheta), 
                                          None)
        
        linear_solutions = jnp.array()
        denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
        
        rr, residual = jax.lax.cond(jnp.abs(denominator) > 1e-8,
                                    lambda _: g1c[jphi] * sintheta / denominator, g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta),
                                    lambda _: 0, 
                                    None)
        
        quadratic_solutions = jnp.array()
        quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
        quadratic_B = costheta * g1c[jphi]
        quadratic_C = g0[jphi]
        
        radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
        
        def large_quadratic_case():
            rr = -quadratic_C / quadratic_B
            residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
            
            append_cond = jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5)
            jax.lax.cond(append_cond, 
                         lambda _: linear_solutions.append(rr), 
                         None, 
                         None)
            return rr, residual
        
        def small_quadratic_case():
            rr, residual = jax.lax.cond(radicand >= 0,
                         lambda _ : large_radicand_case(), 
                         lambda _ : rr, radicand, 
                         None)
            return rr, residual
        
        def large_radicand_case(): 
          radical = jnp.sqrt(radicand)
          sign_quadratic = -1 
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
          
          sign_quadratic = 1
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
            
        rr, residual = jax.lax.cond(jnp.abs(quadratic_A) < 1e-13,
                                    lambda _: large_quadratic_case(),
                                    lambda _: small_quadratic_case(),
                                    None)
        
        quadratic_solutions = jax.lax.cond(len(linear_solutions) > 1,
                     lambda _: [jnp.min(quadratic_solutions)], 
                     lambda _: None, 
                     None)
        
        rr = -1
        rr = jax.lax.cond(len(quadratic_solutions) > 0,
                          lambda _: quadratic_solutions[0],
                          lambda _: jax.lax.cond(len(linear_solutions) > 0,
                                                lambda _: linear_solutions[0],
                                                lambda _: 0, 
                                                None),
                          None)
        
        rc, sintheta_at_rc, costheta_at_rc = jax.lax.cond(jnp.logical_and(rr > 0, rr<rc), 
                                                          lambda _: (rr, sintheta, costheta), 
                                                          lambda _: None, 
                                                          None)
        
        
                    
        r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi.at[jphi].set(rc)
    
        r_singularity_vs_varphi = r_singularity_vs_varphi.at[jphi].set(rc)
        r_singularity_residual_sqnorm = r_singularity_residual_sqnorm.at[jphi].set(0) # Newton_residual_sqnorm
        r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi.at[jphi].set(0) # theta FIX ME!!

        # jr is at index 3
        jr = 3
        sin2theta = real_parts[jr]
        
        # Determine varpi by checking which choice gives the smaller residual in the K equation
        abs_cos2theta = jnp.sqrt(1 - sin2theta * sin2theta)
        residual_if_varpi_plus  = jnp.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] *   abs_cos2theta \
                                             + K4s[jphi] * 2 * sin2theta *   abs_cos2theta  + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
        residual_if_varpi_minus = jnp.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] * (-abs_cos2theta) \
                                             + K4s[jphi] * 2 * sin2theta * (-abs_cos2theta) + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
        
        
        varpi = jax.lax.cond(residual_if_varpi_plus > residual_if_varpi_minus, 
                             lambda _: -1, 
                             lambda _: 1, 
                             None)
        
        cos2theta = varpi * abs_cos2theta
        
        # The next few lines give an older method for computing varpi, which has problems in edge cases
        # where w (the root of the quartic polynomial) is very close to +1 or -1, giving varpi
        # not very close to +1 or -1 due to bad loss of precision.
        #
        #varpi_denominator = ((K4s*2*sin2theta + K2c) * sqrt(1 - sin2theta*sin2theta))
        #if (abs(varpi_denominator) < 1e-8) print *,"WARNING!!! varpi_denominator=",varpi_denominator
        #varpi = -(K0 + K2s * sin2theta + K4c*(1 - 2*sin2theta*sin2theta)) / varpi_denominator
        #if (abs(varpi*varpi-1) > 1e-3) print *,"WARNING!!! abs(varpi*varpi-1) =",abs(varpi*varpi-1)
        #varpi = nint(varpi) ! Ensure varpi is exactly either +1 or -1.
        #cos2theta = varpi * sqrt(1 - sin2theta*sin2theta)

        # To get (sin theta, cos theta) from (sin 2 theta, cos 2 theta), we consider two cases to
        # avoid precision loss when cos2theta is added to or subtracted from 1:
        get_cos_from_cos2 = cos2theta > 0
            
        abs_costheta = jax.lax.cond(get_cos_from_cos2, 
                                    lambda _: np.sqrt(0.5*(1 + cos2theta)), 
                                    lambda _: 0, 
                                    None)
        
        abs_sintheta = jax.lax.cond(get_cos_from_cos2, 
                                    lambda _: 0, 
                                    lambda _: np.sqrt(0.5 * (1 - cos2theta)), 
                                    None)
        
        varsigma = -1 
        
        costheta, sintheta = jax.lax.cond(get_cos_from_cos2, 
                                          lambda _ : (varsigma * abs_costheta, sin2theta / (2 * costheta)), 
                                          lambda _ : (sin2theta / (2 * sintheta), varsigma * abs_sintheta), 
                                          None)
        
        linear_solutions = jnp.array()
        denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
        
        rr, residual = jax.lax.cond(jnp.abs(denominator) > 1e-8,
                                    lambda _: g1c[jphi] * sintheta / denominator, g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta),
                                    lambda _: 0, 
                                    None)
        
        quadratic_solutions = jnp.array()
        quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
        quadratic_B = costheta * g1c[jphi]
        quadratic_C = g0[jphi]
        
        radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
        
        def large_quadratic_case():
            rr = -quadratic_C / quadratic_B
            residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
            
            append_cond = jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5)
            jax.lax.cond(append_cond, 
                         lambda _: linear_solutions.append(rr), 
                         None, 
                         None)
            return rr, residual
        
        def small_quadratic_case():
            rr, residual = jax.lax.cond(radicand >= 0,
                         lambda _ : large_radicand_case(), 
                         lambda _ : rr, radicand, 
                         None)
            return rr, residual
        
        def large_radicand_case(): 
          radical = jnp.sqrt(radicand)
          sign_quadratic = -1 
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
          
          sign_quadratic = 1
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
            
        rr, residual = jax.lax.cond(jnp.abs(quadratic_A) < 1e-13,
                                    lambda _: large_quadratic_case(),
                                    lambda _: small_quadratic_case(),
                                    None)
        
        quadratic_solutions = jax.lax.cond(len(linear_solutions) > 1,
                     lambda _: [jnp.min(quadratic_solutions)], 
                     lambda _: None, 
                     None)
        
        rr = -1
        rr = jax.lax.cond(len(quadratic_solutions) > 0,
                          lambda _: quadratic_solutions[0],
                          lambda _: jax.lax.cond(len(linear_solutions) > 0,
                                                lambda _: linear_solutions[0],
                                                lambda _: 0, 
                                                None),
                          None)
        
        rc, sintheta_at_rc, costheta_at_rc = jax.lax.cond(jnp.logical_and(rr > 0, rr<rc), 
                                                          lambda _: (rr, sintheta, costheta), 
                                                          lambda _: None, 
                                                          None)
        
        varsigma = 1
        
        costheta, sintheta = jax.lax.cond(get_cos_from_cos2, 
                                          lambda _ : (varsigma * abs_costheta, sin2theta / (2 * costheta)), 
                                          lambda _ : (sin2theta / (2 * sintheta), varsigma * abs_sintheta), 
                                          None)
        
        linear_solutions = jnp.array()
        denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
        
        rr, residual = jax.lax.cond(jnp.abs(denominator) > 1e-8,
                                    lambda _: g1c[jphi] * sintheta / denominator, g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta),
                                    lambda _: 0, 
                                    None)
        
        quadratic_solutions = jnp.array()
        quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
        quadratic_B = costheta * g1c[jphi]
        quadratic_C = g0[jphi]
        
        radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
        
        def large_quadratic_case():
            rr = -quadratic_C / quadratic_B
            residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
            
            append_cond = jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5)
            jax.lax.cond(append_cond, 
                         lambda _: linear_solutions.append(rr), 
                         None, 
                         None)
            return rr, residual
        
        def small_quadratic_case():
            rr, residual = jax.lax.cond(radicand >= 0,
                         lambda _ : large_radicand_case(), 
                         lambda _ : rr, radicand, 
                         None)
            return rr, residual
        
        def large_radicand_case(): 
          radical = jnp.sqrt(radicand)
          sign_quadratic = -1 
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
          
          sign_quadratic = 1
          rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
          residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
          jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5),
                       lambda _: quadratic_solutions.append(rr), 
                       None, 
                       None) 
            
        rr, residual = jax.lax.cond(jnp.abs(quadratic_A) < 1e-13,
                                    lambda _: large_quadratic_case(),
                                    lambda _: small_quadratic_case(),
                                    None)
        
        quadratic_solutions = jax.lax.cond(len(linear_solutions) > 1,
                     lambda _: [jnp.min(quadratic_solutions)], 
                     lambda _: None, 
                     None)
        
        rr = -1
        rr = jax.lax.cond(len(quadratic_solutions) > 0,
                          lambda _: quadratic_solutions[0],
                          lambda _: jax.lax.cond(len(linear_solutions) > 0,
                                                lambda _: linear_solutions[0],
                                                lambda _: 0, 
                                                None),
                          None)
        
        rc, sintheta_at_rc, costheta_at_rc = jax.lax.cond(jnp.logical_and(rr > 0, rr<rc), 
                                                          lambda _: (rr, sintheta, costheta), 
                                                          lambda _: None, 
                                                          None)
        
        
                    
        r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi.at[jphi].set(rc)
    
        r_singularity_vs_varphi = r_singularity_vs_varphi.at[jphi].set(rc)
        r_singularity_residual_sqnorm = r_singularity_residual_sqnorm.at[jphi].set(0) # Newton_residual_sqnorm
        r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi.at[jphi].set(0) # theta FIX ME!!

    carry = r_singularity_basic_vs_varphi, r_singularity_vs_varphi, r_singularity_residual_sqnorm, r_singularity_theta_vs_varphi
    
    jax.lax.fori_loop(0, nphi, body, carry)

    r_singularity_vs_varphi = r_singularity_vs_varphi
    inv_r_singularity_vs_varphi = 1 / r_singularity_vs_varphi
    r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi
    r_singularity = jnp.min(r_singularity_vs_varphi)    
    r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi
    r_singularity_residual_sqnorm = r_singularity_residual_sqnorm
    
    return r_singularity_vs_varphi, inv_r_singularity_vs_varphi, r_singularity_basic_vs_varphi, r_singularity, r_singularity_theta_vs_varphi, r_singularity_residual_sqnorm



  
  



def new_calculate_r_singularity(X1c, Y1s, Y1c, X20, X2s, X2c, Y20, Y2s, Y2c, Z20, Z2s, Z2c, iotaN, iota, G0, B0, curvature, torsion, nphi, sG, spsi, I2, G2, p2, B20, B2s, B2c, d_X1c_d_varphi, d_Y1s_d_varphi, d_Y1c_d_varphi, d_X20_d_varphi, d_X2s_d_varphi, d_X2c_d_varphi, d_Y20_d_varphi, d_Y2s_d_varphi, d_Y2c_d_varphi, d_Z20_d_varphi, d_Z2s_d_varphi, d_Z2c_d_varphi, d2_X1c_d_varphi2, d2_Y1s_d_varphi2, d2_Y1c_d_varphi2, d_curvature_d_varphi, d_torsion_d_varphi): 
    """
    """


    iota_N0 = iotaN
    iota = iota
    lp = jnp.abs(G0) / B0

    nphi = nphi
    sign_G = sG
    sign_psi = spsi

    r_singularity_basic_vs_varphi = jnp.zeros(nphi)
    r_singularity_vs_varphi = jnp.zeros(nphi)
    r_singularity_residual_sqnorm = jnp.zeros(nphi)
    r_singularity_theta_vs_varphi = jnp.zeros(nphi)
    
    # Write sqrt(g) = r * [g0 + r*g1c*cos(theta) + (r^2)*(g20 + g2s*sin(2*theta) + g2c*cos(2*theta) + ...]
    # The coefficients are evaluated in "20200322-02 Max r for Garren Boozer.nb", in the section "Order r^2 construction, quasisymmetry"

    g0 = lp * X1c * Y1s

    #g1s = -2*X20*Y1c + 2*X2c*Y1c + 2*X2s*Y1s + 2*X1c*Y20 - 2*X1c*Y2c
    # g1s vanishes for quasisymmetry.

    g1c = lp*(-2*X2s*Y1c + 2*X20*Y1s + 2*X2c*Y1s + 2*X1c*Y2s - X1c*X1c*Y1s*curvature)

    g20 = -4*lp*X2s*Y2c + 4*lp*X2c*Y2s + lp*X1c*X2s*Y1c*curvature - \
        2*lp*X1c*X20*Y1s*curvature - lp*X1c*X2c*Y1s*curvature - \
        lp*X1c*X1c*Y2s*curvature + 2*lp*Y1c*Y1s*Z2c*torsion - \
        lp*X1c*X1c*Z2s*torsion - lp*Y1c*Y1c*Z2s*torsion + lp*Y1s*Y1s*Z2s*torsion - \
        Y1s*Z20*d_X1c_d_varphi - Y1s*Z2c*d_X1c_d_varphi + \
        Y1c*Z2s*d_X1c_d_varphi - X1c*Z2s*d_Y1c_d_varphi - \
        X1c*Z20*d_Y1s_d_varphi + X1c*Z2c*d_Y1s_d_varphi + \
        X1c*Y1s*d_Z20_d_varphi
    
    g2c = -4*lp*X2s*Y20 + 4*lp*X20*Y2s + \
        lp*X1c*X2s*Y1c*curvature - lp*X1c*X20*Y1s*curvature - \
        2*lp*X1c*X2c*Y1s*curvature - lp*X1c*X1c*Y2s*curvature + \
        2*lp*Y1c*Y1s*Z20*torsion - lp*X1c*X1c*Z2s*torsion - \
        lp*Y1c*Y1c*Z2s*torsion - lp*Y1s*Y1s*Z2s*torsion - \
        Y1s*Z20*d_X1c_d_varphi - Y1s*Z2c*d_X1c_d_varphi + \
        Y1c*Z2s*d_X1c_d_varphi - X1c*Z2s*d_Y1c_d_varphi + \
        X1c*Z20*d_Y1s_d_varphi - X1c*Z2c*d_Y1s_d_varphi + \
        X1c*Y1s*d_Z2c_d_varphi
    
    g2s = 4*lp*X2c*Y20 - 4*lp*X20*Y2c + \
        lp*X1c*X20*Y1c*curvature - lp*X1c*X2c*Y1c*curvature - \
        2*lp*X1c*X2s*Y1s*curvature - lp*X1c*X1c*Y20*curvature + \
        lp*X1c*X1c*Y2c*curvature - lp*X1c*X1c*Z20*torsion - \
        lp*Y1c*Y1c*Z20*torsion + lp*Y1s*Y1s*Z20*torsion + \
        lp*X1c*X1c*Z2c*torsion + lp*Y1c*Y1c*Z2c*torsion + \
        lp*Y1s*Y1s*Z2c*torsion + Y1c*Z20*d_X1c_d_varphi - \
        Y1c*Z2c*d_X1c_d_varphi - Y1s*Z2s*d_X1c_d_varphi - \
        X1c*Z20*d_Y1c_d_varphi + X1c*Z2c*d_Y1c_d_varphi - \
        X1c*Z2s*d_Y1s_d_varphi + X1c*Y1s*d_Z2s_d_varphi
  

    # We consider the system sqrt(g) = 0 and
    # d (sqrtg) / d theta = 0.
    # We algebraically eliminate r in "20200322-02 Max r for Garren Boozer.nb", in the section
    # "Keeping first 3 orders in the Jacobian".
    # We end up with the form in "20200322-01 Max r for GarrenBoozer.docx":
    # K0 + K2s*sin(2*theta) + K2c*cos(2*theta) + K4s*sin(4*theta) + K4c*cos(4*theta) = 0.

    K0 = 2*g1c*g1c*g20 - 3*g1c*g1c*g2c + 8*g0*g2c*g2c + 8*g0*g2s*g2s

    K2s = 2*g1c*g1c*g2s

    K2c = -2*g1c*g1c*g20 + 2*g1c*g1c*g2c

    K4s = g1c*g1c*g2s - 16*g0*g2c*g2s

    K4c = g1c*g1c*g2c - 8*g0*g2c*g2c + 8*g0*g2s*g2s

    coefficients = jnp.zeros((nphi,5))
    
    coefficients[:, 4] = 4*(K4c*K4c + K4s*K4s)

    coefficients[:, 3] = 4*(K4s*K2c - K2s*K4c)

    coefficients[:, 2] = K2s*K2s + K2c*K2c - 4*K0*K4c - 4*K4c*K4c - 4*K4s*K4s

    coefficients[:, 1] = 2*K0*K2s + 2*K4c*K2s - 4*K4s*K2c

    coefficients[:, 0] = (K0 + K4c)*(K0 + K4c) - K2c*K2c

    def jphi_loop(j, state):
        jphi = j 
        # Solve for the roots of the quartic polynomial:
        
        roots = jax.numpy.roots(coefficients[jphi, :], strip_zeros = False) # Do I need to reverse the order of the coefficients?        
        real_parts = jnp.real(roots)
        imag_parts = jnp.imag(roots)
 
        # This huge number indicates a true solution has not yet been found.
        rc = 1e+100
        
        def jr_loop(j, state): 
            # Loop over the roots of the equation for w.
            jr = j
            # If root is not purely real, skip it.
            sin2theta = real_parts[jr]
                
            def compute():   
                # Determine varpi by checking which choice gives the smaller residual in the K equation
                abs_cos2theta = np.sqrt(1 - sin2theta * sin2theta)
                residual_if_varpi_plus  = np.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] *   abs_cos2theta \
                                             + K4s[jphi] * 2 * sin2theta *   abs_cos2theta  + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
                residual_if_varpi_minus = np.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] * (-abs_cos2theta) \
                                             + K4s[jphi] * 2 * sin2theta * (-abs_cos2theta) + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))


                varpi = jax.lax.cond(residual_if_varpi_plus > residual_if_varpi_minus, 
                                lambda _: -1, 
                                lambda _: 1, 
                                None)

                cos2theta = varpi * abs_cos2theta

                # The next few lines give an older method for computing varpi, which has problems in edge cases
                # where w (the root of the quartic polynomial) is very close to +1 or -1, giving varpi
                # not very close to +1 or -1 due to bad loss of precision.
                #
                #varpi_denominator = ((K4s*2*sin2theta + K2c) * sqrt(1 - sin2theta*sin2theta))
                #if (abs(varpi_denominator) < 1e-8) print *,"WARNING!!! varpi_denominator=",varpi_denominator
                #varpi = -(K0 + K2s * sin2theta + K4c*(1 - 2*sin2theta*sin2theta)) / varpi_denominator
                #if (abs(varpi*varpi-1) > 1e-3) print *,"WARNING!!! abs(varpi*varpi-1) =",abs(varpi*varpi-1)
                #varpi = nint(varpi) ! Ensure varpi is exactly either +1 or -1.
                #cos2theta = varpi * sqrt(1 - sin2theta*sin2theta)

                # To get (sin theta, cos theta) from (sin 2 theta, cos 2 theta), we consider two cases to
                # avoid precision loss when cos2theta is added to or subtracted from 1:
                get_cos_from_cos2 = cos2theta > 0
              
              
              #if get_cos_from_cos2:
               #   abs_costheta = np.sqrt(0.5*(1 + cos2theta))
              #else:
               #   abs_sintheta = np.sqrt(0.5 * (1 - cos2theta))
               
                abs_costheta_or_abs_sintheta = jax.lax.cond(get_cos_from_cos2, 
                                                        lambda _: jnp.sqrt(0.5*(1 + cos2theta)), 
                                                        lambda _: jnp.sqrt(0.5 * (1 - cos2theta)), 
                                                        None)
                
                def varsigma_loop(j): 
                    
                    varisgma = jax.lax.cond(j==0, 
                                            lambda _: -1, 
                                            lambda _: 1, 
                                            None)
                  
                    def true_get_cos_from_cos2(): 
                        costheta = varsigma * abs_costheta_or_abs_sintheta
                        sintheta = sin2theta / (2 * costheta)
                        return costheta, sintheta
                
                    def false_get_cos_from_cos2():
                        sintheta = varsigma * abs_costheta_or_abs_sintheta
                        costheta = sin2theta / (2 * sintheta)
                        return costheta, sintheta
              
                    costheta, sintheta = jax.lax.cond(get_cos_from_cos2,
                                                  lambda _: true_get_cos_from_cos2(), 
                                                  lambda _: false_get_cos_from_cos2(), 
                                                  None)
                

                    # Try to get r using the simpler method, the equation that is linear in r.
                    linear_solutions = []
                    denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
                    
                    def large_denom_case():
                        rr = g1c[jphi] * sintheta / denominator
                        residual = g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta) # Residual in the equation sqrt(g)=0. 
                        return rr, residual
                    
                    def small_denom_case(): 
                        return rr, residual  
                      
                    rr, residual = jax.lax.cond(jnp.abs(denominator) > 1e-8,
                                            lambda _: large_denom_case(),
                                            lambda _: small_denom_case(),
                                            None)
                    
                    linear_solutions = jax.lax.cond(jnp.logical_and(rr > 0, jnp.abs(residual) < 1e-5, jnp.abs(denominator) > 1e-8),
                                                    lambda _: [rr], 
                                                    lambda _: [],
                                                    None)
                
                            
                    # Use the more complicated method to determine rr by solving a quadratic equation.
                    quadratic_solutions = []
                    quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
                    quadratic_B = costheta * g1c[jphi]
                    quadratic_C = g0[jphi]
                    radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
                    
                    def small_quadratic_A_case():
                        rr = -quadratic_C / quadratic_B
                        residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                       
                        #will face issue with dynamically sized arrays
                        jax.lax.cond(jnp.logical_and((rr>0), jnp.abs(residual) < 1e-5) , 
                                     lambda _: jnp.concatenate([rr], quadratic_solutions), 
                                     lambda _: None,
                                     None)
                    
                    def large_quadratic_A_case():
                        radical = jnp.sqrt(radicand)
                        sign_quadratic = -1
                        rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
                        residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                        
                        #will face issue with dynamically sized arrays
                        jax.lax.cond(jnp.logical_and((rr>0), jnp.abs(residual) < 1e-5) , 
                                     lambda _: jnp.concatenate([rr], quadratic_solutions), 
                                     lambda _: None,
                                     None)
                        
                        sign_quadratic = 1
                        rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
                        residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                        
                        #will face issue with dynamically sized arrays
                        jax.lax.cond(jnp.logical_and((rr>0), jnp.abs(residual) < 1e-5) , 
                                     lambda _: jnp.concatenate([rr], quadratic_solutions), 
                                     lambda _: None,
                                     None)
                                  
                    #logic has been inverted to avoid else with if              
                    rr, residual = jax.lax.cond(jnp.logical_and(jnp.abs(quadratic_A) >= 1e-13, radicand >= 0),
                                                large_quadratic_A_case, 
                                                small_quadratic_A_case,
                                                None)
                                                
        
                    quadratic_solutions = [np.min(quadratic_solutions)]
                    
                    # Prefer the quadratic solution
                    rr = -1
                    
                    rr = jax.lax.cond(quadratic_solutions.size > 0, lambda _: quadratic_solutions[0], lambda _: jax.lax.cond(linear_solutions.size > 0, lambda _: linear_solutions[0], lambda _: -1.0, None), None)

                    rc = jax.lax.cond(jnp.logical_and(rr > 0, rr < rc),
                                      lambda _: rr, 
                                      lambda _: rc, 
                                      None)
                    
                    return rc
                
                jax.lax.fori_loop(0,2, varsigma_loop, None)
                        
            def cont(): 
              return rc
              
            sin2theta = real_parts[jr]

            jax.lax.cond(jnp.logical_or(jnp.abs(imag_parts[jr]) > 1e-7, jnp.abs(sin2theta) > 1),
                         lambda _: compute, 
                         lambda _: cont, 
                         None)
        
        jax.lax.fori_loop(0, 4, jr_loop, None)   
                    
        r_singularity_basic_vs_varphi[jphi] = rc
        #r_singularity_Newton_solve()
        r_singularity_vs_varphi[jphi] = rc
        r_singularity_residual_sqnorm[jphi] = 0 # Newton_residual_sqnorm
        r_singularity_theta_vs_varphi[jphi] = 0 # theta FIX ME!!
        
        state = r_singularity_basic_vs_varphi, r_singularity_vs_varphi, r_singularity_residual_sqnorm, r_singularity_theta_vs_varphi
    
    state = r_singularity_basic_vs_varphi, r_singularity_vs_varphi, r_singularity_residual_sqnorm, r_singularity_theta_vs_varphi
    
    jax.lax.fori_loop(0, nphi, jphi_loop, state)

    r_singularity_vs_varphi = r_singularity_vs_varphi
    inv_r_singularity_vs_varphi = 1 / r_singularity_vs_varphi
    r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi
    r_singularity = np.min(r_singularity_vs_varphi)    
    r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi
    r_singularity_residual_sqnorm = r_singularity_residual_sqnorm
    
    return r_singularity_vs_varphi, inv_r_singularity_vs_varphi, r_singularity_basic_vs_varphi, r_singularity, r_singularity_theta_vs_varphi, r_singularity_residual_sqnorm
    
