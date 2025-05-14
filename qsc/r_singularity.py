#!/usr/bin/env python3

"""
Functions for computing the maximum r at which the flux surfaces
become singular.
"""

from typing import NamedTuple
import logging
import warnings
import numpy as np
import jax
import jax.numpy as jnp

#from .util import Struct, fourier_minimum

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class State(NamedTuple):
    r_singularity_basic_vs_varphi: jnp.ndarray
    r_singularity_vs_varphi: jnp.ndarray 
    r_singularity_residual_sqnorm: jnp.ndarray
    r_singularity_theta_vs_varphi: jnp.ndarray
    coefficients: jnp.ndarray
    real_parts: jnp.ndarray
    imag_parts: jnp.ndarray
    rc: jnp.ndarray
    K0: jnp.ndarray
    K2s: jnp.ndarray
    K4s: jnp.ndarray
    K2c: jnp.ndarray
    K4c: jnp.ndarray
    jphi: jnp.ndarray
    sin2theta: jnp.ndarray
    get_cos_from_cos2: jnp.ndarray
    abs_costheta_or_abs_sintheta: jnp.ndarray
    cos2theta: jnp.ndarray
    g0: jnp.ndarray
    g1c: jnp.ndarray
    g20: jnp.ndarray
    g2c: jnp.ndarray
    g2s: jnp.ndarray
    
    
    

def calculate_r_singularity(X1c, Y1s, Y1c, X20, X2s, X2c, Y20, Y2s, Y2c, Z20, Z2s, Z2c, iotaN, iota, G0, B0, curvature, torsion, nphi, sG, spsi, I2, G2, p2, B20, B2s, B2c, d_X1c_d_varphi, d_Y1s_d_varphi, d_Y1c_d_varphi, d_X20_d_varphi, d_X2s_d_varphi, d_X2c_d_varphi, d_Y20_d_varphi, d_Y2s_d_varphi, d_Y2c_d_varphi, d_Z20_d_varphi, d_Z2s_d_varphi, d_Z2c_d_varphi, d2_X1c_d_varphi2, d2_Y1s_d_varphi2, d2_Y1c_d_varphi2, d_curvature_d_varphi, d_torsion_d_varphi): 
        

    # Shorthand
    
    X1c = X1c
    Y1s = Y1s
    Y1c = Y1c

    X20 = X20
    X2s = X2s
    X2c = X2c

    Y20 = Y20
    Y2s = Y2s
    Y2c = Y2c

    Z20 = Z20
    Z2s = Z2s
    Z2c = Z2c

    iota_N0 = iotaN
    iota = iota
    lp = jnp.abs(G0) / B0

    curvature = curvature
    torsion = torsion

    nphi = nphi
    sign_G = sG
    sign_psi = spsi
    B0 = B0
    G0 = G0
    I2 = I2
    G2 = G2
    p2 = p2

    B20 = B20
    B2s = B2s
    B2c = B2c

    d_X1c_d_varphi = d_X1c_d_varphi
    d_Y1s_d_varphi = d_Y1s_d_varphi
    d_Y1c_d_varphi = d_Y1c_d_varphi

    d_X20_d_varphi = d_X20_d_varphi
    d_X2s_d_varphi = d_X2s_d_varphi
    d_X2c_d_varphi = d_X2c_d_varphi

    d_Y20_d_varphi = d_Y20_d_varphi
    d_Y2s_d_varphi = d_Y2s_d_varphi
    d_Y2c_d_varphi = d_Y2c_d_varphi

    d_Z20_d_varphi = d_Z20_d_varphi
    d_Z2s_d_varphi = d_Z2s_d_varphi
    d_Z2c_d_varphi = d_Z2c_d_varphi

    d2_X1c_d_varphi2 = d2_X1c_d_varphi2
    d2_Y1s_d_varphi2 = d2_Y1s_d_varphi2
    d2_Y1c_d_varphi2 = d2_Y1c_d_varphi2

    d_curvature_d_varphi = d_curvature_d_varphi
    d_torsion_d_varphi = d_torsion_d_varphi

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
    
    coefficients = coefficients.at[:, 4].set( 4*(K4c*K4c + K4s*K4s))

    coefficients = coefficients.at[:, 3].set( 4*(K4s*K2c - K2s*K4c))

    coefficients = coefficients.at[:, 2].set (K2s*K2s + K2c*K2c - 4*K0*K4c - 4*K4c*K4c - 4*K4s*K4s)

    coefficients = coefficients.at[:, 1].set( 2*K0*K2s + 2*K4c*K2s - 4*K4s*K2c)

    coefficients = coefficients.at[:, 0].set( (K0 + K4c)*(K0 + K4c) - K2c*K2c)

    for jphi in range(nphi):
        # Solve for the roots of the quartic polynomial:
       
        roots = jnp.roots(coefficients[jphi, :], strip_zeros=False) # Do I need to reverse the order of the coefficients?
       

        real_parts = jnp.real(roots)
        imag_parts = jnp.imag(roots)

        logger.debug('jphi={} g0={} g1c={} g20={} g2s={} g2c={} K0={} K2s={} K2c={} K4s={} K4c={} coefficients={} real={} imag={}'.format(jphi, g0[jphi], g1c[jphi], g20[jphi], g2s[jphi], g2c[jphi], K0[jphi], K2s[jphi], K2c[jphi], K4s[jphi], K4c[jphi], coefficients[jphi,:], real_parts, imag_parts))

        # This huge number indicates a true solution has not yet been found.
        rc = 1e+100

        for jr in range(4):
            # Loop over the roots of the equation for w.

            

            sin2theta = real_parts[jr]

            

            # Determine varpi by checking which choice gives the smaller residual in the K equation
            abs_cos2theta = jnp.sqrt(1 - sin2theta * sin2theta)
            residual_if_varpi_plus  = jnp.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] *   abs_cos2theta \
                                             + K4s[jphi] * 2 * sin2theta *   abs_cos2theta  + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
            residual_if_varpi_minus = jnp.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] * (-abs_cos2theta) \
                                             + K4s[jphi] * 2 * sin2theta * (-abs_cos2theta) + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))

            if residual_if_varpi_plus > residual_if_varpi_minus:
                varpi = -1
            else:
                varpi = 1

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
            if get_cos_from_cos2:
                abs_costheta = jnp.sqrt(0.5*(1 + cos2theta))
            else:
                abs_sintheta = jnp.sqrt(0.5 * (1 - cos2theta))
                
            logger.debug("  jr={}  sin2theta={}  cos2theta={}".format(jr, sin2theta, cos2theta))
            for varsigma in [-1, 1]:
                if get_cos_from_cos2:
                    costheta = varsigma * abs_costheta
                    sintheta = sin2theta / (2 * costheta)
                else:
                    sintheta = varsigma * abs_sintheta
                    costheta = sin2theta / (2 * sintheta)
                logger.debug("    varsigma={}  costheta={}  sintheta={}".format(varsigma, costheta, sintheta))

                # Sanity test
                if jnp.abs(costheta*costheta + sintheta*sintheta - 1) > 1e-13:
                    msg = "Error! sintheta={} costheta={} jphi={} jr={} sin2theta={} cos2theta={} abs(costheta*costheta + sintheta*sintheta - 1)={}".format(sintheta, costheta, jphi, jr, sin2theta, cos2theta, jnp.abs(costheta*costheta + sintheta*sintheta - 1))
                    logger.error(msg)
                    raise RuntimeError(msg)

                """
                # Try to get r using the simpler method, the equation that is linear in r.
                denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
                if np.abs(denominator) > 1e-8:
                    # This method cannot be used if we would need to divide by 0
                    rr = g1c[jphi] * sintheta / denominator
                    residual = g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta) # Residual in the equation sqrt(g)=0.
                    logger.debug("    Linear method: rr={}  residual={}".format(rr, residual))
                    if (rr>0) and np.abs(residual) < 1e-5:
                        if rr < rc:
                            # If this is a new minimum
                            rc = rr
                            sintheta_at_rc = sintheta
                            costheta_at_rc = costheta
                            logger.debug("      New minimum: rc={}".format(rc))
                else:
                    # Use the more complicated method to determine rr by solving a quadratic equation.
                    quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
                    quadratic_B = costheta * g1c[jphi]
                    quadratic_C = g0[jphi]
                    radical = np.sqrt(quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C)
                    for sign_quadratic in [-1, 1]:
                        rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
                        residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                        logger.debug("    Quadratic method: A={} B={} C={} radicand={}, radical={}  rr={}  residual={}".format(quadratic_A, quadratic_B, quadratic_C, quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C, radical, rr, residual))
                        if (rr>0) and np.abs(residual) < 1e-5:
                            if rr < rc:
                                # If this is a new minimum
                                rc = rr
                                sintheta_at_rc = sintheta
                                costheta_at_rc = costheta
                                logger.debug("      New minimum: rc={}".format(rc))
                """

                # Try to get r using the simpler method, the equation that is linear in r.
                linear_solutions = []
                denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
                if jnp.abs(denominator) > 1e-8:
                    # This method cannot be used if we would need to divide by 0
                    rr = g1c[jphi] * sintheta / denominator
                    residual = g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta) # Residual in the equation sqrt(g)=0.
                    logger.debug("    Linear method: rr={}  residual={}".format(rr, residual))
                    if (rr>0) and jnp.abs(residual) < 1e-5:
                        linear_solutions = [rr]
                        
                # Use the more complicated method to determine rr by solving a quadratic equation.
                quadratic_solutions = []
                quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
                quadratic_B = costheta * g1c[jphi]
                quadratic_C = g0[jphi]
                radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
                if jnp.abs(quadratic_A) < 1e-13:
                    rr = -quadratic_C / quadratic_B
                    residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                    logger.debug("    Quadratic method but A is small: A={} rr={}  residual={}".format(quadratic_A, rr, residual))
                    if rr > 0 and jnp.abs(residual) < 1e-5:
                        quadratic_solutions.append(rr)
                else:
                    # quadratic_A is nonzero, so we can apply the quadratic formula.
                    # I've seen a case where radicand is <0 due I think to under-resolution in phi.
                    if radicand >= 0:
                        radical = jnp.sqrt(radicand)
                        for sign_quadratic in [-1, 1]:
                            rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
                            residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                            logger.debug("    Quadratic method: A={} B={} C={} radicand={}, radical={}  rr={}  residual={}".format(quadratic_A, quadratic_B, quadratic_C, quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C, radical, rr, residual))
                            if (rr>0) and jnp.abs(residual) < 1e-5:
                                quadratic_solutions.append(rr)

                logger.debug("    # linear solutions={}  # quadratic solutions={}".format(len(linear_solutions), len(quadratic_solutions)))
                if len(quadratic_solutions) > 1:
                    # Pick the smaller one
                    quadratic_solutions = [jnp.min(quadratic_solutions)]
                    
                # If both methods find a solution, check that they agree:
                if len(linear_solutions) > 0 and len(quadratic_solutions) > 0:
                    diff = jnp.abs(linear_solutions[0] - quadratic_solutions[0])
                    logger.debug("  linear solution={}  quadratic solution={}  diff={}".format(linear_solutions[0], quadratic_solutions[0], diff))
                    if diff > 1e-5:
                        warnings.warn("  Difference between linear solution {} and quadratic solution {} is {}".format(linear_solutions[0], quadratic_solutions[0], diff))
                        
                    #assert np.abs(linear_solutions[0] - quadratic_solutions[0]) < 1e-5, "Difference between linear solution {} and quadratic solution {} is {}".format(linear_solutions[0], quadratic_solutions[0], linear_solutions[0] - quadratic_solutions[0])
                    
                # Prefer the quadratic solution
                rr = -1
                if len(quadratic_solutions) > 0:
                    rr = quadratic_solutions[0]
                elif len(linear_solutions) > 0:
                    rr = linear_solutions[0]

                if rr > 0 and rr < rc:
                    # This is a new minimum
                    rc = rr
                    sintheta_at_rc = sintheta
                    costheta_at_rc = costheta
                    logger.debug("      New minimum: rc={}".format(rc))
                    
        r_singularity_basic_vs_varphi[jphi] = rc
        #r_singularity_Newton_solve()
        r_singularity_vs_varphi[jphi] = rc
        r_singularity_residual_sqnorm[jphi] = 0 # Newton_residual_sqnorm
        r_singularity_theta_vs_varphi[jphi] = 0 # theta FIX ME!!

    self.r_singularity_vs_varphi = r_singularity_vs_varphi
    self.inv_r_singularity_vs_varphi = 1 / r_singularity_vs_varphi
    self.r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi
    self.r_singularity = jnp.min(r_singularity_vs_varphi)    
    self.r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi
    self.r_singularity_residual_sqnorm = r_singularity_residual_sqnorm
    


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
        
    #state = np.array(20)
    
    #state[18] = g0
    #state[19] = g1c
    #state[20] = g20
    #state[21] = g2c
    #state[22] = g2s
  

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
    
    coefficients = coefficients.at[:, 4].set(4*(K4c*K4c + K4s*K4s))

    coefficients = coefficients.at[:, 3].set(4*(K4s*K2c - K2s*K4c))

    coefficients = coefficients.at[: ,2].set(K2s*K2s + K2c*K2c - 4*K0*K4c - 4*K4c*K4c - 4*K4s*K4s)

    coefficients = coefficients.at[: ,1].set(2*K0*K2s + 2*K4c*K2s - 4*K4s*K2c)

    coefficients = coefficients.at[: ,0].set((K0 + K4c)*(K0 + K4c) - K2c*K2c)

      
    #state = r_singularity_basic_vs_varphi, r_singularity_vs_varphi, r_singularity_residual_sqnorm, r_singularity_theta_vs_varphi, real_parts
    
    
    #state[0] =r_singularity_basic_vs_varphi 
    #state[1] =r_singularity_vs_varphi 
    #state[2] =r_singularity_residual_sqnorm 
    #state[3] = r_singularity_theta_vs_varphi
    
   
    #state[8] = K0  
    #state[9] = K2s
    #state[10] = K4s
    #state[11] = K2c
    #state[12] = K4c
    
    state = State(
        r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi,
        r_singularity_vs_varphi = r_singularity_vs_varphi,
        r_singularity_residual_sqnorm = r_singularity_residual_sqnorm,
        r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi,
        coefficients = jnp.zeros((61, 5)),
        real_parts = jnp.zeros(4),
        imag_parts = jnp.zeros(4),
        rc = jnp.array(0.0),   # Scalar placeholder
        K0 = K0,
        K2s = K2s,
        K4s = K4s,
        K2c = K2c,
        K4c = K4c,
        jphi= jnp.array(0, dtype=jnp.int64),  # Integer value
        sin2theta = jnp.array(0.0),  # Scalar
        get_cos_from_cos2 = jnp.array(0.0),
        abs_costheta_or_abs_sintheta = jnp.array(0.0),
        cos2theta = jnp.array(0.0),
        g0 = jnp.zeros(61),
        g1c = jnp.zeros(61),
        g20 = jnp.zeros(61),
        g2c = jnp.zeros(61),
        g2s = jnp.zeros(61),
    )
     
    jax.lax.fori_loop(0, nphi, jphi_loop, state)

    r_singularity_vs_varphi = r_singularity_vs_varphi
    inv_r_singularity_vs_varphi = 1 / r_singularity_vs_varphi
    r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi
    r_singularity = jnp.min(r_singularity_vs_varphi)    
    r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi
    r_singularity_residual_sqnorm = r_singularity_residual_sqnorm
    
    return r_singularity_vs_varphi, inv_r_singularity_vs_varphi, r_singularity_basic_vs_varphi, r_singularity, r_singularity_theta_vs_varphi, r_singularity_residual_sqnorm

def jr_loop(j, state): 
    """
    called by jphi_loop to loop over the roots of the quartic polynomial in order to find the best root for r_singularity.

    Args:
        j (_type_): _description_
        state (_type_): _description_
    """
    #real_parts = state[5]
    #imag_parts = state[6]
    
    real_parts = state.real_parts
    imag_parts = state.imag_parts
    
        # Loop over the roots of the equation for w.
    jr = j
        # If root is not purely real, skip it.
    sin2theta = real_parts[jr]
    
    #state[14] = sin2theta                
    state = state._replace(sin2theta = sin2theta)
    
    jax.lax.cond(jnp.logical_or(jnp.abs(imag_parts[jr]) > 1e-7, jnp.abs(sin2theta) > 1),
            lambda _: compute(state), 
            lambda _: None, 
            None)
    
    return state

def jphi_loop(j, state):
    jphi = j 
    #state[13] = jphi 
    # Solve for the roots of the quartic polynomial:
   #coefficients = state[4]    
    #r_singularity_basic_vs_varphi = state[0] 
    #r_singularity_vs_varphi = state[1] 
    #r_singularity_residual_sqnorm  = state[2] 
    #r_singularity_theta_vs_varphi = state[3]
   
    coefficients = state.coefficients
    r_singularity_basic_vs_varphi = state.r_singularity_basic_vs_varphi
    r_singularity_vs_varphi = state.r_singularity_vs_varphi
    r_singularity_residual_sqnorm = state.r_singularity_residual_sqnorm
    r_singularity_theta_vs_varphi = state.r_singularity_theta_vs_varphi
    
    roots = jax.numpy.roots(coefficients[jphi, :], strip_zeros = False) # Do I need to reverse the order of the coefficients?        
    real_parts = jnp.real(roots)
    imag_parts = jnp.imag(roots)

    #state[5] = real_parts
    #state[6] = imag_parts
    # This huge number indicates a true solution has not yet been found.
    rc = 1e+100
    #state[7] = rc 
    
    state = state._replace(jphi = jphi, real_parts = real_parts, imag_parts = imag_parts, rc = rc)
        
    jax.lax.fori_loop(0, 4, jr_loop, state)   
                    
    r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi.at[jphi].set(rc)
    #r_singularity_Newton_solve()
    r_singularity_vs_varphi = r_singularity_vs_varphi.at[jphi].set(rc)
    r_singularity_residual_sqnorm = r_singularity_residual_sqnorm.at[jphi].set(0) # Newton_residual_sqnorm
    r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi.at[jphi].set(0) # theta FIX ME!!
    
    state = state._replace(r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi, r_singularity_vs_varphi = r_singularity_vs_varphi, r_singularity_residual_sqnorm = r_singularity_residual_sqnorm, r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi)
    
    return state
 

    
    
def compute(state):   
    #sin2theta = state[14]
    #jphi = state[13]
    #K0 = state[8] 
    #K2s = state[9]
    #K4s = state[10]
    #K2c = state[11]
    #K4c = state[12]
   
    sin2theta = state.sin2theta
    jphi = state.jphi
    K0 = state.K0
    K2s = state.K2s
    K4s = state.K4s
    K2c = state.K2c
    K4c = state.K4c
    
    
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
    #state[17] = cos2theta
    
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
    #state[15] = get_cos_from_cos2 
              
              
              #if get_cos_from_cos2:
               #   abs_costheta = np.sqrt(0.5*(1 + cos2theta))
              #else:
               #   abs_sintheta = np.sqrt(0.5 * (1 - cos2theta))
               
    abs_costheta_or_abs_sintheta = jax.lax.cond(get_cos_from_cos2, 
                                                        lambda _: jnp.sqrt(0.5*(1 + cos2theta)), 
                                                        lambda _: jnp.sqrt(0.5 * (1 - cos2theta)), 
                                                        None)
    #state[16] = abs_costheta_or_abs_sintheta 
    
    state = state._replace(cos2theta = cos2theta, get_cos_from_cos2 = get_cos_from_cos2, abs_costheta_or_abs_sintheta = abs_costheta_or_abs_sintheta)
    
    jax.lax.fori_loop(0,2, varsigma_loop, state)
    
    
    

def true_get_cos_from_cos2(varsigma, abs_costheta_or_abs_sintheta, sin2theta):
    costheta = varsigma * abs_costheta_or_abs_sintheta
    sintheta = sin2theta / (2 * costheta)
    return costheta, sintheta

def false_get_cos_from_cos2(varsigma, abs_costheta_or_abs_sintheta, sin2theta):
    sintheta = varsigma * abs_costheta_or_abs_sintheta
    costheta = sin2theta / (2 * sintheta)
    return costheta, sintheta
                
def large_denom_case(g1c, jphi, sintheta, denominator, g0, costheta, g20, g2s, sin2theta, g2c, cos2theta): 
    rr = g1c[jphi] * sintheta / denominator
    residual = g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta) # Residual in the equation sqrt(g)=0. 
    return rr, residual
                    
def small_denom_case(): 
    return (jnp.array(0, dtype=jnp.float64), jnp.array(0, dtype=jnp.float64))  

def small_quadratic_A_case(quadratic_C, quadratic_B, g1c, jphi, sintheta, g2s, cos2theta, g2c, sin2theta, quadratic_solutions):
    rr = -quadratic_C / quadratic_B
    residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                       
    #will face issue with dynamically sized arrays
    quadratic_solutions = jax.lax.cond(jnp.logical_and((rr>0), jnp.abs(residual) < 1e-5) , 
                                     lambda _: jnp.concatenate([quadratic_solutions, jnp.array([rr])]), 
                                     lambda _: jnp.concatenate([quadratic_solutions, jnp.array([jnp.nan])]),
                                     None)
    
    quadratic_solutions = jnp.concatenate([quadratic_solutions, jnp.array([jnp.nan])])
    
    return quadratic_solutions
                        
def large_quadratic_A_case(quadratic_A, quadratic_B, g1c, jphi, sintheta, g2s, cos2theta, g2c, sin2theta, quadratic_solutions, radicand):
    radical = jnp.sqrt(radicand)
    sign_quadratic = -1
    rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
    residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                        
    #will face issue with dynamically sized arrays
    quadratic_solutions = jax.lax.cond(jnp.logical_and((rr>0), jnp.abs(residual) < 1e-5) , 
                                     lambda _: jnp.concatenate([quadratic_solutions, jnp.array([rr])]), 
                                     lambda _: jnp.concatenate([quadratic_solutions, jnp.array([jnp.nan])]),
                                     None)
                        
    sign_quadratic = 1
    rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
    residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                        
    #will face issue with dynamically sized arrays
    quadratic_solutions = jax.lax.cond(jnp.logical_and((rr>0), jnp.abs(residual) < 1e-5) , 
                                     lambda _: jnp.concatenate([quadratic_solutions, jnp.array([rr])]), 
                                     lambda _: jnp.concatenate([quadratic_solutions, jnp.array([jnp.nan])]),
                                     None)
    
    return quadratic_solutions
    
    
                     
def varsigma_loop(j, state): 
    #get_cos_from_cos2 = state[15] 
    #abs_costheta_or_abs_sintheta = state[16] 
    #sin2theta = state[14]
    #jphi = state[13] 
    #cos2theta = state[17]
    #g2s = state[22]
    #g2c = state[21]
    #g1c = state[19]
    #g0 = state[18]
    #g20 = state[20]
    
    get_cos_from_cos2 = state.get_cos_from_cos2
    abs_costheta_or_abs_sintheta = state.abs_costheta_or_abs_sintheta
    sin2theta = state.sin2theta
    jphi = state.jphi
    cos2theta = state.cos2theta
    g2s = state.g2s
    g2c = state.g2c
    g1c = state.g1c
    g0 = state.g0
    g20 = state.g20
               
    varsigma = jax.lax.cond(j==0, 
                        lambda _: -1, 
                        lambda _: 1, 
                        None)
                  
             
    costheta, sintheta = jax.lax.cond(get_cos_from_cos2,
                                lambda _: true_get_cos_from_cos2(varsigma, abs_costheta_or_abs_sintheta, sin2theta), 
                                lambda _: false_get_cos_from_cos2(varsigma, abs_costheta_or_abs_sintheta, sin2theta), 
                                None)
                

    # Try to get r using the simpler method, the equation that is linear in r.
    linear_solutions = []
    denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
                    
                     
    rr, residual = jax.lax.cond(jnp.abs(denominator) > 1e-8,
                            lambda _: large_denom_case(g1c, jphi, sintheta, denominator, g0, costheta, g20, g2s, sin2theta, g2c, cos2theta),
                            lambda _: small_denom_case(),
                            None)
                    
    cond = (rr > 0) & (jnp.abs(residual) < 1e-5) & (jnp.abs(denominator) > 1e-8)
    linear_solutions = jax.lax.cond(cond,
                                lambda _: jnp.array([rr]), 
                                lambda _: jnp.array([jnp.nan]),
                                None)
                
                            
    # Use the more complicated method to determine rr by solving a quadratic equation.
    quadratic_solutions = jnp.array([])
    quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
    quadratic_B = costheta * g1c[jphi]
    quadratic_C = g0[jphi]
    radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
                             
                                    
    #logic has been inverted to avoid else with if              
    quadratic_solutions = jax.lax.cond(jnp.logical_and(jnp.abs(quadratic_A) >= 1e-13, radicand >= 0),
                            lambda _: large_quadratic_A_case(quadratic_A, quadratic_B, g1c, jphi, sintheta, g2s, cos2theta, g2c, sin2theta, quadratic_solutions, radicand), 
                            lambda _: small_quadratic_A_case(quadratic_C, quadratic_B, g1c, jphi, sintheta, g2s, cos2theta, g2c, sin2theta, quadratic_solutions),
                            None)
   
    quadratic_solution = jax.lax.cond(quadratic_solutions.size > 0, 
                                    lambda _: jnp.min(quadratic_solutions), 
                                    lambda _: jnp.nan, 
                                    None)
                    
    # Prefer the quadratic solution
    rr = -1
                    
    rr = jax.lax.cond(quadratic_solutions.size > 0,
                    lambda _: quadratic_solution, 
                    lambda _: jax.lax.cond(linear_solutions.size > 0,
                                        lambda _: linear_solutions[0],
                                        lambda _: -1.0,
                                        None),
                    None)
    rc = state.rc

    rc = jax.lax.cond(jnp.logical_and(rr > 0, rr < rc),
                lambda _: rr, 
                lambda _: rc, 
                None)
                   
    state = state._replace(rc = rc)
    
    return state