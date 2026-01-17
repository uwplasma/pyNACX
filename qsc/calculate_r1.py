"""
This module contains the functions for solving the sigma equation
and computing diagnostics of the O(r^1) solution.
"""

import logging
import numpy as np
from qsc.types import *
from qsc.grad_B_tensor import calculate_grad_B_tensor
from .util import fourier_minimum, jax_fourier_minimum
from .newton import new_new_newton, newton
import jax.numpy as jnp
from jax import jacobian
from .calculate_r1_helpers import *

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
    

def solve_sigma_equation(nphi, sigma0, helicity, nfp, d_d_varphi, etabar_squared_over_curvature_squared, spsi, torsion, I2, B0, G0) -> Solve_Sigma_Equation_Results:
    """
    in progress solve sigma to equation that is unreliant on self
    """
    x0 = jnp.full(nphi, sigma0)
    x0 = x0.at[0].set(0) # Initial guess for iota
    
    def _residual(x):
        """
        Residual in the sigma equation, used for Newton's method.  x is
        the state vector, corresponding to sigma on the phi grid,
        except that the first element of x is actually iota.
        """
        sigma = jnp.copy(x)
        sigma = sigma.at[0].set(sigma0)

        iota = x[0]
        r = jnp.matmul(d_d_varphi, sigma) \
            + (iota + helicity * nfp) * \
            (etabar_squared_over_curvature_squared * etabar_squared_over_curvature_squared + 1 + sigma * sigma) \
            - 2 * etabar_squared_over_curvature_squared * (-spsi * torsion + I2 / B0) * G0 / B0
    
        return r
    
    def _jacobian(x):
        """
        Compute the Jacobian matrix for solving the sigma equation. x is
        the state vector, corresponding to sigma on the phi grid,
        except that the first element of x is actually iota.
        """
        sigma = x.at[0].set(sigma0)
        iota = x[0]

        # Create the diagonal update
        diag_vals = (iota + helicity * nfp) * 2 * sigma

        # Add the diagonal to the derivative matrix
        jac = d_d_varphi + jnp.diag(diag_vals)

        # Set the first column (d_residual / d_iota)
        iota_deriv = etabar_squared_over_curvature_squared ** 2 + 1 + sigma ** 2
        jac = jac.at[:, 0].set(iota_deriv)
        
        return jac
    
    jitted_new_new_newton = jax.jit(new_new_newton, static_argnames=["f", "jac", "niter", "tol", "nlinesearch"])
    
    sigma = jitted_new_new_newton(_residual, x0, _jacobian) # helper residual is a functon that runs without self but still returns r
    iota = sigma[0]
    iotaN = calc_iotaN(iota, helicity, nfp)
    sigma = sigma.at[0].set(sigma0)
    return Solve_Sigma_Equation_Results(sigma, iota, iotaN)

def r1_diagnostics(nfp, etabar, sG, spsi, curvature, sigma, helicity, varphi, X1s, X1c, d_l_d_phi, d_d_varphi, B0, d_l_d_varphi, tangent_cylindrical, normal_cylindrical, binormal_cylindrical, iotaN, torsion) -> Complete_R1_Results:
    """
    Compute various properties of the O(r^1) solution, once sigma and
    iota are solved for.
    """
    Y1s = sG * spsi * curvature / etabar
    Y1c = sG * spsi * curvature * sigma / etabar
    
    # If helicity is nonzero, then the original X1s/X1c/Y1s/Y1c variables are defined with respect to a "poloidal" angle that
    # is actually helical, with the theta=0 curve wrapping around the magnetic axis as you follow phi around toroidally. Therefore
    # here we convert to an untwisted poloidal angle, such that the theta=0 curve does not wrap around the axis.
   
    
    angle = calc_angle(helicity, nfp, varphi)
    sinangle = calc_sinangle(angle)
    cosangle = calc_cosangle(angle)
    
    X1s_untwisted = calc_X1s_untwisted(X1s, cosangle, X1c, sinangle)
        
    X1c_untwisted = calc_X1c_untwisted(X1s, sinangle, X1c, cosangle)
        
    Y1s_untwisted = calc_X1s_untwisted(Y1s, cosangle, Y1c, sinangle)
        
    Y1c_untwisted = calc_Y1c_untwisted(Y1s, sinangle, Y1c, cosangle)

    # Use (R,Z) for elongation in the (R,Z) plane,
    # or use (X,Y) for elongation in the plane perpendicular to the magnetic axis.
    
    p = calc_p(X1s, X1c, Y1s, Y1c)
    q = calc_q(X1s, Y1c, X1c, Y1s)

    elongation = (p + jnp.sqrt(p * p - 4 * q * q)) / (2 * jnp.abs(q))
    mean_elongation = jnp.sum(elongation * d_l_d_phi) / jnp.sum(d_l_d_phi)
    
    
    max_elongation = -jax_fourier_minimum(-elongation).x

    d_X1c_d_varphi = jnp.matmul(d_d_varphi, X1c)
    d_X1s_d_varphi = jnp.matmul(d_d_varphi, X1s)
    d_Y1s_d_varphi = jnp.matmul(d_d_varphi, Y1s)
    d_Y1c_d_varphi = jnp.matmul(d_d_varphi, Y1c)
    
    grad_b_tensor_results = calculate_grad_B_tensor(spsi, B0, d_l_d_varphi, sG, curvature, X1c, d_Y1s_d_varphi, iotaN, Y1c, d_X1c_d_varphi, Y1s, torsion, d_Y1c_d_varphi, d_d_varphi, tangent_cylindrical, normal_cylindrical, binormal_cylindrical )

    r1_results = R1_Results(Y1s, Y1c, X1s_untwisted, X1c_untwisted, Y1s_untwisted, Y1c_untwisted, elongation, mean_elongation, max_elongation, d_X1c_d_varphi, d_X1s_d_varphi, d_Y1s_d_varphi, d_Y1c_d_varphi)
    return Complete_R1_Results(r1_results, grad_b_tensor_results)
