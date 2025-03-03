"""
This module contains the functions for solving the sigma equation
and computing diagnostics of the O(r^1) solution.
"""

import logging
import numpy as np
from .util import fourier_minimum
from .newton import new_new_newton, newton
import jax.numpy as jnp
from jax import jacobian
from .calculate_r1_helpers import *

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _residual(self, x):
    """
    Residual in the sigma equation, used for Newton's method.  x is
    the state vector, corresponding to sigma on the phi grid,
    except that the first element of x is actually iota.
    """
    sigma = np.copy(x)
    sigma[0] = (self.sigma0) # somthing is not right here

    #sigma[0] = self.sigma0
    iota = x[0]
    r = np.matmul(self.d_d_varphi, sigma) \
        + (iota + self.helicity * self.nfp) * \
        (self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma) \
        - 2 * self.etabar_squared_over_curvature_squared * (-self.spsi * self.torsion + self.I2 / self.B0) * self.G0 / self.B0
    #logger.debug("_residual called with x={}, r={}".format(x, r))
    
    return r

def _new_residual(x, sigma0, d_d_varphi, helicity, nfp, etabar_squared_over_curvature_squared, spsi, torsion, I2, B0, G0):
    sigma = np.copy(x)
    sigma[0] = (sigma0) # somthing is not right here

    #sigma[0] = self.sigma0
    iota = x[0]
    r = np.matmul(d_d_varphi, sigma) \
        + (iota + helicity * nfp) * \
        (etabar_squared_over_curvature_squared * etabar_squared_over_curvature_squared + 1 + sigma * sigma) \
        - 2 * etabar_squared_over_curvature_squared * (-spsi * torsion + I2 / B0) * G0 / B0
    #logger.debug("_residual called with x={}, r={}".format(x, r))
    
    return r
    

def _jacobian(self, x):
    """
    Compute the Jacobian matrix for solving the sigma equation. x is
    the state vector, corresponding to sigma on the phi grid,
    except that the first element of x is actually iota.
    """
    sigma = np.copy(x)
    sigma[0] = self.sigma0
    iota = x[0]

    # d (Riccati equation) / d sigma:
    # For convenience we will fill all the columns now, and re-write the first column in a moment.
    jac = np.copy(self.d_d_varphi)
    for j in range(self.nphi):
        jac[j, j] += (iota + self.helicity * self.nfp) * 2 * sigma[j]

    # d (Riccati equation) / d iota:
    jac[:, 0] = self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma

    #logger.debug("_jacobian called with x={}, jac={}".format(x, jac))
    return jac

def new_solve_sigma_equation(nphi, sigma0, helicity, nfp):
    from .solve_sigma_helpers import helper_residual
    """
    in progress solve sigma to equation that is unreliant on self
    """
    x0 = jnp.full(nphi, sigma0)
    x0.at[0].set(0) # Initial guess for iota
    """
    soln = scipy.optimize.root(self._residual, x0, jac=self._jacobian, method='lm')
    self.iota = soln.x[0]
    self.sigma = np.copy(soln.x)
    self.sigma[0] = self.sigma0
    """
    sigma = new_new_newton(helper_residual, x0) # helper residual is a functon that runs without self but still returns r  
    iota = sigma[0]
    iotaN = calc_iotaN(iota, helicity, nfp)
    sigma[0] = sigma0
    return sigma, iota, iotaN

def solve_sigma_equation( self, nphi, sigma0, helicity, nfp):
    """
    Solve the sigma equation.
    """
    x0 = np.full(nphi, sigma0)
    x0[0] = 0 # Initial guess for iota
    """
    soln = scipy.optimize.root(self._residual, x0, jac=self._jacobian, method='lm')
    self.iota = soln.x[0]
    self.sigma = np.copy(soln.x)
    self.sigma[0] = self.sigma0
    """
    sigma = newton(self._residual, x0, jac= self._jacobian)
    iota = sigma[0]
    iotaN = iota + helicity * nfp
    sigma = sigma.at[0].set(sigma0)
    return sigma, iota, iotaN

def _determine_helicity(self):
    """
    Determine the integer N associated with the type of quasisymmetry
    by counting the number
    of times the normal vector rotates
    poloidally as you follow the axis around toroidally.
    """
    quadrant = np.zeros(self.nphi + 1)
    for j in range(self.nphi):
        if self.normal_cylindrical[j,0] >= 0:
            if self.normal_cylindrical[j,2] >= 0:
                quadrant[j] = 1
            else:
                quadrant[j] = 4
        else:
            if self.normal_cylindrical[j,2] >= 0:
                quadrant[j] = 2
            else:
                quadrant[j] = 3
    quadrant[self.nphi] = quadrant[0]

    counter = 0
    for j in range(self.nphi):
        if quadrant[j] == 4 and quadrant[j+1] == 1:
            counter += 1
        elif quadrant[j] == 1 and quadrant[j+1] == 4:
            counter -= 1
        else:
            counter += quadrant[j+1] - quadrant[j]

    # It is necessary to flip the sign of axis_helicity in order
    # to maintain "iota_N = iota + axis_helicity" under the parity
    # transformations.
    counter *= self.spsi * self.sG
    self.helicity = counter / 4

def r1_diagnostics(nfp, etabar, sG, spsi, curvature, sigma, helicity, varphi, X1s, X1c, d_l_d_phi, d_d_varphi):
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
    
    index = np.argmax(elongation)
    
    max_elongation = -fourier_minimum(-elongation)

    d_X1c_d_varphi = jnp.matmul(d_d_varphi, X1c)
    d_X1s_d_varphi = jnp.matmul(d_d_varphi, X1s)
    d_Y1s_d_varphi = jnp.matmul(d_d_varphi, Y1s)
    d_Y1c_d_varphi = jnp.matmul(d_d_varphi, Y1c)

    return Y1s, Y1c, X1s_untwisted, X1c_untwisted, Y1s_untwisted, Y1c_untwisted, elongation, mean_elongation, max_elongation, d_X1c_d_varphi, d_X1s_d_varphi, d_Y1s_d_varphi, d_Y1c_d_varphi
