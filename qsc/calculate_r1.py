"""
This module contains the functions for solving the sigma equation
and computing diagnostics of the O(r^1) solution.
"""

import logging
import numpy as np
from .util import fourier_minimum
from .newton import new_new_newton
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
    sigma = jnp.copy(x)
    sigma.at[0].set(self.sigma0)

    iota = x[0]
    r = jnp.matmul(self.d_d_varphi, sigma) \
        + (iota + self.helicity * self.nfp) * \
        (self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma) \
        - 2 * self.etabar_squared_over_curvature_squared * (-self.spsi * self.torsion + self.I2 / self.B0) * self.G0 / self.B0
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

def solve_sigma_equation(nphi, sigma0, helicity, nfp):
    from solve_sigma_helpers import helper_residual
    """
    Solve the sigma equation.
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

def r1_diagnostics(self, rc, zs, rs, zc, nfp, etabar, sigma0, B0,
                 I2, sG, spsi, nphi, B2s, B2c, p2):
    """
    Compute various properties of the O(r^1) solution, once sigma and
    iota are solved for.
    """
    self.Y1s = derive_calc_Y1s(sG, spsi, nphi, nfp, rc, rs, zc, zs, etabar)
    self.Y1c = derive_calc_Y1c(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)
    
    # If helicity is nonzero, then the original X1s/X1c/Y1s/Y1c variables are defined with respect to a "poloidal" angle that
    # is actually helical, with the theta=0 curve wrapping around the magnetic axis as you follow phi around toroidally. Therefore
    # here we convert to an untwisted poloidal angle, such that the theta=0 curve does not wrap around the axis.
   
    helicity = derive_helicity(rc, nfp, zs, rs, zc, nphi, sG, spsi)
    varphi = derive_varphi(nphi, nfp, rc, rs, zc, zs)
    
    angle = calc_angle(helicity, nfp, varphi)
    sinangle = calc_sinangle(angle)
    cosangle = calc_cosangle(angle)
    
    X1s = derive_calc_X1s(nphi)
    X1c = derive_calc_X1c(etabar, nphi, nfp, rc, rs, zc, zs)
    Y1s = derive_calc_Y1s(sG, spsi, nphi, nfp, rc, rs, zc, zs, etabar)
    Y1c = derive_calc_Y1c(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)
    
    self.X1s_untwisted = calc_X1s_untwisted(X1s, cosangle, X1c, sinangle)
        
    self.X1c_untwisted = calc_X1c_untwisted(X1s, sinangle, X1c, cosangle)
        
    self.Y1s_untwisted = calc_X1s_untwisted(Y1s, cosangle, Y1c, sinangle)
        
    self.Y1c_untwisted = calc_Y1c_untwisted(Y1s, sinangle, Y1c, cosangle)

    # Use (R,Z) for elongation in the (R,Z) plane,
    # or use (X,Y) for elongation in the plane perpendicular to the magnetic axis.
    
    p = calc_p(X1s, X1c, Y1s, Y1c)
    q = calc_q(X1s, Y1c, X1c, Y1s)

    self.elongation = derive_elongation(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)
    self.mean_elongation = derive_mean_elongation(sG, spsi, sigma0, etabar, nphi, nfp, rc, rs, zc, zs)
    
    index = np.argmax(self.elongation)
    
    self.max_elongation = derive_max_elongation(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)

    self.d_X1c_d_varphi = derive_d_X1c_d_varphi(etabar, nphi, nfp, rc, rs, zc, zs)
    self.d_X1s_d_varphi = derive_d_X1s_d_varphi(rc, zs, rs, zc, nfp,  nphi)
    self.d_Y1s_d_varphi = derive_d_Y1s_d_varphi(sG, spsi, nphi, nfp, rc, rs, zc, zs, etabar)
    self.d_Y1c_d_varphi = derive_d_Y1c_d_varphi(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)

    self.calculate_grad_B_tensor()

