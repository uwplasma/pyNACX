"""
This module contains calculations for O(r**3) terms.
"""

import logging
import numpy as np
from scipy import integrate as integ
from .util import mu0
import jax.numpy as jnp
from qsc.types import Complete_R3_Results


#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calc_r3(B0, G0, X20, Y1c, X2c, X2s, B1c, X1c, X1s, Y1s, I2, iotaN, B20, Y20, Y2c, Y2s, Z20, abs_G0_over_B0, Z2c, Z2s, torsion, d_X1c_d_varphi, d_Y1c_d_varphi, d_d_varphi, spsi, p2, curvature, d_Z20_d_varphi, sG, G2, N_helicity, helicity, nfp, varphi) -> Complete_R3_Results: 
    flux_constraint_coefficient = (-4*B0**2*G0*X20**2*Y1c**2 + 8*B0**2*G0*X20*X2c*Y1c**2 - 4*B0**2*G0*X2c**2*Y1c**2 - \
        4*B0**2*G0*X2s**2*Y1c**2 + 8*B0*G0*B1c*X1c*X2s*Y1c*Y1s + 16*B0**2*G0*X20*X2s*Y1c*Y1s + \
        2*B0**2*I2*iotaN*X1c**2*Y1s**2 - G0*B1c**2*X1c**2*Y1s**2 - 4*B0*G0*B20*X1c**2*Y1s**2 - \
        8*B0*G0*B1c*X1c*X20*Y1s**2 - 4*B0**2*G0*X20**2*Y1s**2 - 8*B0*G0*B1c*X1c*X2c*Y1s**2 - \
        8*B0**2*G0*X20*X2c*Y1s**2 - 4*B0**2*G0*X2c**2*Y1s**2 - 4*B0**2*G0*X2s**2*Y1s**2 + \
        8*B0**2*G0*X1c*X20*Y1c*Y20 - 8*B0**2*G0*X1c*X2c*Y1c*Y20 - 8*B0**2*G0*X1c*X2s*Y1s*Y20 - \
        4*B0**2*G0*X1c**2*Y20**2 - 8*B0**2*G0*X1c*X20*Y1c*Y2c + 8*B0**2*G0*X1c*X2c*Y1c*Y2c + \
        24*B0**2*G0*X1c*X2s*Y1s*Y2c + 8*B0**2*G0*X1c**2*Y20*Y2c - 4*B0**2*G0*X1c**2*Y2c**2 + \
        8*B0**2*G0*X1c*X2s*Y1c*Y2s - 8*B0*G0*B1c*X1c**2*Y1s*Y2s - 8*B0**2*G0*X1c*X20*Y1s*Y2s - \
        24*B0**2*G0*X1c*X2c*Y1s*Y2s - 4*B0**2*G0*X1c**2*Y2s**2 - 4*B0**2*G0*X1c**2*Z20**2 - \
        4*B0**2*G0*Y1c**2*Z20**2 - 4*B0**2*G0*Y1s**2*Z20**2 - 4*B0**2*abs_G0_over_B0*I2*Y1c*Y1s*Z2c + \
        8*B0**2*G0*X1c**2*Z20*Z2c + 8*B0**2*G0*Y1c**2*Z20*Z2c - 8*B0**2*G0*Y1s**2*Z20*Z2c - \
        4*B0**2*G0*X1c**2*Z2c**2 - 4*B0**2*G0*Y1c**2*Z2c**2 - 4*B0**2*G0*Y1s**2*Z2c**2 + \
        2*B0**2*abs_G0_over_B0*I2*X1c**2*Z2s + 2*B0**2*abs_G0_over_B0*I2*Y1c**2*Z2s - 2*B0**2*abs_G0_over_B0*I2*Y1s**2*Z2s + \
        16*B0**2*G0*Y1c*Y1s*Z20*Z2s - 4*B0**2*G0*X1c**2*Z2s**2 - 4*B0**2*G0*Y1c**2*Z2s**2 - \
        4*B0**2*G0*Y1s**2*Z2s**2 + B0**2*abs_G0_over_B0*I2*X1c**3*Y1s*torsion + B0**2*abs_G0_over_B0*I2*X1c*Y1c**2*Y1s*torsion + \
        B0**2*abs_G0_over_B0*I2*X1c*Y1s**3*torsion - B0**2*I2*X1c*Y1c*Y1s*d_X1c_d_varphi + \
        B0**2*I2*X1c**2*Y1s*d_Y1c_d_varphi)/(16*B0**2*G0*X1c**2*Y1s**2)
    
    """
    print(f'B0 {B0}')
    print(f'G0 {G0}')
    print(f'X20 {X20}')
    print(f'Y1c {Y1c}')
    print(f'X2c {X2c}')
    print(f'X2s {X2s}')
    print(f'B1c {B1c}')
    print(f'X1c {X1c}')
    print(f'Y1s {Y1s}')
    print(f'I2 {I2}')
    print(f'iotaN {iotaN}')
    print(f'B20 {B20}')
    print(f'Y20 {Y20}')
    print(f'Y2c {Y2c}')
    print(f'Y2s {Y2s}')
    print(f'Z20 {Z20}')
    print(f'Z2c {Z2c}')
    print(f'abs_G0_over_B0 {abs_G0_over_B0}')
    print(f'Z2s {Z2s}')
    print(f'torsion {torsion}')
    print(f'd_X1c_d_varphi {d_X1c_d_varphi}')
    print(f'd_Y1c_d_varphi {d_Y1c_d_varphi}')
    """


   

    
    
    X3c1 = X1c * flux_constraint_coefficient
    Y3c1 = Y1c * flux_constraint_coefficient
    Y3s1 = Y1s * flux_constraint_coefficient
    X3s1 = X1s * flux_constraint_coefficient
    Z3c1 = 0
    Z3s1 = 0
   
    X3c3 = 0
    X3s3 = 0
    Y3c3 = 0
    Y3s3 = 0
    Z3c3 = 0
    Z3s3 = 0
    
    d_X3c1_d_varphi = d_d_varphi @ X3c1
    d_Y3c1_d_varphi = d_d_varphi @ Y3c1
    d_Y3s1_d_varphi = d_d_varphi @ Y3s1
    
    # The expression below is derived in the O(r**2) paper, and in "20190318-01 Wrick's streamlined Garren-Boozer method, MHD.nb" in the section "Not assuming quasisymmetry".
    # Note Q = (1/2) * (XYEquation0 without X3 and Y3 terms) where XYEquation0 is the quantity in the above notebook.
    Q = -spsi * B0 * abs_G0_over_B0 / (2*G0*G0) * (iotaN * I2 + mu0 * p2 * G0 / (B0 * B0)) + 2 * (X2c * Y2s - X2s * Y2c) \
            + spsi * B0 / (2*G0) * (abs_G0_over_B0 * X20 * curvature - d_Z20_d_varphi) \
            + I2 / (4 * G0) * (-abs_G0_over_B0 * torsion * (X1c*X1c + Y1s*Y1s + Y1c*Y1c) + Y1c * d_X1c_d_varphi - X1c * d_Y1c_d_varphi)
    predicted_flux_constraint_coefficient = - Q / (2 * sG * spsi)
    
    B0_order_a_squared_to_cancel = -sG * B0 * B0 * (G2 + I2 * N_helicity) * abs_G0_over_B0 / (2*G0*G0) \
        -spsi * spsi * B0 * 2 * (X2c * Y2s - X2s * Y2c) \
        -spsi * B0 * B0 / (2*G0) * (abs_G0_over_B0 * X20 * curvature - d_Z20_d_varphi) \
        -spsi * spsi * B0 * I2 / (4*G0) * (-abs_G0_over_B0 * torsion * (X1c*X1c + Y1c*Y1c + Y1s*Y1s) + Y1c * d_X1c_d_varphi - X1c * d_Y1c_d_varphi)
    
    #if helicity == 0:
    X3c1_untwisted = X3c1
    Y3c1_untwisted = Y3c1
    Y3s1_untwisted = Y3s1
    X3s1_untwisted = X3s1
    X3s3_untwisted = X3s3
    X3c3_untwisted = X3c3
    Y3c3_untwisted = Y3c3
    Y3s3_untwisted = Y3s3
    Z3s1_untwisted = Z3s1
    Z3s3_untwisted = Z3s3
    Z3c1_untwisted = Z3c1
    Z3c3_untwisted = Z3c3
    #else:
    angle = -helicity * nfp * varphi
    sinangle = jnp.sin(angle)
    cosangle = jnp.cos(angle)
    X3s1_untwisted = X3s1 *   cosangle  + X3c1 * sinangle
    X3c1_untwisted = X3s1 * (-sinangle) + X3c1 * cosangle
    Y3s1_untwisted = Y3s1 *   cosangle  + Y3c1 * sinangle
    Y3c1_untwisted = Y3s1 * (-sinangle) + Y3c1 * cosangle
    Z3s1_untwisted = Z3s1 *   cosangle  + Z3c1 * sinangle
    Z3c1_untwisted = Z3s1 * (-sinangle) + Z3c1 * cosangle
    sinangle = jnp.sin(3*angle)
    cosangle = jnp.cos(3*angle)
    X3s3_untwisted = X3s3 *   cosangle  + X3c3 * sinangle
    X3c3_untwisted = X3s3 * (-sinangle) + X3c3 * cosangle
    Y3s3_untwisted = Y3s3 *   cosangle  + Y3c3 * sinangle
    Y3c3_untwisted = Y3s3 * (-sinangle) + Y3c3 * cosangle
    Z3s3_untwisted = Z3s3 *   cosangle  + Z3c3 * sinangle
    Z3c3_untwisted = Z3s3 * (-sinangle) + Z3c3 * cosangle
    
    r3_results = Complete_R3_Results(X3c1, Y3c1, Y3s1, X3s1, Z3c1, Z3s1, X3c3, X3s3, Y3c3, Y3s3, Z3c3, Z3s3, d_X3c1_d_varphi, d_Y3c1_d_varphi, d_Y3s1_d_varphi, flux_constraint_coefficient, B0_order_a_squared_to_cancel, X3c1_untwisted, Y3c1_untwisted, Y3s1_untwisted, X3s1_untwisted, X3s3_untwisted, X3c3_untwisted, Y3c3_untwisted, Y3s3_untwisted, Z3s1_untwisted, Z3s3_untwisted, Z3c1_untwisted, Z3c3_untwisted)
    
    return r3_results

    

    
def calculate_shear(self,B31c = 0):
    """
    Compute the magnetic shear iota_2 (so iota=iota0+r^2*iota2) which comes
    from the solvability condition of the generalised sigma equation at order
    O(r**3), as detailed in Rodriguez et al., to be published (2021). 
    This calculation is taken for a standard MHS equilibrium configuration. 
    B31c can be given as an input. 
    One may generalise this calculation straightforwardly.
    """
    logger.debug('Calculating magnetic shear...')
 
    # Shorthand introduced: we also have to ransform to 1/B**2 expansion parameters, taking into account the 
    # difference in the definition of the radial coordinate. In the work of Rodriguez et al.,
    # Phys. Plasmas, (2021), epsilon=sqrt(psi) while in the work of Landreman et al.,
    # J. Plasma Physics (2019) it is defined r=\sqrt(2*psi/B0). Need to transform between the
    # two.

    eps_scale = jnp.sqrt(2/self.B0) 

    # sign_psi = self.spsi
    # sign_G   = self.sG  # Sign is taken to be positive for simplicity. To include this, need to track expressions
    d_d_varphi = self.d_d_varphi
    G2 = self.G2*eps_scale**2
    G0 = self.G0
    I2 = self.I2*eps_scale**2
    X1c = self.X1c*eps_scale
    Y1c = self.Y1c*eps_scale
    Y1s = self.Y1s*eps_scale
    X20 = self.X20*eps_scale**2
    X2s = self.X2s*eps_scale**2
    X2c = self.X2c*eps_scale**2
    Y20 = self.Y20*eps_scale**2
    Y2s = self.Y2s*eps_scale**2
    Y2c = self.Y2c*eps_scale**2
    Z20 = self.Z20*eps_scale**2
    Z2s = self.Z2s*eps_scale**2
    Z2c = self.Z2c*eps_scale**2
    torsion = -self.torsion # I use opposite sign for the torsion
    curvature = self.curvature
    iota = self.iotaN
    dldp = self.abs_G0_over_B0
    dXc1v = self.d_X1c_d_varphi*eps_scale
    dY1cdp = self.d_Y1c_d_varphi*eps_scale
    dY1sdp = self.d_Y1s_d_varphi*eps_scale
    dZ20dp = self.d_Z20_d_varphi*eps_scale**2
    dZ2cdp = self.d_Z2c_d_varphi*eps_scale**2
    dZ2sdp = self.d_Z2s_d_varphi*eps_scale**2
    dX20dp = self.d_X20_d_varphi*eps_scale**2
    dX2cdp = self.d_X2c_d_varphi*eps_scale**2
    dX2sdp = self.d_X2s_d_varphi*eps_scale**2
    dY20dp = self.d_Y20_d_varphi*eps_scale**2
    dY2cdp = self.d_Y2c_d_varphi*eps_scale**2
    dY2sdp = self.d_Y2s_d_varphi*eps_scale**2
    # Transformation to 1/B**2 parameters 
    B0 = 1/self.B0**2
    Ba0 = G0
    Ba1 = G2 + self.iotaN*I2
    eta = self.etabar*np.sqrt(2)*B0**0.25
    B1c = -2*B0*eta
    B20 = (0.75*self.etabar**2/jnp.sqrt(B0) - self.B20)*4*B0**2
    B31s = 0 # To preserve stellarator symmetry
    I4 = 0 # Take current variations at this order to be 0
            
    # Compute Z31c and Z31s from Cp2: we assume standard equilibria, meaning that we may
    # pick Bpsi0=0 and Bpsi1=0
    Z31c = -1/3/Ba0/X1c/Y1s*(2*iota*(X1c*X2s - Y2c*Y1s + Y1c*Y2s) - 2*Ba0*X2s*Y1c*Z20 +
        2*Ba0* X2c*Y1s*Z20 + 2*Ba0*X1c*Y2s*Z20 - 4*Ba0*X2s*Y1c*Z2c - 2*Ba0* X20*Y1s*Z2c +
        4*Ba0*X1c*Y2s*Z2c - dldp*(torsion*(2*X20*Y1c + X2c*Y1c - 2*X1c*Y20 - X1c*Y2c +
        X2s*Y1s) + I2*(2*X20*Y1c + X2c*Y1c - 2*X1c*Y20 - X1c*Y2c + X2s*Y1s) - 
        2*curvature*X1c*Z20 - curvature*X1c*Z2c) + 2*Ba0*X20*Y1c*Z2s + 4*Ba0*X2c*Y1c*Z2s - 
        2*Ba0*X1c*Y20*Z2s - 4*Ba0*X1c*Y2c*Z2s + 2*X1c*dX20dp + X1c*dX2cdp+2*Y1c*dY20dp +
        Y1c*dY2cdp + Y1s*dY2sdp)
         
    dZ31cdp = jnp.matmul(d_d_varphi, Z31c)
            
    Z31s = 1/3/Ba0/X1c/Y1s*(2*iota*(X1c*X2c + Y1c*Y2c + Y1s*Y2s) - 2*Ba0*X2c*Y1c*Z20 + 
        2*Ba0*X1c*Y2c*Z20 - 2*Ba0*X2s*Y1s*Z20 + 2*Ba0*X20*Y1c*Z2c - 2*Ba0*X1c*Y20*Z2c +
        4*Ba0*X2s*Y1s*Z2c + 2*Ba0*X20*Y1s*Z2s - 4*Ba0*X2c*Y1s*Z2s + dldp*(I2*X2s*Y1c + 
        2*I2*X20*Y1s - I2*X2c*Y1s - I2*X1c*Y2s + torsion*(X2s*Y1c + 2*X20*Y1s - X2c*Y1s -
        X1c*Y2s) - curvature*X1c*Z2s) - X1c*dX2sdp - 2*Y1s*dY20dp + Y1s*dY2cdp - Y1c*dY2sdp)
            
    dZ31sdp = jnp.matmul(d_d_varphi, Z31s)

            
    # Equation J3: expression for X31c/s
    X31c = 1/2/dldp**2/curvature*(-2*Ba0*Ba1*B1c - Ba0**2*B31c+2*dldp**2*torsion**2*X1c*X20 +
        2*iota**2*X1c*X2c + dldp**2*torsion**2*X1c*X2c + dldp**2*curvature**2*X1c*(2*X20 + X2c) + 
        3*dldp*iota*torsion*X2s*Y1c + 2*dldp**2*torsion**2*Y1c*Y20 + 2*iota**2*Y1c*Y2c +
        dldp**2*torsion**2*Y1c*Y2c - 2*dldp*iota*torsion*X20*Y1s - 3*dldp*iota*torsion*X2c*Y1s -
        3*dldp*iota*torsion*X1c*Y2s + 2*iota**2*Y1s*Y2s + dldp**2*torsion**2*Y1s*Y2s + 
        2*dldp*iota*Z31s + 2*iota*X2s*dXc1v + 2*dldp*torsion*Y20*dXc1v + dldp*torsion*Y2c*dXc1v + 
        2*dldp*torsion*Y1c*dX20dp + 2*dXc1v*dX20dp + dldp*torsion*Y1c*dX2cdp + dXc1v*dX2cdp - 
        iota*X1c*dX2sdp + dldp*torsion*Y1s*dX2sdp - 2*dldp*torsion*X20*dY1cdp - dldp*torsion*X2c*dY1cdp +
        2*iota*Y2s*dY1cdp - 2*dldp*torsion*X1c*dY20dp + 2*iota*Y1s*dY20dp + 2*dY1cdp*dY20dp - 
        dldp*torsion*X1c*dY2cdp + iota*Y1s*dY2cdp + dY1cdp*dY2cdp - dldp*torsion*X2s*dY1sdp - 
        2*iota*Y2c*dY1sdp - iota*Y1c*dY2sdp + dY1sdp*dY2sdp + dldp*curvature*(-3*iota*X1c*Z2s + 
        dldp*torsion*(Y1c*(2*Z20 + Z2c) + Y1s*Z2s) + 2*Z20*dXc1v + Z2c*dXc1v - 2*X1c*dZ20dp - 
        X1c*dZ2cdp) + 2*dldp*dZ31cdp)
                
    X31s = 1/2/dldp**2/curvature*(-Ba0**2*B31s + dldp**2*curvature**2*X1c*X2s + dldp**2*torsion**2*X1c*X2s +
        2*dldp**2*torsion**2*Y20*Y1s - dldp**2*torsion**2*Y2c*Y1s + dldp**2*torsion**2*Y1c*Y2s +
        2*iota**2*(X1c*X2s - Y2c*Y1s + Y1c*Y2s) + 2*dldp**2*curvature*torsion*Y1s*Z20 - 
        dldp**2*curvature*torsion*Y1s*Z2c + dldp**2*curvature*torsion*Y1c*Z2s + dldp*torsion*Y2s*dXc1v +
        dldp*curvature*Z2s*dXc1v + 2*dldp*torsion*Y1s*dX20dp - dldp*torsion*Y1s*dX2cdp + 
        dldp*torsion*Y1c*dX2sdp + dXc1v*dX2sdp - dldp*torsion*X2s*dY1cdp - 2*dldp*torsion*X20*dY1sdp + 
        dldp*torsion*X2c*dY1sdp + 2*dY20dp*dY1sdp - dY2cdp*dY1sdp - dldp*torsion*X1c*dY2sdp + dY1cdp*dY2sdp +
        iota*(dldp*torsion*(2*X20*Y1c - 3*X2c*Y1c - 2*X1c*Y20 + 3*X1c*Y2c - 3*X2s*Y1s) + dldp*curvature*X1c*
        (-2*Z20 + 3*Z2c) - 2*dldp*Z31c - 2*X2c*dXc1v - 2*X1c*dX20dp + X1c*dX2cdp - 2*Y2c*dY1cdp -
        2*Y1c*dY20dp + Y1c*dY2cdp - 2*Y2s*dY1sdp + Y1s*dY2sdp) - dldp*curvature*X1c*dZ2sdp +2*dldp*dZ31sdp)

    dX31sdp = np.matmul(d_d_varphi, X31s)
                        
    # Equation Cb2
    Y31s = 1/4/Ba0/X1c*(-2*Ba1*X1c*Y1s + 2*iota*I2*X1c*Y1s - dldp*(4*curvature*X20 + torsion*I2*
        (X1c**2 + Y1c**2 + Y1s**2)) + 4*Ba0*(X31s*Y1c + 2*X2s*Y2c - X31c*Y1s - 2*X2c*Y2s) -
        I2*Y1c*dXc1v + I2*X1c*dY1cdp + 4*dZ20dp) 

    dY31sdp = np.matmul(d_d_varphi, Y31s)

    
    # From the equation for Bt to order n=4, and looking at m=0
    LamTilde = 2/Y1s**2*(Ba0*B0*I4 + (Ba1*B0 + Ba0*B20)*I2) + 1/Y1s**2*(-2*iota*(2*X2c**2 + X1c*X31c + 
        2*X2s**2 + 2*Y2c**2 + 2*Y2s**2 + Y1s*Y31s + 2*Z2c**2 + 2*Z2s**2) + 2*dldp*(torsion*(-X31s*Y1c -
        2*X2s*Y2c + X31c*Y1s + 2*X2c*Y2s + X1c*Y31s) + curvature*(-2*X2s*Z2c + 2*X2c*Z2s + X1c*Z31s)) -
        X31s*dXc1v - 2*X2s*dX2cdp + 2*X2c*dX2sdp + X1c*dX31sdp - Y31s*dY1cdp - 2*Y2s*dY2cdp +
        2*Y2c*dY2sdp + Y1c*dY31sdp - 2*Z2s*dZ2cdp + 2*Z2c*dZ2sdp)

    # Need to compute the integration factor necessary for computing the shear
    DMred = d_d_varphi[1:,1:]   # The differentiation matrix has a linearly dependent row, focus on submatrix

    # Distinguish between the stellarator symmetric case and the non-symmetric one at order r^1.
    # Distinction leads to the expSig function being periodic (stell. sym.) or not.
    if self.sigma0 == 0 and jnp.max(jnp.abs(self.rs)) == 0 and jnp.max(jnp.abs(self.zc)) == 0:
        # Case in which sigma is stellarator-symmetric:
        integSig = jnp.linalg.solve(DMred,self.sigma[1:])   # Invert differentiation matrix: as if first entry a zero, need to add it later
        integSig = jnp.insert(integSig,0,0)  # Add the first entry 0
        expSig = jnp.exp(2*iota*integSig)
        # d_phi_d_varphi = 1 + np.matmul(d_d_varphi,self.phi-self.varphi)
        self.iota2 = self.B0/2*sum(expSig*LamTilde*self.d_varphi_d_phi)/sum(expSig*(X1c**2 + Y1c**2 + Y1s**2)/Y1s**2*self.d_varphi_d_phi) 
    else:
        # Case in which sigma is not stellarator-symmetric:
        # d_phi_d_varphi = 1 + np.matmul(d_d_varphi,self.phi-self.varphi)
        avSig = sum(self.sigma*self.d_varphi_d_phi)/len(self.sigma)     # Separate the piece that gives secular part, so all things periodic
        integSigPer = jnp.linalg.solve(DMred,self.sigma[1:]-avSig)   # Invert differentiation matrix: as if first entry a zero, need to add it later
        integSig = integSigPer + avSig*self.varphi[1:]  # Include the secular piece
        integSig = jnp.insert(integSig,0,0)  # Add the first entry 0
        expSig_ext = jnp.append(jnp.exp(2*iota*integSig),jnp.exp(2*iota*(avSig*2*jnp.pi/self.nfp))) # Add endpoint at 2*pi for better integration
        LamTilde_ext = jnp.append(LamTilde,LamTilde[0])
        fac_denom = (X1c**2 + Y1c**2 + Y1s**2) / Y1s**2
        fac_denom_ext = jnp.append(fac_denom, fac_denom[0])
        varphi_ext = jnp.append(self.varphi, 2 * jnp.pi / self.nfp)
        self.iota2 = self.B0 / 2 \
            * integ.trapezoid(expSig_ext * LamTilde_ext, varphi_ext) \
            / integ.trapezoid(expSig_ext * fac_denom_ext, varphi_ext)
    
    # Using cumtrapz without exploiting periodicity
    # expSig = np.exp(2*iota*integ.cumtrapz(self.sigma,self.varphi,initial=0))
