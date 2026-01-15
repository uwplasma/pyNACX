"""
This module contains the routine for computing the terms in
Mercier's criterion.
"""

import numpy as np
from .util import mu0
import jax.numpy as jnp
from qsc.types import Mercier_Results

def mercier(d_l_d_phi, B0, G0, p2, etabar, curvature, sigma, iotaN, iota, d_phi, nfp,  axis_length, B20_mean, G2, I2) -> Mercier_Results:
    integrand = d_l_d_phi * (etabar*etabar*etabar*etabar + curvature*curvature*curvature*curvature*sigma*sigma + etabar*etabar*curvature*curvature) \
        / (etabar*etabar*etabar*etabar + curvature*curvature*curvature*curvature*(1+sigma*sigma) + 2*etabar*etabar*curvature*curvature)

    integral = jnp.sum(integrand) * d_phi * nfp * 2 * jnp.pi / axis_length

    DGeod_times_r2 = -(2 * mu0 * mu0 * p2 * p2 * G0 * G0 * G0 * G0 * etabar * etabar \
                       / (jnp.pi * jnp.pi * jnp.pi * B0 * B0 * B0 * B0 * B0 * B0 * B0 * B0 * B0 * B0 * iotaN * iotaN)) \
                       * integral
                       
    d2_volume_d_psi2 = 4*jnp.pi*jnp.pi*abs(G0)/(B0*B0*B0)*(3*etabar*etabar - 4*B20_mean/B0 + 2 * (G2 + iota * I2)/G0)
    
    DWell_times_r2 = (mu0 * p2 * abs(G0) / (8 * jnp.pi * jnp.pi * jnp.pi * jnp.pi * B0 * B0 * B0)) * \
        (d2_volume_d_psi2 - 8 * jnp.pi * jnp.pi * mu0 * p2 * abs(G0) / (B0 * B0 * B0 * B0 * B0))
        
    DMerc_times_r2 = DWell_times_r2 + DGeod_times_r2
    
    return Mercier_Results(
        DGeod_times_r2,
        d2_volume_d_psi2,
        DWell_times_r2,
        DMerc_times_r2   
    )