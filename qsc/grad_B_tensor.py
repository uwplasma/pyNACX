#!/usr/bin/env python3

"""
Functions for computing the grad B tensor and grad grad B tensor.
"""

import logging
import numpy as np
import jax
import jax.numpy as jnp
from .util import Struct, fourier_minimum, jax_fourier_minimum
from qsc.types import *

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_grad_B_tensor(spsi, B0, d_l_d_varphi, sG, curvature, X1c, d_Y1s_d_varphi, iotaN, Y1c, d_X1c_d_varphi, Y1s, torsion, d_Y1c_d_varphi, d_d_varphi, tangent_cylindrical, normal_cylindrical, binormal_cylindrical) -> Grad_B_Tensor_Results:
    """
    Compute the components of the grad B tensor, and the scale
    length L grad B associated with the Frobenius norm of this
    tensor.
    The formula for the grad B tensor is eq (3.12) of
    Landreman (2021): Figures of merit for stellarators near the magnetic axis, JPP

    self should be an instance of Qsc with X1c, Y1s etc populated.
    """

    #s = self # Shorthand
    #tensor = Struct()
    
    factor = spsi * B0 / d_l_d_varphi
    tn = sG * B0 * curvature
    nt = tn
    bb = factor * (X1c * d_Y1s_d_varphi - iotaN * X1c * Y1c)
    nn = factor * (d_X1c_d_varphi * Y1s + iotaN * X1c * Y1c) # calling d_X1c_d_varphi
    bn = factor * (-sG * spsi * d_l_d_varphi * torsion \
                          - iotaN * X1c * X1c)
    nb = factor * (d_Y1c_d_varphi * Y1s - d_Y1s_d_varphi * Y1c \
                          + sG * spsi * d_l_d_varphi * torsion \
                          + iotaN * (Y1s * Y1s + Y1c * Y1c))

    cond = jnp.ndim(B0) > 1
    
    B0 = jnp.broadcast_to(B0, (61,))
    
    tt = jax.lax.cond(cond,  
                        lambda _: sG * jnp.matmul(d_d_varphi, B0) / d_l_d_varphi, 
                        lambda _: B0,
                        None)
    
    grad_B_tensor = (tn, nt, bb, nn, bn, nb, tt)
    
    t = tangent_cylindrical.transpose()
    n = normal_cylindrical.transpose()
    b = binormal_cylindrical.transpose()
    
    grad_B_tensor_cylindrical = jnp.array([[
                              nn * n[i] * n[j] \
                            + bn * b[i] * n[j] + nb * n[i] * b[j] \
                            + bb * b[i] * b[j] \
                            + tn * t[i] * n[j] + nt * n[i] * t[j] \
                            + tt * t[i] * t[j]
                        for i in range(3)] for j in range(3)])

    grad_B_colon_grad_B = tn * tn + nt * nt \
        + bb * bb + nn * nn \
        + nb * nb + bn * bn \
        + tt * tt

    L_grad_B = B0 * jnp.sqrt(2 / grad_B_colon_grad_B)
    inv_L_grad_B = 1.0 / L_grad_B
    min_L_grad_B = jax_fourier_minimum(L_grad_B).x
    
    return Grad_B_Tensor_Results(
        grad_B_tensor,
        grad_B_tensor_cylindrical,
        grad_B_colon_grad_B,
        L_grad_B,
        inv_L_grad_B,
        min_L_grad_B
     )
    
    
    
def calculate_grad_grad_B_tensor(X1c, Y1s, Y1c, X20, X2s, X2c, Y20, Y2s, Y2c, Z20, Z2s, Z2c, iotaN, iota, curvature, torsion, G0, B0, sG, spsi, I2, G2, p2, B20, B2s, B2c, d_X1c_d_varphi, d_Y1s_d_varphi, d_Y1c_d_varphi, d_X20_d_varphi, d_X2s_d_varphi, d_X2c_d_varphi, d_Y20_d_varphi, d_Y2s_d_varphi, d_Y2c_d_varphi, d_Z20_d_varphi, d_Z2s_d_varphi, d_Z2c_d_varphi, d2_X1c_d_varphi2, d2_Y1s_d_varphi2, d2_Y1c_d_varphi2, d_curvature_d_varphi, d_torsion_d_varphi, nphi) -> Grad_Grad_B_Results:
    """
    Compute the components of the grad grad B tensor, and the scale
    length L grad grad B associated with the Frobenius norm of this
    tensor.
    self should be an instance of Qsc with X1c, Y1s etc populated.
    The grad grad B tensor in discussed around eq (3.13)
    Landreman (2021): Figures of merit for stellarators near the magnetic axis, JPP
    although an explicit formula is not given there.

    If ``two_ways`` is ``True``, an independent calculation of
    the tensor is also computed, to confirm the answer is the same.
    """
    
    iota_N0 = iotaN
    lp = jnp.abs(G0) / B0

    grad_grad_B = jnp.zeros((nphi, 3, 3, 3))
    #grad_grad_B_alt = jnp.zeros((s.nphi, 3, 3, 3))

    # The elements that follow are computed in the Mathematica notebook "20200407-01 Grad grad B tensor near axis"
    # and then formatted for fortran by the python script process_grad_grad_B_tensor_code

    # The order is (normal, binormal, tangent). So element 123 means nbt.

    # Element 111
    grad_grad_B = grad_grad_B.at[:,0,0,0].set((B0*B0*B0*B0*lp*lp*(8*iota_N0*X2c*Y1c*\
                                              Y1s + 4*iota_N0*X2s*\
                                              (-Y1c*Y1c + Y1s*Y1s) + \
                                              2*iota_N0*X1c*Y1s*Y20 + \
                                              2*iota_N0*X1c*Y1s*Y2c - \
                                              2*iota_N0*X1c*Y1c*Y2s + \
                                              5*iota_N0*X1c*X1c*Y1c*Y1s*\
                                              curvature - \
                                              2*Y1c*Y20*d_X1c_d_varphi + \
                                              2*Y1c*Y2c*d_X1c_d_varphi + \
                                              2*Y1s*Y2s*d_X1c_d_varphi + \
                                              5*X1c*Y1s*Y1s*curvature*\
                                              d_X1c_d_varphi + \
                                              2*Y1c*Y1c*d_X20_d_varphi + \
                                              2*Y1s*Y1s*d_X20_d_varphi - \
                                              2*Y1c*Y1c*d_X2c_d_varphi + \
                                              2*Y1s*Y1s*d_X2c_d_varphi - \
                                              4*Y1c*Y1s*d_X2s_d_varphi))/\
                                              (G0*G0*G0))

    # Element 112
    grad_grad_B = grad_grad_B.at[:,0,0,1].set((B0*B0*B0*B0*lp*lp*(Y1c*Y1c*\
                                              (-6*iota_N0*Y2s + \
                                               5*iota_N0*X1c*Y1s*\
                                               curvature + \
                                               2*(lp*X20*torsion - \
                                                  lp*X2c*torsion + \
                                                  d_Y20_d_varphi - \
                                                  d_Y2c_d_varphi)) + \
                                              Y1s*(5*iota_N0*X1c*Y1s*Y1s*\
                                                   curvature + \
                                                   2*(lp*X1c*Y2s*torsion + \
                                                      Y2s*d_Y1c_d_varphi - \
                                                      (Y20 + Y2c)*\
                                                      d_Y1s_d_varphi) + \
                                                   Y1s*(6*iota_N0*Y2s + \
                                                        2*lp*X20*torsion + \
                                                        2*lp*X2c*torsion + \
                                                        5*lp*X1c*X1c*curvature*\
                                                        torsion + \
                                                        5*X1c*curvature*\
                                                        d_Y1c_d_varphi + \
                                                        2*d_Y20_d_varphi + \
                                                        2*d_Y2c_d_varphi)) + \
                                              Y1c*(2*(lp*X1c*\
                                                      (-Y20 + Y2c)*torsion - \
                                                      Y20*d_Y1c_d_varphi + \
                                                      Y2c*d_Y1c_d_varphi + \
                                                      Y2s*d_Y1s_d_varphi) + \
                                                   Y1s*(12*iota_N0*Y2c - \
                                                        4*lp*X2s*torsion - \
                                                        5*X1c*curvature*\
                                                        d_Y1s_d_varphi - \
                                                        4*d_Y2s_d_varphi))))/(G0*G0*G0))

    # Element 113
    grad_grad_B = grad_grad_B.at[:,0,0,2].set(-((B0*B0*B0*lp*lp*(2*Y1c*Y1c*\
                                             (2*B2c*G0*lp + B0*G2*lp + B0*I2*lp*iota - \
                                              2*G0*lp*B20 + 2*B0*G0*iota_N0*Z2s + \
                                              B0*G0*lp*X20*curvature - \
                                              B0*G0*lp*X2c*curvature - \
                                              B0*G0*d_Z20_d_varphi + \
                                              B0*G0*d_Z2c_d_varphi) + \
                                             Y1s*(-2*B0*G0*lp*X1c*Y2s*\
                                                  curvature + \
                                                  Y1s*(-4*B2c*G0*lp + 2*B0*G2*lp + \
                                                       2*B0*I2*lp*iota - 4*G0*lp*B20 - \
                                                       4*B0*G0*iota_N0*Z2s + \
                                                       2*B0*G0*lp*X20*curvature + \
                                                       2*B0*G0*lp*X2c*curvature + \
                                                       B0*G0*lp*X1c*X1c*curvature*curvature - \
                                                       2*B0*G0*d_Z20_d_varphi - \
                                                       2*B0*G0*d_Z2c_d_varphi)) + \
                                             2*G0*Y1c*(B0*lp*X1c*\
                                                       (Y20 - Y2c)*curvature + \
                                                       2*Y1s*(2*B2s*lp - 2*B0*iota_N0*Z2c - \
                                                              B0*lp*X2s*curvature + \
                                                              B0*d_Z2s_d_varphi))))/(G0*G0*G0*G0)))

    # Element 121
    grad_grad_B = grad_grad_B.at[:,0,1,0].set(-((B0*B0*B0*B0*lp*lp*(3*iota_N0*X1c*X1c*X1c*Y1s*\
                                                curvature + \
                                                3*lp*X1c*X1c*Y1s*Y1s*curvature*\
                                                torsion + \
                                                2*(X2s*Y1s*\
                                                   (-2*lp*Y1c*torsion + \
                                                    d_X1c_d_varphi) + \
                                                   X20*(lp*Y1c*Y1c*torsion + \
                                                        lp*Y1s*Y1s*torsion - \
                                                        Y1c*d_X1c_d_varphi) + \
                                                   X2c*(-(lp*Y1c*Y1c*\
                                                          torsion) + \
                                                        lp*Y1s*Y1s*torsion + \
                                                        Y1c*d_X1c_d_varphi)) - \
                                                2*X1c*(3*iota_N0*X2s*Y1c - \
                                                       iota_N0*X20*Y1s - \
                                                       3*iota_N0*X2c*Y1s + \
                                                       lp*Y1c*Y20*torsion - \
                                                       lp*Y1c*Y2c*torsion - \
                                                       lp*Y1s*Y2s*torsion - \
                                                       Y1c*d_X20_d_varphi + \
                                                       Y1c*d_X2c_d_varphi + \
                                                       Y1s*d_X2s_d_varphi)))/\
                            (G0*G0*G0)))

    # Element 122
    grad_grad_B = grad_grad_B.at[:,0,1,1].set((B0*B0*B0*B0*lp*lp*(-4*iota_N0*X1c*Y1s*\
                                              Y2c + 4*iota_N0*X1c*Y1c*\
                                              Y2s - 3*iota_N0*X1c*X1c*Y1c*\
                                              Y1s*curvature + \
                                              2*X20*Y1c*d_Y1c_d_varphi + \
                                              2*X20*Y1s*d_Y1s_d_varphi + \
                                              3*X1c*X1c*Y1s*curvature*\
                                              d_Y1s_d_varphi + \
                                              2*X2s*(iota_N0*Y1c*Y1c - \
                                                     Y1s*(iota_N0*Y1s + \
                                                          d_Y1c_d_varphi) - \
                                                     Y1c*d_Y1s_d_varphi) - \
                                              2*X2c*(Y1c*\
                                                     (2*iota_N0*Y1s + d_Y1c_d_varphi) \
                                                     - Y1s*d_Y1s_d_varphi) - \
                                              2*X1c*Y1c*d_Y20_d_varphi + \
                                              2*X1c*Y1c*d_Y2c_d_varphi + \
                                              2*X1c*Y1s*d_Y2s_d_varphi))/\
                                              (G0*G0*G0))
    #       (2*iota_N0*Y1s + d_Y1c_d_varphi) \\

    # Element 123
    grad_grad_B = grad_grad_B.at[:,0,1,2].set((2*B0*B0*B0*lp*lp*X1c*\
                           (Y1c*(2*B2c*G0*lp + B0*G2*lp + B0*I2*lp*iota - \
                                 2*G0*lp*B20 + 2*B0*G0*iota_N0*Z2s + \
                                 2*B0*G0*lp*X20*curvature - \
                                 2*B0*G0*lp*X2c*curvature - \
                                 B0*G0*d_Z20_d_varphi + \
                                 B0*G0*d_Z2c_d_varphi) + \
                            G0*Y1s*(2*B2s*lp - 2*B0*iota_N0*Z2c - \
                                    2*B0*lp*X2s*curvature + \
                                    B0*d_Z2s_d_varphi)))/(G0*G0*G0*G0))

    # Element 131
    grad_grad_B = grad_grad_B.at[:,0,2,0].set((B0*B0*B0*B0*lp*(-4*lp*lp*X2s*Y1c*Y1s*\
                                           curvature + \
                                           2*lp*lp*X2c*(-Y1c*Y1c + Y1s*Y1s)*\
                                           curvature + \
                                           2*lp*lp*X20*(Y1c*Y1c + Y1s*Y1s)*\
                                           curvature - \
                                           2*lp*lp*X1c*Y1c*Y20*\
                                           curvature + \
                                           2*lp*lp*X1c*Y1c*Y2c*\
                                           curvature + \
                                           2*lp*lp*X1c*Y1s*Y2s*\
                                           curvature + \
                                           3*lp*lp*X1c*X1c*Y1s*Y1s*\
                                           curvature*curvature + \
                                           lp*iota_N0*X1c*X1c*X1c*Y1s*\
                                           torsion - lp*iota_N0*X1c*\
                                           Y1c*Y1c*Y1s*torsion - \
                                           lp*iota_N0*X1c*Y1s*Y1s*Y1s*\
                                           torsion - Y1s*Y1s*\
                                           d_X1c_d_varphi*d_X1c_d_varphi + \
                                           iota_N0*X1c*X1c*Y1s*\
                                           d_Y1c_d_varphi - \
                                           lp*X1c*Y1s*Y1s*torsion*\
                                           d_Y1c_d_varphi - \
                                           iota_N0*X1c*X1c*Y1c*\
                                           d_Y1s_d_varphi + \
                                           lp*X1c*Y1c*Y1s*\
                                           torsion*d_Y1s_d_varphi + \
                                           X1c*Y1s*Y1s*d2_X1c_d_varphi2))/\
                                           (G0*G0*G0))

    # Element 132
    grad_grad_B = grad_grad_B.at[:,0,2,1].set((B0*B0*B0*B0*lp*(-(Y1s*d_X1c_d_varphi*\
                                             (iota_N0*Y1c*Y1c + \
                                              Y1s*(iota_N0*Y1s + \
                                                   d_Y1c_d_varphi) - \
                                              Y1c*d_Y1s_d_varphi)) + \
                                           lp*X1c*X1c*Y1s*\
                                           (2*iota_N0*Y1c*torsion - \
                                            torsion*d_Y1s_d_varphi + \
                                            Y1s*d_torsion_d_varphi) + \
                                           X1c*(Y1c*d_Y1s_d_varphi*\
                                                (-(iota_N0*Y1c) + d_Y1s_d_varphi) \
                                                + Y1s*Y1s*(lp*torsion*\
                                                           d_X1c_d_varphi + \
                                                           iota_N0*d_Y1s_d_varphi + \
                                                           d2_Y1c_d_varphi2) - \
                                                Y1s*(d_Y1c_d_varphi*\
                                                     d_Y1s_d_varphi + \
                                                     Y1c*(-2*iota_N0*d_Y1c_d_varphi + \
                                                          d2_Y1s_d_varphi2)))))/(G0*G0*G0))
    #       (-(iota_N0*Y1c) + d_Y1s_d_varphi) \\

    # Element 133
    grad_grad_B = grad_grad_B.at[:,0,2,2].set((B0*B0*B0*B0*lp*lp*X1c*Y1s*\
                           (-(Y1s*curvature*\
                              d_X1c_d_varphi) + \
                            X1c*(-(iota_N0*Y1c*\
                                   curvature) + \
                                 Y1s*d_curvature_d_varphi)))/\
                                 (G0*G0*G0))

    # Element 211
    grad_grad_B = grad_grad_B.at[:,1,0,0].set((-2*B0*B0*B0*B0*lp*lp*X1c*\
                           (-2*iota_N0*X2s*Y1c + \
                            2*iota_N0*X2c*Y1s - \
                            iota_N0*X1c*Y2s + \
                            iota_N0*X1c*X1c*Y1s*curvature + \
                            lp*X1c*Y1s*Y1s*curvature*\
                            torsion - Y20*\
                            d_X1c_d_varphi + \
                            Y2c*d_X1c_d_varphi + \
                            Y1c*d_X20_d_varphi - \
                            Y1c*d_X2c_d_varphi - \
                            Y1s*d_X2s_d_varphi))/(G0*G0*G0))

    # Element 212
    grad_grad_B = grad_grad_B.at[:,1,0,1].set((2*B0*B0*B0*B0*lp*lp*X1c*\
                           (lp*X1c*Y20*torsion - \
                            lp*X1c*Y2c*torsion + \
                            Y20*d_Y1c_d_varphi - \
                            Y2c*d_Y1c_d_varphi - \
                            Y2s*d_Y1s_d_varphi + \
                            Y1c*(3*iota_N0*Y2s - \
                                 lp*X20*torsion + \
                                 lp*X2c*torsion - \
                                 d_Y20_d_varphi + d_Y2c_d_varphi) \
                            + Y1s*(iota_N0*Y20 - \
                                   3*iota_N0*Y2c - \
                                   iota_N0*X1c*Y1c*curvature + \
                                   lp*X2s*torsion + \
                                   X1c*curvature*\
                                   d_Y1s_d_varphi + d_Y2s_d_varphi))\
                           )/(G0*G0*G0))
    #       d_Y20_d_varphi + d_Y2c_d_varphi) \\
    #       d_Y1s_d_varphi + d_Y2s_d_varphi))\\

    # Element 213
    grad_grad_B = grad_grad_B.at[:,1,0,2].set((2*B0*B0*B0*lp*lp*X1c*\
                           (Y1c*(2*B2c*G0*lp + B0*G2*lp + B0*I2*lp*iota - \
                                 2*G0*lp*B20 + 2*B0*G0*iota_N0*Z2s + \
                                 B0*G0*lp*X20*curvature - \
                                 B0*G0*lp*X2c*curvature - \
                                 B0*G0*d_Z20_d_varphi + \
                                 B0*G0*d_Z2c_d_varphi) + \
                            G0*(B0*lp*X1c*(Y20 - Y2c)*\
                                curvature + \
                                Y1s*(2*B2s*lp - 2*B0*iota_N0*Z2c - \
                                     B0*lp*X2s*curvature + \
                                     B0*d_Z2s_d_varphi))))/(G0*G0*G0*G0))

    # Element 221
    grad_grad_B =  grad_grad_B.at[:,1,1,0].set((-2*B0*B0*B0*B0*lp*lp*X1c*\
                           (lp*X2c*Y1c*torsion + \
                            lp*X2s*Y1s*torsion - \
                            X2c*d_X1c_d_varphi + \
                            X20*(-(lp*Y1c*torsion) + \
                                 d_X1c_d_varphi) + \
                            X1c*(3*iota_N0*X2s + \
                                 lp*Y20*torsion - \
                                 lp*Y2c*torsion - \
                                 d_X20_d_varphi + d_X2c_d_varphi)))/\
                                 (G0*G0*G0))

    # Element 222
    grad_grad_B = grad_grad_B.at[:,1,1,1].set((-2*B0*B0*B0*B0*lp*lp*X1c*\
                           (-(iota_N0*X2c*Y1s) + \
                            2*iota_N0*X1c*Y2s - \
                            X2c*d_Y1c_d_varphi + \
                            X20*(iota_N0*Y1s + \
                                 d_Y1c_d_varphi) + \
                            X2s*(iota_N0*Y1c - \
                                 d_Y1s_d_varphi) - \
                            X1c*d_Y20_d_varphi + \
                            X1c*d_Y2c_d_varphi))/(G0*G0*G0))

    # Element 223
    grad_grad_B = grad_grad_B.at[:,1,1,2].set((-2*B0*B0*B0*lp*lp*X1c*X1c*\
                           (2*B2c*G0*lp + B0*G2*lp + B0*I2*lp*iota - 2*G0*lp*B20 + \
                            2*B0*G0*iota_N0*Z2s + \
                            2*B0*G0*lp*X20*curvature - \
                            2*B0*G0*lp*X2c*curvature - \
                            B0*G0*d_Z20_d_varphi + \
                            B0*G0*d_Z2c_d_varphi))/(G0*G0*G0*G0))

    # Element 231
    grad_grad_B = grad_grad_B.at[:,1,2,0].set((B0*B0*B0*B0*lp*X1c*(-2*lp*lp*X20*Y1c*\
                                               curvature + \
                                               2*lp*lp*X2c*Y1c*curvature + \
                                               2*lp*lp*X2s*Y1s*curvature + \
                                               2*lp*lp*X1c*Y20*curvature - \
                                               2*lp*lp*X1c*Y2c*curvature + \
                                               2*lp*iota_N0*X1c*Y1c*Y1s*\
                                               torsion - iota_N0*X1c*Y1s*\
                                               d_X1c_d_varphi + \
                                               lp*Y1s*Y1s*torsion*\
                                               d_X1c_d_varphi + \
                                               iota_N0*X1c*X1c*d_Y1s_d_varphi - \
                                               lp*X1c*Y1s*torsion*\
                                               d_Y1s_d_varphi - \
                                               lp*X1c*Y1s*Y1s*\
                                               d_torsion_d_varphi))/(G0*G0*G0))

    # Element 232
    grad_grad_B = grad_grad_B.at[:,1,2,1].set((B0*B0*B0*B0*lp*X1c*(-(lp*iota_N0*X1c*X1c*\
                                                 Y1s*torsion) + \
                                               lp*Y1s*torsion*\
                                               (iota_N0*Y1c*Y1c + \
                                                Y1s*(iota_N0*Y1s + \
                                                     d_Y1c_d_varphi) - \
                                                Y1c*d_Y1s_d_varphi) + \
                                               X1c*((iota_N0*Y1c - \
                                                     d_Y1s_d_varphi)*d_Y1s_d_varphi \
                                                    + Y1s*(-(iota_N0*d_Y1c_d_varphi) + \
                                                           d2_Y1s_d_varphi2))))/(G0*G0*G0))
    #       d_Y1s_d_varphi)*d_Y1s_d_varphi \\

    # Element 233
    grad_grad_B = grad_grad_B.at[:,1,2,2].set((B0*B0*B0*B0*lp*lp*X1c*X1c*Y1s*curvature*\
                           (iota_N0*X1c + 2*lp*Y1s*torsion))/\
                           (G0*G0*G0))

    # Element 311
    grad_grad_B = grad_grad_B.at[:,2,0,0].set((B0*B0*B0*B0*lp*X1c*Y1s*\
                           (lp*iota_N0*X1c*X1c*torsion - \
                            lp*iota_N0*Y1c*Y1c*torsion - \
                            lp*iota_N0*Y1s*Y1s*torsion - \
                            lp*Y1s*torsion*\
                            d_Y1c_d_varphi + \
                            X1c*(2*lp*lp*Y1s*curvature*curvature + \
                                 iota_N0*d_Y1c_d_varphi) + \
                            d_X1c_d_varphi*d_Y1s_d_varphi + \
                            Y1c*(iota_N0*d_X1c_d_varphi + \
                                 lp*torsion*d_Y1s_d_varphi) + \
                            Y1s*d2_X1c_d_varphi2))/(G0*G0*G0))

    # Element 312
    grad_grad_B = grad_grad_B.at[:,2,0,1].set((B0*B0*B0*B0*lp*X1c*Y1s*\
                           (lp*X1c*(2*iota_N0*Y1c*\
                                    torsion + \
                                    Y1s*d_torsion_d_varphi) + \
                            Y1s*(2*lp*torsion*\
                                 d_X1c_d_varphi + \
                                 2*iota_N0*d_Y1s_d_varphi + \
                                 d2_Y1c_d_varphi2) + \
                            Y1c*(2*iota_N0*d_Y1c_d_varphi - \
                                 d2_Y1s_d_varphi2)))/(G0*G0*G0))

    # Element 313
    grad_grad_B = grad_grad_B.at[:,2,0,2].set((B0*B0*B0*B0*lp*lp*X1c*X1c*Y1s*\
                           (-(iota_N0*Y1c*curvature) + \
                            curvature*d_Y1s_d_varphi + \
                            Y1s*d_curvature_d_varphi))/\
                            (G0*G0*G0))

    # Element 321
    grad_grad_B = grad_grad_B.at[:,2,1,0].set(-((B0*B0*B0*B0*lp*X1c*X1c*Y1s*\
                             (-2*lp*iota_N0*Y1c*torsion + \
                              2*iota_N0*d_X1c_d_varphi + \
                              2*lp*torsion*d_Y1s_d_varphi + \
                              lp*Y1s*d_torsion_d_varphi))/\
                            (G0*G0*G0)))

    # Element 322
    grad_grad_B = grad_grad_B.at[:,2,1,1].set(-((B0*B0*B0*B0*lp*X1c*Y1s*\
                             (lp*iota_N0*X1c*X1c*torsion - \
                              lp*iota_N0*Y1c*Y1c*torsion - \
                              lp*iota_N0*Y1s*Y1s*torsion - \
                              lp*Y1s*torsion*\
                              d_Y1c_d_varphi - \
                              d_X1c_d_varphi*d_Y1s_d_varphi + \
                              Y1c*(iota_N0*d_X1c_d_varphi + \
                                   lp*torsion*d_Y1s_d_varphi) + \
                              X1c*(iota_N0*d_Y1c_d_varphi - \
                                   d2_Y1s_d_varphi2)))/(G0*G0*G0)))

    # Element 323
    grad_grad_B = grad_grad_B.at[:,2,1,2].set((B0*B0*B0*B0*lp*lp*X1c*X1c*Y1s*curvature*\
                           (iota_N0*X1c + 2*lp*Y1s*torsion))/\
                           (G0*G0*G0))

    # Element 331
    grad_grad_B = grad_grad_B.at[:,2,2,0].set((B0*B0*B0*B0*lp*lp*X1c*X1c*Y1s*\
                           (-(iota_N0*Y1c*curvature) + \
                            curvature*d_Y1s_d_varphi + \
                            Y1s*d_curvature_d_varphi))/\
                            (G0*G0*G0))

    # Element 332
    grad_grad_B = grad_grad_B.at[:,2,2,1].set(-((B0*B0*B0*B0*lp*lp*X1c*Y1s*curvature*\
                             (iota_N0*Y1c*Y1c + \
                              Y1s*(iota_N0*Y1s + \
                                   d_Y1c_d_varphi) - \
                              Y1c*d_Y1s_d_varphi))/(G0*G0*G0)))

    # Element 333
    grad_grad_B = grad_grad_B.at[:,2,2,2].set((-2*B0*B0*B0*B0*lp*lp*lp*X1c*X1c*Y1s*Y1s*\
                           curvature*curvature)/(G0*G0*G0))


    grad_grad_B = grad_grad_B

    # Compute the (inverse) scale length
    squared = grad_grad_B * grad_grad_B
    norm_squared = jnp.sum(squared, axis=(1,2,3))
    grad_grad_B_inverse_scale_length_vs_varphi = jnp.sqrt(jnp.sqrt(norm_squared) / (4*B0))
    L_grad_grad_B = 1 / grad_grad_B_inverse_scale_length_vs_varphi
    grad_grad_B_inverse_scale_length = jnp.max(grad_grad_B_inverse_scale_length_vs_varphi)
    
    return Grad_Grad_B_Results(
      grad_grad_B,
      grad_grad_B_inverse_scale_length_vs_varphi,
      L_grad_grad_B, grad_grad_B_inverse_scale_length
    )
    #two way is not implemeted
    

def Bfield_cylindrical(self, r=0, theta=0):
    '''
    Function to calculate the magnetic field vector B=(B_R,B_phi,B_Z) at
    every point along the axis (hence with nphi points) where R, phi and Z
    are the standard cylindrical coordinates for a given
    near-axis radius r and a Boozer poloidal angle vartheta (not theta).
    The formulae implemented here are eq (3.5) and (3.6) of
    Landreman (2021): Figures of merit for stellarators near the magnetic axis, JPP

    Args:
      r: the near-axis radius
      theta: the Boozer poloidal angle vartheta (= theta-N*phi)
    '''
    
    # Define auxiliary variables
    t = self.tangent_cylindrical.transpose()
    n = self.normal_cylindrical.transpose()
    b = self.binormal_cylindrical.transpose()
    B0 = self.B0
    sG = self.sG
    G0 = self.G0
    X1c = self.X1c
    X1s = self.X1s
    Y1c = self.Y1c
    Y1s = self.Y1s
    d_l_d_varphi = self.d_l_d_varphi
    curvature = self.curvature
    torsion = self.torsion
    iotaN = self.iotaN
    d_X1c_d_varphi = self.d_X1c_d_varphi
    d_X1s_d_varphi = self.d_X1s_d_varphi
    d_Y1s_d_varphi = self.d_Y1s_d_varphi
    d_Y1c_d_varphi = self.d_Y1c_d_varphi

    B0_vector = sG * B0 * t

    if r == 0:
        return B0_vector
    else:
        factor = B0 * B0 / G0
        B1_vector_t = factor * (X1c * jnp.cos(theta) + X1s * jnp.sin(theta)) * d_l_d_varphi * curvature
        B1_vector_n = factor * (jnp.cos(theta) * (d_X1c_d_varphi - Y1c * d_l_d_varphi * torsion + iotaN * X1s) \
                                + jnp.sin(theta) * (d_X1s_d_varphi - Y1s * d_l_d_varphi * torsion - iotaN * X1c))
        B1_vector_b = factor * (jnp.cos(theta) * (d_Y1c_d_varphi + X1c * d_l_d_varphi * torsion + iotaN * Y1s) \
                                + jnp.sin(theta) * (d_Y1s_d_varphi + X1s * d_l_d_varphi * torsion - iotaN * Y1c))

        B1_vector = B1_vector_t * t + B1_vector_n * n + B1_vector_b * b
        B_vector_cylindrical = B0_vector + r * B1_vector

        return B_vector_cylindrical

def Bfield_cartesian(self, r=0, theta=0):
    '''
    Function to calculate the magnetic field vector B=(B_x,B_y,B_z) at
    every point along the axis (hence with nphi points) where x, y and z
    are the standard cartesian coordinates for a given
    near-axis radius r and a Boozer poloidal angle vartheta (not theta).

    Args:
      r: the near-axis radius
      theta: the Boozer poloidal angle vartheta (= theta-N*phi)
    '''
    B_vector_cylindrical = self.Bfield_cylindrical(r,theta)
    phi = self.phi

    B_x = jnp.cos(phi) * B_vector_cylindrical[0] - jnp.sin(phi) * B_vector_cylindrical[1]
    B_y = jnp.sin(phi) * B_vector_cylindrical[0] + jnp.cos(phi) * B_vector_cylindrical[1]
    B_z = B_vector_cylindrical[2]

    B_vector_cartesian = jnp.array([B_x, B_y, B_z])

    return B_vector_cartesian

def grad_B_tensor_cartesian(self):
    '''
    Function to calculate the gradient of the magnetic field vector B=(B_x,B_y,B_z)
    at every point along the axis (hence with nphi points) where x, y and z
    are the standard cartesian coordinates.
    '''

    B0, B1, B2 = self.Bfield_cylindrical()
    nablaB = self.grad_B_tensor_cylindrical
    cosphi = jnp.cos(self.phi)
    sinphi = jnp.sin(self.phi)
    R0 = self.R0

    grad_B_vector_cartesian = jnp.array([
[cosphi**2*nablaB[0, 0] - cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + 
   sinphi**2*nablaB[1, 1], cosphi**2*nablaB[0, 1] - sinphi**2*nablaB[1, 0] + 
   cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), cosphi*nablaB[0, 2] - 
   sinphi*nablaB[1, 2]], [-(sinphi**2*nablaB[0, 1]) + cosphi**2*nablaB[1, 0] + 
   cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), sinphi**2*nablaB[0, 0] + 
   cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + cosphi**2*nablaB[1, 1], 
  sinphi*nablaB[0, 2] + cosphi*nablaB[1, 2]], 
 [cosphi*nablaB[2, 0] - sinphi*nablaB[2, 1], sinphi*nablaB[2, 0] + cosphi*nablaB[2, 1], 
  nablaB[2, 2]]
    ])

    return grad_B_vector_cartesian

def grad_grad_B_tensor_cylindrical(self):
    '''
    Function to calculate the gradient of of the gradient the magnetic field
    vector B=(B_R,B_phi,B_Z) at every point along the axis (hence with nphi points)
    where R, phi and Z are the standard cylindrical coordinates.
    '''
    return jnp.transpose(self.grad_grad_B,(1,2,3,0))

def grad_grad_B_tensor_cartesian(self):
    '''
    Function to calculate the gradient of of the gradient the magnetic field
    vector B=(B_x,B_y,B_z) at every point along the axis (hence with nphi points)
    where x, y and z are the standard cartesian coordinates.
    '''
    nablanablaB = self.grad_grad_B_tensor_cylindrical()
    cosphi = jnp.cos(self.phi)
    sinphi = jnp.sin(self.phi)

    grad_grad_B_vector_cartesian = jnp.array([[
[cosphi**3*nablanablaB[0, 0, 0] - cosphi**2*sinphi*(nablanablaB[0, 0, 1] + 
      nablanablaB[0, 1, 0] + nablanablaB[1, 0, 0]) + 
    cosphi*sinphi**2*(nablanablaB[0, 1, 1] + nablanablaB[1, 0, 1] + 
      nablanablaB[1, 1, 0]) - sinphi**3*nablanablaB[1, 1, 1], 
   cosphi**3*nablanablaB[0, 0, 1] + cosphi**2*sinphi*(nablanablaB[0, 0, 0] - 
      nablanablaB[0, 1, 1] - nablanablaB[1, 0, 1]) + sinphi**3*nablanablaB[1, 1, 0] - 
    cosphi*sinphi**2*(nablanablaB[0, 1, 0] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), cosphi**2*nablanablaB[0, 0, 2] - 
    cosphi*sinphi*(nablanablaB[0, 1, 2] + nablanablaB[1, 0, 2]) + 
    sinphi**2*nablanablaB[1, 1, 2]], [cosphi**3*nablanablaB[0, 1, 0] + 
    sinphi**3*nablanablaB[1, 0, 1] + cosphi**2*sinphi*(nablanablaB[0, 0, 0] - 
      nablanablaB[0, 1, 1] - nablanablaB[1, 1, 0]) - 
    cosphi*sinphi**2*(nablanablaB[0, 0, 1] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), cosphi**3*nablanablaB[0, 1, 1] - 
    sinphi**3*nablanablaB[1, 0, 0] + cosphi*sinphi**2*(nablanablaB[0, 0, 0] - 
      nablanablaB[1, 0, 1] - nablanablaB[1, 1, 0]) + 
    cosphi**2*sinphi*(nablanablaB[0, 0, 1] + nablanablaB[0, 1, 0] - 
      nablanablaB[1, 1, 1]), cosphi**2*nablanablaB[0, 1, 2] - 
    sinphi**2*nablanablaB[1, 0, 2] + cosphi*sinphi*(nablanablaB[0, 0, 2] - 
      nablanablaB[1, 1, 2])], [cosphi**2*nablanablaB[0, 2, 0] - 
    cosphi*sinphi*(nablanablaB[0, 2, 1] + nablanablaB[1, 2, 0]) + 
    sinphi**2*nablanablaB[1, 2, 1], cosphi**2*nablanablaB[0, 2, 1] - 
    sinphi**2*nablanablaB[1, 2, 0] + cosphi*sinphi*(nablanablaB[0, 2, 0] - 
      nablanablaB[1, 2, 1]), cosphi*nablanablaB[0, 2, 2] - 
    sinphi*nablanablaB[1, 2, 2]]], 
 [[sinphi**3*nablanablaB[0, 1, 1] + cosphi**3*nablanablaB[1, 0, 0] + 
    cosphi**2*sinphi*(nablanablaB[0, 0, 0] - nablanablaB[1, 0, 1] - 
      nablanablaB[1, 1, 0]) - cosphi*sinphi**2*(nablanablaB[0, 0, 1] + 
      nablanablaB[0, 1, 0] - nablanablaB[1, 1, 1]), -(sinphi**3*nablanablaB[0, 1, 0]) + 
    cosphi**3*nablanablaB[1, 0, 1] + cosphi*sinphi**2*(nablanablaB[0, 0, 0] - 
      nablanablaB[0, 1, 1] - nablanablaB[1, 1, 0]) + 
    cosphi**2*sinphi*(nablanablaB[0, 0, 1] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), -(sinphi**2*nablanablaB[0, 1, 2]) + 
    cosphi**2*nablanablaB[1, 0, 2] + cosphi*sinphi*(nablanablaB[0, 0, 2] - 
      nablanablaB[1, 1, 2])], [-(sinphi**3*nablanablaB[0, 0, 1]) + 
    cosphi*sinphi**2*(nablanablaB[0, 0, 0] - nablanablaB[0, 1, 1] - 
      nablanablaB[1, 0, 1]) + cosphi**3*nablanablaB[1, 1, 0] + 
    cosphi**2*sinphi*(nablanablaB[0, 1, 0] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), sinphi**3*nablanablaB[0, 0, 0] + 
    cosphi*sinphi**2*(nablanablaB[0, 0, 1] + nablanablaB[0, 1, 0] + 
      nablanablaB[1, 0, 0]) + cosphi**2*sinphi*(nablanablaB[0, 1, 1] + 
      nablanablaB[1, 0, 1] + nablanablaB[1, 1, 0]) + cosphi**3*nablanablaB[1, 1, 1], 
   sinphi**2*nablanablaB[0, 0, 2] + cosphi*sinphi*(nablanablaB[0, 1, 2] + 
      nablanablaB[1, 0, 2]) + cosphi**2*nablanablaB[1, 1, 2]], 
  [-(sinphi**2*nablanablaB[0, 2, 1]) + cosphi**2*nablanablaB[1, 2, 0] + 
    cosphi*sinphi*(nablanablaB[0, 2, 0] - nablanablaB[1, 2, 1]), 
   sinphi**2*nablanablaB[0, 2, 0] + cosphi*sinphi*(nablanablaB[0, 2, 1] + 
      nablanablaB[1, 2, 0]) + cosphi**2*nablanablaB[1, 2, 1], 
   sinphi*nablanablaB[0, 2, 2] + cosphi*nablanablaB[1, 2, 2]]], 
 [[cosphi**2*nablanablaB[2, 0, 0] - cosphi*sinphi*(nablanablaB[2, 0, 1] + 
      nablanablaB[2, 1, 0]) + sinphi**2*nablanablaB[2, 1, 1], 
   cosphi**2*nablanablaB[2, 0, 1] - sinphi**2*nablanablaB[2, 1, 0] + 
    cosphi*sinphi*(nablanablaB[2, 0, 0] - nablanablaB[2, 1, 1]), 
   cosphi*nablanablaB[2, 0, 2] - sinphi*nablanablaB[2, 1, 2]], 
  [-(sinphi**2*nablanablaB[2, 0, 1]) + cosphi**2*nablanablaB[2, 1, 0] + 
    cosphi*sinphi*(nablanablaB[2, 0, 0] - nablanablaB[2, 1, 1]), 
   sinphi**2*nablanablaB[2, 0, 0] + cosphi*sinphi*(nablanablaB[2, 0, 1] + 
      nablanablaB[2, 1, 0]) + cosphi**2*nablanablaB[2, 1, 1], 
   sinphi*nablanablaB[2, 0, 2] + cosphi*nablanablaB[2, 1, 2]], 
  [cosphi*nablanablaB[2, 2, 0] - sinphi*nablanablaB[2, 2, 1], 
   sinphi*nablanablaB[2, 2, 0] + cosphi*nablanablaB[2, 2, 1], nablanablaB[2, 2, 2]]
      ]])

    return grad_grad_B_vector_cartesian
