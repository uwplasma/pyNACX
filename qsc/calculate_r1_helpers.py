"""
This module contains methods for performing mathematical operations in calculate_r1.py
"""
import jax.nump as jnp
 

"""
helpers for solve_sigma_equation
"""

def calc_iotaN(iota, helicity, nfp): 
  return iota + helicity * nfp 


"""
helpers for r1_diagnostics
"""

def calc_Y1s(sG, spsi, curvature, etabar):
  return sG * spsi * curvature / etabar 

def calc_Y1c(sG, spsi, curvature, sigma, etabar):
  return  sG * spsi * curvature * sigma / etabar 

def calc_angle(helicity, nfp, varphi): 
  return -helicity * nfp * varphi

def calc_X1s_untwisted(X1s, cosangle, X1c, sinangle): 
  return X1s * cosangle + X1c * sinangle

def calc_X1c_untwisted(X1s, sinangle, X1c, cosangle): 
  return X1s * (-sinangle) + X1c * cosangle 

def calc_Y1s_untwisted(Y1s, cosangle, Y1c, sinangle): 
  return Y1s * cosangle + Y1c * sinangle

def calc_Y1c_untwisted(Y1s, sinangle, Y1c, cosangle): 
  return Y1s * (-sinangle) + Y1c * cosangle

def calc_p(X1s, X1c, Y1s, Y1c): 
  return X1s * X1s + X1c * X1c + Y1s * Y1s + Y1c * Y1c

def calc_q(X1s, Y1c, X1c, Y1s): 
  return X1s * Y1c - X1c * Y1s

def calc_elongation(p, q): 
  return (p + jnp.sqrt(p * p - 4 * q * q)) / (2 * jnp.abs(q))

def derive_elongation(p=calc_p(),q=calc_q()): 
  return calc_elongation(p,q)

def calc_mean_elongation(elongation, d_l_d_phi): 
  return jnp.sum(elongation * d_l_d_phi) / jnp.sum(d_l_d_phi)

def derive_mean_elongation(d_l_d_phi, elongation = calc_elongation()): 
  return calc_mean_elongation(elongation, d_l_d_phi)
